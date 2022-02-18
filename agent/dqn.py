import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
import os
from IPython.display import clear_output

from network import dqn_network

warnings.simplefilter('ignore')


class Agent:
    gamma = 0.99
    train_rewards = []
    test_rewards = []
    max_profits = -1e100
    max_pips = -1e100
    train_loss, val_loss = [], []
    test_pip, test_profit = [], []

    account_size = 100000
    spread = 10
    risk = 0.033

    def __init__(self, model_name, s, action_type=4, pip_scale=1, n=1, loss_cut=False, use_device="tpu", dueling=False):
        """
        :param model_name:
        :param s: 0~10
        :param action_type: 1 or 2 or 3 or 4. 1->[buy, none]. 2->[none, sell]. 3->[buy, sell], 4->[buy, sell, none]
        :param pip_scale:
        :param n:
        :param loss_cut: True or False
        :param use_device: tpu or cpu or gpu
        :param dueling: True or False
        """
        self.model_name = model_name
        self.s = s
        self.action_type = action_type
        self.pip_scale = pip_scale
        self.n = n
        self.loss_cut = loss_cut
        self.use_device = use_device
        self.dueling = dueling

        self.action_size = 3 if action_type == 4 else 2
        self.actions = {0: 1, 1: -1} if self.action_size == 2 else {0: 1, 1: -1, 2: 0}
        if action_type == 1:
            self.actions[1] = 0
        elif action_type == 2:
            self.actions[0] = 0

        np.random.seed(1010)

        if self.use_device == "tpu":
            try:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
                tf.config.experimental_connect_to_cluster(resolver)
                # This is the TPU initialization code that has to be at the beginning.
                tf.tpu.experimental.initialize_tpu_system(resolver)
                self.strategy = tf.distribute.TPUStrategy(resolver)
            except:
                self.use_device = "cpu"
        elif self.use_device == "gpu":
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

        self.env()

        if self.model_name is not None:
            self.train_data()
            self.build_model()
            self.ind = np.arange(len(self.returns[0]))
            np.random.shuffle(self.ind)

    def env(self):
        self.x = np.load("rl/data/x.npy")
        x_ = [2, 3, 4]
        print(x_)
        self.x = self.x[:, :, :, x_]

        for s in range(self.x.shape[0]):
            for i in range(self.x.shape[-1]):
                self.x[s, :, :, i] /= np.quantile(np.abs(self.x[s, :, :, i]), 0.99)

        self.x = np.clip(self.x, -1, 1)

        y = np.load("rl/data/target.npy")
        self.low = y[:, :, 2].reshape((self.x.shape[0], -1))
        self.high = y[:, :, 1].reshape((self.x.shape[0], -1))
        self.y = y[:, :, 0].reshape((self.x.shape[0], -1))

        self.atr = np.load("rl/data/atr.npy").reshape((self.x.shape[0], -1)).astype(np.int32)

        self.train_step = np.arange(0, int(self.x.shape[1] * 0.9))
        self.test_step = np.arange(self.train_step[-1], int(self.x.shape[1]))

    def loss_function(self):
        def loss(q_backup, q):
            k = 2

            error = q_backup - q
            loss = tf.where(tf.abs(error) <= k, error ** 2 * 0.5, 0.5 * k ** 2 + k * (tf.abs(error) - k))
            loss = tf.reduce_mean(loss)

            return loss

        return loss

    def _build_model(self, lr):
        loss = self.loss_function()

        if self.dueling:
            dqn_network.dueling = True
        model = dqn_network.network.build_model(self.model_name, self.x.shape[-2:], self.action_size)

        model.compile(
            tf.keras.optimizers.Adam(lr, clipnorm=1.), loss=loss, steps_per_execution=100
        )

        return model

    def build_model(self, lr=1e-4):
        if self.use_device == "tpu":
            with self.strategy.scope():
                self.model = self._build_model(lr)
                self.target_model = tf.keras.models.clone_model(self.model)
                self.target_model.set_weights(self.model.get_weights())
        else:
            self.model = self._build_model(lr)
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

        self.model.summary()

    def train_data(self):
        states, returns, new_states, old_actions = [], [], [], []
        h, h_ = 0, self.train_step[-1]
        n = self.n
        s = self.s
        for s in range(self.x.shape[0]):
            df = self.x[s, h:h_ - n].copy()
            trend = self.y[s, h:h_]

            buy = np.array([trend[i + n] - trend[i] for i in range(len(trend) - n)]).reshape((-1,))
            scale = np.quantile(abs(buy), 0.99) / 1
            buy = np.clip(buy / scale, -1, 1)
            spread = int((np.quantile(self.atr[s], 0.25)) / scale)
            spread = np.clip(spread, 0.02, None) * self.pip_scale

            buy *= self.pip_scale
            sell = -buy

            pip = np.zeros((len(trend) - n, self.action_size, self.action_size))

            b = 0 if self.action_type == 2 else 1
            s = 0 if self.action_type == 1 else 1

            pip[:, 0, 0] = buy * b
            pip[:, 0, 1] = (sell - spread) * s
            pip[:, 1, 0] = (buy - spread) * b
            pip[:, 1, 1] = sell * s
            if self.action_size == 3:
                pip[:, 2, 0] = buy - spread
                pip[:, 2, 1] = sell - spread

            states.append(df[:-self.n])
            returns.append(pip[:-self.n])
            new_states.append(np.roll(df, -self.n, axis=0)[:-self.n])

        concat = np.concatenate
        self.states, self.returns, self.new_states = \
            np.array(states), np.array(returns), np.array(new_states)
        self.returns = np.round(self.returns, 2).astype(np.float32)

    def get_actions(self, df):
        q = self.model.predict(df, 102800, workers=10000, use_multiprocessing=True)
        actions = np.argmax(q, -1)
        a = np.argmax([q[0, 0, 1], q[0, 1, 0]])
        act = [a]

        for i in range(1, len(actions)):
            a = actions[i, a]
            act.append(a)

        return act, actions

    def get_q(self, df):
        return self.model.predict(df, 102800, workers=10000, use_multiprocessing=True)

    def trade(self, s: int, h, h_, loss_cut_=False, train=False, plot=False):
        """
        returns :
         train=False, plot=False : profit, profits, total_pip, pips\n
         train=True, plot=False : profit, actions, q\n
         train=False, plot=True : profit, profits, total_pip, pips, buy, sell\n
        """

        profit, total_pip, trend, buy, sell, loss_cuts, none = [0 for _ in range(7)]
        loss_cut_ = self.loss_cut

        all_pip = np.zeros((h_ - h, self.x.shape[0]))
        all_profit = np.zeros((h_ - h, self.x.shape[0]))
        all_draw_down = np.zeros((h_ - h, self.x.shape[0]))

        s_ = s

        for s in ([s] if isinstance(s, int) else s if s is not None else range(self.x.shape[0])):
            df = self.x[s, h:h_]
            trend = self.y[s, h:h_]
            high = self.high[s, h:h_]
            low = self.low[s, h:h_]
            atr = self.atr[s, h:h_]

            actions, actions_ = self.get_actions(df)
            actions = np.array(actions)
            # actions = np.array([self.actions[a] for a in actions])

            profit = self.account_size
            self.max_profit = self.account_size
            self.max_pip = 0
            total_pip = 0
            pips = []
            profits = []
            buy, sell, loss_cuts, none = [], [], [], []
            draw_down = []
            dd = []
            plus_pip = []
            minus_pip = []
            max_pip = 0
            if plot:
                if actions[0] == 0:
                    buy.append(0)
                elif actions[0] == 1:
                    sell.append(0)
                elif actions[0] == 2:
                    none.append(0)

            old_a = actions[0]
            old_price = trend[0]

            loss_cut = -atr[0] * 2
            position_size = int((profit * self.risk) / -loss_cut)
            position_size = np.minimum(position_size, 500 * 200 * 100)
            position_size = np.maximum(position_size, 1)

            end = False
            for i, (price, atr) in enumerate(zip(trend, atr)):
                act = self.actions[actions[i]]
                if old_a != act:
                    if not end and old_a != 0:
                        pip = (price - old_price) * old_a
                        total_pip += pip - self.spread
                        pips.append(pip - self.spread)
                        all_pip[i, s] = pip - self.spread

                        if old_a == 1:
                            plus_pip.append(pip - self.spread)
                        elif old_a == -1:
                            minus_pip.append(pip - self.spread)

                        if train:
                            if max_pip < total_pip:
                                if dd:
                                    draw_down.append(min(dd))
                                    all_draw_down[i, s] = min(dd)
                                    dd = []
                                max_pip = total_pip
                            if max_pip > total_pip:
                                dd.append(total_pip - max_pip)

                        gain = pip * position_size - self.spread * position_size
                        profit += gain
                        profits.append(gain)
                        all_profit[i, s] = gain

                        self.max_pip = np.maximum(self.max_pip, total_pip)
                        self.max_profit = np.maximum(self.max_profit, profit)

                    old_price = price
                    loss_cut = -atr * 2

                    # position_size = int(profit / (price / 500) * self.risk)

                    position_size = int((profit * self.risk) / -loss_cut)
                    position_size = np.minimum(position_size, 500 * 200 * 100)
                    position_size = np.maximum(position_size, 0)

                    if plot:
                        if act == 1:
                            buy.append(i)
                        elif act == -1:
                            sell.append(i)
                        elif act == 0:
                            none.append(i)

                    end = False

                if loss_cut_ and not end:
                    try:
                        pip = low[i + 1] - old_price if act == 1 else old_price - high[i + 1]

                        end = (pip - self.spread) <= loss_cut
                        if end:
                            pip = loss_cut
                            total_pip += pip
                            pips.append(pip)

                            gain = pip * position_size
                            profit += gain
                            profits.append(gain)
                            loss_cuts.append(i)
                            actions[i + 1] = 2
                            if self.action_size == 3 and actions_ is not None:
                                a = 2
                                for i_ in range(i + 2, len(actions)):
                                    a = actions_[i, a]
                                    actions[i] = a
                    except:
                        pass

                old_a = act

            if train and dd:
                draw_down.append(min(dd))
                all_draw_down[-1, s] = min(dd)

        pips = np.sum(all_pip, axis=-1)
        profits = np.sum(all_profit, axis=-1)
        draw_down = np.sum(all_draw_down, axis=-1)

        profits = profits[pips != 0]
        pips = pips[pips != 0]
        draw_down = draw_down[draw_down != 0]

        if train:
            r = pips, profits, draw_down, plus_pip, minus_pip
            return r
        else:
            r = (profit, profits, total_pip, pips)
            if plot:
                r += (trend, buy, sell, loss_cuts, none)
            return r

    def evolute(self, s, h, h_):
        pips, profits, draw_down, plus_pip, minus_pip = self.trade(s, h, h_, train=True)

        acc = np.mean(pips > 0)
        total_win = np.sum(pips[pips > 0])
        total_lose = np.sum(pips[pips < 0])
        rr = total_win / abs(total_lose)
        ev = (np.mean(pips[pips > 0]) * acc + np.mean(pips[pips < 0]) * (1 - acc)) / abs(np.mean(pips[pips < 0]))
        mean_dd = np.mean(draw_down)
        min_dd = np.min(draw_down)

        pip_ = [0]
        profit_ = [self.account_size]
        total_pip = 0
        total_profit = self.account_size
        for pip, profit in zip(pips, profits):
            total_pip += pip
            total_profit += profit

            pip_.append(total_pip)
            profit_.append(total_profit)

        plt.figure(figsize=(10, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(pip_)
        plt.subplot(1, 2, 2)
        plt.plot(profit_)
        plt.show()

        print(
            f"acc = {acc}, pips = {sum(pips)}\n"
            f"total_win = {total_win}, total_lose = {total_lose}\n"
            f"rr = {rr}, ev = {ev}\n"
            f"mean draw down = {mean_dd}, min draw down = {min_dd}\n"
            f"total plus pip = {sum(plus_pip)}, total minus_pip = {sum(minus_pip)}"
        )

    def target_q(self, returns, target_q, target_a):
        if self.train_loss:
            target_a = np.argmax(target_a, -1)
            rr = range(len(returns))
            returns[:, 0, 0] += self.gamma * target_q[rr, 0, target_a[rr, 0]]
            returns[:, 0, 1] += self.gamma * target_q[rr, 1, target_a[rr, 1]]
            returns[:, 1, 0] += self.gamma * target_q[rr, 0, target_a[rr, 0]]
            returns[:, 1, 1] += self.gamma * target_q[rr, 1, target_a[rr, 1]]
            if self.action_size == 3:
                returns[:, 0, 2] += self.gamma * target_q[rr, 2, target_a[rr, 2]]
                returns[:, 1, 2] += self.gamma * target_q[rr, 2, target_a[rr, 2]]
                returns[:, 2, 0] += self.gamma * target_q[rr, 0, target_a[rr, 0]]
                returns[:, 2, 1] += self.gamma * target_q[rr, 1, target_a[rr, 1]]
                returns[:, 2, 2] += self.gamma * target_q[rr, 2, target_a[rr, 2]]

        assert np.mean(np.isnan(returns) == False) == 1

        return returns

    def _train(self, epoch=100, s=0, batch_size=2056):
        assert isinstance(s, int)

        ind = self.ind

        states, new_states, returns = \
            self.states[s][ind].copy(), self.new_states[s][ind].copy(), self.returns[s][ind].copy()

        if self.train_loss:
            target_q = self.target_model.predict(new_states, 102800)
        else:
            target_q = np.zeros((len(returns), self.action_size, self.action_size), np.float32)

        sp = 10000# split
        for _ in range(epoch):
            returns = self.returns[s][ind].copy()
            noise = np.random.randn(*states.shape) * 0.1

            target_a = self.model.predict(new_states + noise, 102800)
            returns = self.target_q(returns, target_q, target_a)

            self.model.fit(states[-sp:] + noise[-sp:], returns[-sp:], batch_size, 1,)
            h = self.model.fit(states[:-sp] + noise[:-sp], returns[:-sp], batch_size, 1, validation_split=0.2, verbose=0)
            self.train_loss.extend(h.history["loss"])
            self.val_loss.extend(h.history["val_loss"])

            if len(self.train_loss) >= 200:

                pips, profits, _, _, _ = self.trade(s, self.train_step[-1]-10000, self.train_step[-1], train=True)
                self.train_rewards.append(np.sum(pips))
                pips, profits, _, _, _ = self.trade(s, self.test_step[0], self.test_step[0] + 960 * 9, train=True)
                self.test_rewards.append(np.sum(pips))
                
                self.max_profit /= self.account_size

                self.test_pip.append(self.max_pip)
                self.test_profit.append(self.max_profit)

                if self.max_pips <= self.max_pip:
                    self.best_w = self.model.get_weights()
                    self.max_profits = self.max_profit

                self.max_profits = np.maximum(self.max_profit, self.max_profits)
                self.max_pips = np.maximum(self.max_pip, self.max_pips)

                plt.figure(figsize=(20, 5), dpi=100)
                plt.subplot(1, 2, 1)
                plt.plot(self.train_rewards)
                plt.subplot(1, 2, 2)
                plt.plot(self.test_rewards)
                plt.show()

                print(f"profits = {self.max_profit}, max profits = {self.max_profits}\n"
                      f"pips = {self.max_pip}, max pip = {self.max_pips}")

    def train(self, epoch1=40, epoch2=15, batch_size=2056, agent_name="dqn", risk=.04):
        self.risk = risk
        for _ in range(epoch2):
            clear_output()
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_loss)
            plt.plot(self.val_loss)
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
            self._train(epoch1, self.s, batch_size)
            self.target_model.set_weights(self.model.get_weights())
            
    def plot_trade(self, s, train=False, test=False, period=1):
        assert train or test
        h = 0
        if test:
            h = self.test_step[0]
        elif train:
            h = np.random.randint(0, int(self.train_step[-1] - 960 * period))
        h_ = h + 960 * period

        profit, profits, total_pip, pips, trend, buy, sell, loss_cuts, none = self.trade(s, h, h_, plot=True)

        plt.figure(figsize=(20, 10), dpi=100)
        plt.plot(trend, color="g", alpha=1, label="close")
        plt.plot(trend, "^", markevery=buy, c="red", label='buy', alpha=0.7)
        plt.plot(trend, "v", markevery=sell, c="blue", label='sell', alpha=0.7)
        plt.plot(trend, "D", markevery=loss_cuts, c="y", label='stop loss', alpha=0.7)
        plt.plot(trend, "s", markevery=none, c="g", label='none', alpha=0.7)

        plt.legend()
        plt.show()

        print(f"pip = {np.sum(pips)}"
              f"\naccount size = {profit}"
              f"\ngrowth rate = {profit / self.account_size}"
              f"\naccuracy = {np.mean(np.array(pips) > 0)}")

    def plot_result(self, w, risk=0.1, s=None):
        self.model.set_weights(w)
        s = self.s if s is None else s
        self.risk = risk

        plt.figure(figsize=(20, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(self.test_pip)
        plt.subplot(1, 2, 2)
        plt.plot(self.test_profit)
        plt.show()

        ################################################################################
        self.plot_trade(s, train=False, test=True, period=9)
        ################################################################################
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        len_ = len(self.train_loss) // 2
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss[len_:])
        plt.plot(self.val_loss[len_:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        ################################################################################
        self.loss_cut = False
        self.evolute(s, self.test_step[0], self.test_step[-1])
        # self.loss_cut = True
        # self.evolute(s, self.test_step[0], self.test_step[-1])
        ################################################################################
        plt.figure(figsize=(20, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(self.train_rewards)
        plt.subplot(1, 2, 2)
        plt.plot(self.test_rewards)
        plt.show()

        print(f"profits = {self.max_profit}, max profits = {self.max_profits}\n"
              f"pips = {self.max_pip}, max pip = {self.max_pips}")
        ################################################################################
        self.evolute(s, self.test_step[0] - 11513, self.test_step[0])
