import warnings

import numpy as np
import tensorflow as tf
from agent.dqn import Agent as dqn

warnings.simplefilter('ignore')

k = 1
plus_tau = np.arange(1, 33, dtype=np.float32) / 33
plus_tau = np.reshape(plus_tau, (1, 1, 32, 1))
minus_tau = np.abs(plus_tau - 1)


class Agent(dqn):
    def loss_function(self):
        def loss(q_backup, q):

            error = q_backup - q
            loss = tf.where(tf.abs(error) <= k, error ** 2 * 0.5, 0.5 * k ** 2 + k * (tf.abs(error) - k))
            loss = tf.where(error > 0, loss * plus_tau, loss * minus_tau)
            loss = tf.reduce_mean(loss, (0, 1, 3))
            loss = tf.reduce_sum(loss)
            return loss

        return loss

    def _build_model(self, lr=1e-4):
        tf.random.set_seed(1010)
        loss = self.loss_function()

        from network import qrdqn_network
        model = qrdqn_network.network.build_model(self.model_name, self.x.shape[-2:], self.action_size)

        model.compile(
            tf.keras.optimizers.Adam(lr, clipnorm=1.), loss=loss, steps_per_execution=100
        )

        return model

    def get_actions(self, df):
        q = self.model.predict(df, 10280, workers=10000, use_multiprocessing=True)
        q = np.mean(q, 2)
        actions = np.argmax(q, -1)
        if self.action_size == 3:
            a = np.argmax(q[0, 2])
        else:
            a = np.argmax([q[0, 0, 1], q[0, 1, 0]])
        act = [a]

        for i in range(1, len(actions)):
            a = actions[i, a]
            act.append(a)

        return act, actions

    def get_q(self, df):
        return np.mean(super(Agent, self).get_q(df), axis=2)

    def target_q(self, returns, target_q, target_a):
        returns = np.reshape(returns, (-1, self.action_size, 1, self.action_size))
        returns = np.tile(returns, (1, 1, 32, 1))

        if self.train_loss and target_q.shape == returns.shape:
            target_a = np.argmax(np.mean(target_a, axis=2), -1)
            rr = range(len(returns))
            returns[:, 0, :, 0] += self.gamma * target_q[rr, 0, :, target_a[:, 0]]
            returns[:, 0, :, 1] += self.gamma * target_q[rr, 1, :, target_a[:, 1]]
            returns[:, 1, :, 0] += self.gamma * target_q[rr, 0, :, target_a[:, 0]]
            returns[:, 1, :, 1] += self.gamma * target_q[rr, 1, :, target_a[:, 1]]
            if self.action_size == 3:
                returns[:, 0, :, 2] += self.gamma * target_q[rr, 2, :, target_a[:, 2]]
                returns[:, 1, :, 2] += self.gamma * target_q[rr, 2, :, target_a[:, 2]]
                returns[:, 2, :, 0] += self.gamma * target_q[rr, 0, :, target_a[:, 0]]
                returns[:, 2, :, 1] += self.gamma * target_q[rr, 1, :, target_a[:, 1]]
                returns[:, 2, :, 2] += self.gamma * target_q[rr, 2, :, target_a[:, 2]]

        assert np.mean(np.isnan(returns) == False) == 1

        return returns

    def train(self, epoch1=50, epoch2=12, batch_size=2056, save=True, agent_name="qrdqn", risk=.04):
        super(Agent, self).train(epoch1, epoch2, batch_size, save, agent_name, risk)
