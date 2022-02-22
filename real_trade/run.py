import time

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import tensorflow as tf


def run(model_name, s, action_type, timeframe="h1", lot_size=0.01):
    """
    :param model_name: type str
    :param s: type str or int
    :param action_type: 1 or 2 or 3 or 4. 1->[buy, none]. 2->[none, sell]. 3->[buy, sell], 4->[buy, sell, none]
    :param timeframe: m1 or m10 or m15 or m30 or h1 or h4 or d1
    :param lot_size: 0.01 ~
    """
    symbol = {0: "EURUSD", 1: "GBPUSD", 2: "NZDUSD", 3: "AUDUSD", 4: "EURGBP",
              5: "GBPJPY", 6: "USDJPY", 7: "EURJPY", 8: "AUDJPY", 9: "NZDJPY", 10: "USDCHF"}
    frame = {"m1": mt5.TIMEFRAME_M1, "m5": mt5.TIMEFRAME_M5, "m10": mt5.TIMEFRAME_M10, "m15": mt5.TIMEFRAME_M15,
             "m30": mt5.TIMEFRAME_M30, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4, "d1": mt5.TIMEFRAME_D1}
    period = {"m1": 3456000, "m5": 691200, "m10": 345600, "m15": 230400, "m30": 115200,
              "h1": 57600, "h4": 14400, "d1": 2400}
    if isinstance(s, int):
        s = symbol[s]
    timeframe = frame[timeframe]

    action_size = 3 if action_type == 4 else 2
    actions = {0: 1, 1: -1} if action_size == 2 else {0: 1, 1: -1, 2: 0}
    if action_type == 1:
        actions[1] = 0
    elif action_type == 2:
        actions[0] = 0

    model = tf.keras.models.load_model("saved_model/" + model_name)

    init = mt5.initialize()
    assert init
    r = mt5.copy_rates_from_pos(s, timeframe, 0, period)
    df = pd.DataFrame(r)
    scaling = np.max(np.array((df.dig - df.sig.shift(1))[1:-1]))
    last_time = df.time.values[-1]

    old_action = None

    while True:
        r = mt5.copy_rates_from_pos(s, timeframe, 0, 32)
        df = pd.DataFrame(r)
        now_time = df.time.values[-1]
        if last_time == now_time:
            time.sleep(1)
        else:
            last_time = now_time
            sig = (np.array((df.dig - df.sig.shift(1))[1:-1]).reshape((1, 30, 1)) / scaling).astype(np.float32)
            act = model(sig)[0]
            if old_action is None:
                act = np.argmax([act[0, 1], act[1, 0]])
            else:
                act = np.argmax(act, axis=0)[old_action]

            if act != old_action:
                mt5.Close(s)

                a = actions[act]
                if a == 1:
                    mt5.Buy(s, lot_size)
                elif a == -1:
                    mt5.Sell(s, lot_size)

            old_action = act


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dqn agent real trade")
    parser.add_argument("model_name", type=str, help="model name, type str")
    parser.add_argument("s", type=int, help="symbol -> 0 ~ 10\n"
                                            '{0: "EURUSD", 1: "GBPUSD", 2: "NZDUSD", 3: "AUDUSD", 4: "EURGBP", '
                                            '5: "GBPJPY", 6: "USDJPY", 7: "EURJPY", 8: "AUDJPY", 9: "NZDJPY", '
                                            '10: "USDCHF"}')
    parser.add_argument("action_type", type=int, help="1->[buy, none]. 2->[none, sell]. 3->[buy, sell], 4->[buy, "
                                                      "sell, none]")
    parser.add_argument("timeframe", type=str, help="m1 or m10 or m15 or m30 or h1 or h4 or d1")
    parser.add_argument("lot_size", type=float, help="0.01~")
    args = parser.parse_args()

    run(args.model_name, args.s, args.action_type, args.timeframe, args.lot_size)
