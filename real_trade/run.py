import MetaTrader5 as mt5
import tensorflow as tf
import pandas as pd
import time
import numpy as np


def run(model_name, s, timeframe="h1"):
    """
    Parameters
    ----------
    model_name -> type str
    s -> type str or int
    timeframe -> m1 or m10 or m15 or m30 or h1 or h4 or d1
    -------

    """
    symbol = {0: "EURUSD", 1: "GBPUSD", 2: "NZDUSD", 3: "AUDUSD", 4: "EURGBP",
              5: "GBPJPY", 6: "USDJPY", 7: "EURJPY", 8: "AUDJPY", 9: "NZDJPY", 10: "USDCHF"}
    frame = {"m1": mt5.TIMEFRAME_M1, "m5": mt5.TIMEFRAME_M5, "m10": mt5.TIMEFRAME_M10, "m15": mt5.TIMEFRAME_M15,
             "m30": mt5.TIMEFRAME_M30, "h1": mt5.TIMEFRAME_H1, "h4": mt5.TIMEFRAME_H4, "d1": mt5.TIMEFRAME_D1}
    if isinstance(s, int):
        s = symbol[s]
    timeframe = frame[timeframe]

    model = tf.keras.models.load_model(model_name)

    init = mt5.initialize()
    assert init
    r = mt5.copy_rates_from_pos(s, timeframe, 0, 32)
    df = pd.DataFrame(r)
    last_time = df.time.values[-1]

    while True:
        r = mt5.copy_rates_from_pos(s, timeframe, 0, 32)
        df = pd.DataFrame(r)
        now_time = df.time.values[-1]
        if last_time == now_time:
            time.sleep(1)
        else:
            last_time = now_time
            sig = np.array((df.dig - df.sig.shift(1))[1:-1]).reshape((1, 30, 1))
            pred =
