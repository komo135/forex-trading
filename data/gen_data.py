import numpy as np
import pandas as pd
import sys
try:
  import MetaTrader5 as mt5
except:
  pip.main(['install', '--user', 'MetaTrader5'])
  import MetaTrader5 as mt5
try:
  import ta
  from ta.momentum import stoch
  from ta.trend import _ema as ema
except:
  pip.main(['install', '--user', 'ta'])
  import ta
  from ta.momentum import stoch
  from ta.trend import _ema as ema


def gen_data(symbol="EURUSD"):
    init = mt5.initialize()
    assert init == True
    x_list = []
    y_list = []
    atr = []  
    
    r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H1, 0, 5760 * 10)
    df = pd.DataFrame(r)
    
    point = mt5.symbol_info(s).point
    str_point = str(point)
    time = df["time"]
    if "e-" in str_point:
        point = int(str_point[-1])
        print(point)
        df *= 10 ** point
    else:
        point = len(str_point.rsplit(".")[1])
        if point == 1:
            point = 0
        df *= 10 ** point
    # df = np.round(df, 0)
    df["time"] = pd.to_datetime(time, unit='s')
    print(df["time"])
    df.index = df.time
    df.index.name = "Date"
    df = df[["open", "high", "low", "close", "tick_volume"]]


    df["sig"] = df["close"] - df["close"].shift(1)
    df = df.dropna()

    x = df["sig"]
    x = np.array(x)

    y = np.array(df[["close", "high", "low"]])
    atr_ = np.array(df[["atr"]])

    print("gen time series data")

    window_size = 30
    # window_size = 240
    time_x = []
    time_y = []
    time_atr = []

    for i in range(len(y)):
        if i > window_size:
            time_x.append(x[i - window_size: i])
            time_y.append(y[i - 1])
            time_atr.append(atr_[i - 1])

    x = np.array(time_x).reshape((-1, window_size, x.shape[-1]))
    y = np.array(time_y).reshape((-1, y.shape[-1]))
    atr = np.array(atr).reshape((-1,))

    return x, y, atr
