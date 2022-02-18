import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import ta
from ta.momentum import stoch
from ta.trend import _ema as ema

symbol = [
    "EURUSD", "GBPUSD", "NZDUSD", "AUDUSD", "EURGBP", "GBPJPY", "USDJPY", "EURJPY", "AUDJPY", "NZDJPY", "USDCHF"
]

"""
install Metatrader5 and create demo account(example XMTrading)

pip install MetaTrader5
pip install ta
"""

def gen_data(symbol=symbol):
    init = mt5.initialize()
    assert init == True
    x_list = []
    y_list = []
    atr = []

    s_ = -1

    for s in symbol:
        s_ += 1
        while True:
            try:
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, 69120 * 10)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M10, 0, 34560 * 10)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, 23040 * 10)
                r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M30, 0, 11520 * 10)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H1, 0, 5760 * 10)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H4, 0, 1440 * 10)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_D1, 0, 240 * 10)
                df = pd.DataFrame(r)
                df.close
                break
            except:
                # pass
                print(1)
        try:
            print(s)
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
        except:
            pass

        df["sig"] = df["close"] - df["close"].shift(1)
        df = df.dropna()

        lists = ["sig"]
        x = df[lists]
        x = np.array(x)
        
        y = np.array(df[["close", "high", "low"]])
        atr_ = np.array(df[["atr"]])

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

        x_list.append(x)
        y_list.append(y)
        atr.append(time_atr)

    np.save(f"x", np.array(x_list).astype(np.float32))
    np.save(f"target", np.array(y_list).astype(np.float32))
    np.save(f"atr", np.array(atr).astype(np.float32))
    

if __name__ == "__main__":
    gen_data()
