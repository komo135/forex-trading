import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import ta

symbol = [
    "EURUSD", "GBPUSD", "NZDUSD", "AUDUSD", "EURGBP", "GBPJPY", "USDJPY", "EURJPY", "AUDJPY", "NZDJPY", "USDCHF"
]

"""
install Metatrader5 and create demo account(example XMTrading)

pip install MetaTrader5
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
                r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, 23040 * 10)
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
        df["atr"] = ta.volatility.average_true_range(df.high, df.low, df.close)
        df = df.dropna()

        lists = ["sig"]
        x = df[lists]
        x = np.array(x)
        x = np.clip(x / np.quantile(np.abs(x), 0.975, 0, keepdims=True), -1, 1)
        
        y = np.array(df[["close", "high", "low"]])
        atr_ = np.array(df[["atr"]])

        window_size = 30
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
