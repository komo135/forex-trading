# Download Forex data
1. Download Metatrader5 from this [link](https://www.metatrader5.com/en/download) and install.
2. Verify that a demo account exists.
3. If the demo account does not exist, create a demo account.

https://user-images.githubusercontent.com/66017773/155930300-f6e2c949-72e7-4448-9925-3b7ccf837390.mp4

4. Go from Tools to Options (Ctr + o).
5. Go to the chart bar and change Max bars in chart to unlimited.


https://user-images.githubusercontent.com/66017773/155931351-b478b6d6-36cc-4ce1-8ee3-7c97e792d4ab.mp4


6. install Metatrader5 package
```console
$ pip install MetaTrader5
```
7. run generate_data.py
```console
$ cd data
$ python generate_data.py
```

### run agent
```console
$ python
```

```python
>>> from agent import dqn

>>> agent = dqn.Agent(model_name="efficientnet_b0", s=5, action_type=3, pip_scale=1, n=1, loss_cut=False, use_device="tpu", dueling=False)
>>> agent.run()
>>>
>>> agent.plot_result(w=self.best_w, risk=0.04, s=self.s)
```
![image](https://github.com/nagikomo/forex-trading/blob/main/image/FireShot%20Capture%20002%20-%20forex-trading_dqn.ipynb%20at%20main%20%C2%B7%20nagikomo_forex-trading%20-%20github.com.png)
