# Example

### install package
```console
$ pip install ta
$ pip install MetaTrader5
```

### generate data
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
>>> self.plot_result(w=self.best_w, risk=0.04, s=self.s)
```
![image](https://github.com/nagikomo/forex-trading/blob/main/image/FireShot%20Capture%20002%20-%20forex-trading_dqn.ipynb%20at%20main%20%C2%B7%20nagikomo_forex-trading%20-%20github.com.png)
