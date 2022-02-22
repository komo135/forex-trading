### save model weights
```python
agent.model.save("saved_model/efficientnet_b0_s0_H1")
```
### run script
```console
$ cd real_trade
$ python run.py efficientnet_b0_s0_H1 0 3 H1 0.01
```
### run.py argments
```python
"""
:param model_name: type str
:param s: type str or int
:param action_type: 1 or 2 or 3 or 4. 1->[buy, none]. 2->[none, sell]. 3->[buy, sell], 4->[buy, sell, none]
:param timeframe: m1 or m10 or m15 or m30 or h1 or h4 or d1
:param lot_size: 0.01 ~
"""
```
