# Example

### install package
```console
pip install ta
pip install MetaTrader5
```

### generate data
```console
cd data
python generate_data.py
```

### run agent
```python
from agent import dqn

agent = dqn.Agent(model_name="efficientnet_b0", s=0, action_type=4, pip_scale=1, n=1, loss_cut=False, use_device="tpu", dueling=False)
agent.run()
```
