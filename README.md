# DepthwiseConv1D
tensorflow DepthwiseConv1D

## Example
```python3
import numpy as np
from DepthwiseConv1D import DepthwiseConv1D

x = np.random.randn(1, 10, 3)
DepthwiseConv1D(3, padding="same")(x)
```
```python3
#output
<tf.Tensor: shape=(1, 10, 3), dtype=float32, numpy=
array([[[-2.85661072e-01,  1.35172045e+00,  5.36159202e-02],
        [-6.43872321e-02, -4.80441123e-01, -1.58486649e-01],
        [ 1.30057253e-03, -5.36365151e-01,  1.01073563e+00],
        [ 1.08433388e-01, -1.83498010e-01,  8.34330082e-01],
        [-3.70749176e-01, -1.05961430e+00,  1.68418014e+00],
        [ 7.05580041e-02,  2.58851439e-01,  1.23660095e-01],
        [ 6.99286997e-01,  9.52996671e-01,  1.89271018e-01],
        [ 3.09711039e-01, -6.87863350e-01,  3.15888785e-02],
        [-6.13673151e-01,  8.99892330e-01, -8.90250280e-02],
        [-3.10445458e-01,  2.13373974e-01,  1.22141886e+00]]],
      dtype=float32)>
```
