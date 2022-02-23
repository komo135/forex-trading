# Example
```python
from network import build_model

model = build_model("efficientnet_b0", (30, 1), 2)
model = build_model("efficientnet_b7", (30, 1), 2)
model = build_model("same_efficientnet_b0", (30, 1), 2)
model = build_model("convnext_b4", (30, 1), 2)
model = build_model("sam_se_convnext_b3, (30, 1), 2)
```
# Available networks
| model | sam | se | sam_se |
|:---:|:---:|:---:|:---:|
|efficientnet |Yes |No |No |
|lambda_efficientnet |Yes |No |No |
|dense_efficientnet |Yes |No |No |
|efficientnetv2 |Yes |No |No |
|resnet |Yes |Yes |Yes |
|densenet |Yes |Yes |Yes |
|lambda_resnet |Yes |Yes |Yes |
|convnext |Yes |Yes |Yes |
|lambda_convnext |Yes |Yes |Yes |
