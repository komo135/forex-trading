# Example
```python
from network import build_model

model = build_model("efficientnet_b0", input_shape=(30, 1), output_size=2)
model = build_model("efficientnet_b7", (30, 1), 2)
model = build_model("same_efficientnet_b0", (30, 1), 2)
model = build_model("convnext_b4", (30, 1), 2)
model = build_model("sam_se_convnext_b3, (30, 1), 2)

model.compile("adam", "mse", ["mae"])
```
# Available networks
| model | sam | se | sam_se | b0~b7 |
|:---:|:---:|:---:|:---:|:---:|
|efficientnet |Yes |No |No |Yes |
|lambda_efficientnet |Yes |No |No |Yes |
|dense_efficientnet |Yes |No |No |Yes |
|efficientnetv2 |Yes |No |No |Yes |
|resnet |Yes |Yes |Yes |Yes |
|densenet |Yes |Yes |Yes |Yes |
|lambda_resnet |Yes |Yes |Yes |Yes |
|convnext |Yes |Yes |Yes |Yes |
|lambda_convnext |Yes |Yes |Yes |Yes |
