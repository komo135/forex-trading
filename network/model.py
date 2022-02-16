from network.network import Model

# efficientnet
efficientnet_b0 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 0, efficientv1=True)
efficientnet_b1 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 1, efficientv1=True)
efficientnet_b2 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 2, efficientv1=True)
efficientnet_b3 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 3, efficientv1=True)
efficientnet_b4 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 4, efficientv1=True)
efficientnet_b5 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 5, efficientv1=True)
efficientnet_b6 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 6, efficientv1=True)
efficientnet_b7 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 7, efficientv1=True)

sam_efficientnet_b0 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 0, efficientv1=True, sam=True)
sam_efficientnet_b1 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 1, efficientv1=True, sam=True)
sam_efficientnet_b2 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 2, efficientv1=True, sam=True)
sam_efficientnet_b3 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 3, efficientv1=True, sam=True)
sam_efficientnet_b4 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 4, efficientv1=True, sam=True)
sam_efficientnet_b7 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 7, efficientv1=True, sam=True)

dense_efficientnet_b0 = lambda: Model([], 0, "DepthwiseConv1D", "densenet", 0, efficientv1=True)

lambda_efficientnet_b0 = lambda: Model([], 0, "lambdalayer", "resnet", 0, efficientv1=True)
lambda_efficientnet_b1 = lambda: Model([], 0, "lambdalayer", "resnet", 1, efficientv1=True)
lambda_efficientnet_b4 = lambda: Model([], 0, "lambdalayer", "resnet", 4, efficientv1=True)
lambda_efficientnet_b6 = lambda: Model([], 0, "lambdalayer", "resnet", 6, efficientv1=True)
lambda_efficientnet_b7 = lambda: Model([], 0, "lambdalayer", "resnet", 7, efficientv1=True)

sam_lambda_efficientnet_b2 = lambda: Model([], 0, "lambdalayer", "resnet", 2, efficientv1=True, sam=True)
sam_lambda_efficientnet_b3 = lambda: Model([], 0, "lambdalayer", "resnet", 3, efficientv1=True, sam=True)
sam_lambda_efficientnet_b4 = lambda: Model([], 0, "lambdalayer", "resnet", 4, efficientv1=True, sam=True)

# efficientnetv2
efficientnetv2_b0 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 0, efficientv2=True)
efficientnetv2_b2 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 2, efficientv2=True)
efficientnetv2_b4 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 4, efficientv2=True)
efficientnetv2_b7 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 7, efficientv2=True)

sam_efficientnetv2_b2 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 2, efficientv2=True, sam=True)
sam_efficientnetv2_b4 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 4, efficientv2=True, sam=True)
sam_efficientnetv2_b7 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 7, efficientv2=True, sam=True)

#efficientv3
efficientnetv3_b0 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 0, efficientv3=True)

# resnet
se_lambda_resnet_b0 = lambda: Model([], 48, "LambdaLayer", "resnet", 0, se=True)
se_lambda_resnet_b1 = lambda: Model([], 48, "LambdaLayer", "resnet", 1, se=True)

# convnext
se_convnext_b0 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 0, convnext=True, se=True)
se_convnext_b1 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 1, convnext=True, se=True)
se_convnext_b4 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 4, convnext=True, se=True)
se_convnext_b7 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 7, convnext=True, se=True)

sam_se_convnext_b0 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 0, convnext=True, se=True, sam=True)
sam_se_convnext_b2 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 2, convnext=True, se=True, sam=True)
sam_se_convnext_b4 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 4, convnext=True, se=True, sam=True)
sam_se_convnext_b7 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 7, convnext=True, se=True, sam=True)

cbam_convnext_b0 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 0, convnext=True, cbam=True)
cbam_convnext_b2 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 2, convnext=True, cbam=True)
cbam_convnext_b4 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 4, convnext=True, cbam=True)
cbam_convnext_b7 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 7, convnext=True, cbam=True)

sam_cbam_convnext_b2 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 2, convnext=True, cbam=True, sam=True)
sam_cbam_convnext_b4 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 4, convnext=True, cbam=True, sam=True)
sam_cbam_convnext_b7 = lambda: Model([], 48, "DepthwiseConv1D", "resnet", 7, convnext=True, cbam=True, sam=True)

se_lambda_convnext_b0 = lambda: Model([], 48, "LambdaLayer", "resnet", 0, convnext=True, se=True)
se_lambda_convnext_b1 = lambda: Model([], 48, "LambdaLayer", "resnet", 1, convnext=True, se=True)
se_lambda_convnext_b4 = lambda: Model([], 48, "LambdaLayer", "resnet", 4, convnext=True, se=True)
se_lambda_convnext_b7 = lambda: Model([], 48, "LambdaLayer", "resnet", 7, convnext=True, se=True)

sam_se_lambda_convnext_b0 = lambda: Model([], 48, "LambdaLayer", "resnet", 0, convnext=True, se=True, sam=True)
sam_se_lambda_convnext_b1 = lambda: Model([], 48, "LambdaLayer", "resnet", 1, convnext=True, se=True, sam=True)
sam_se_lambda_convnext_b2 = lambda: Model([], 48, "LambdaLayer", "resnet", 2, convnext=True, se=True, sam=True)
sam_se_lambda_convnext_b4 = lambda: Model([], 48, "LambdaLayer", "resnet", 4, convnext=True, se=True, sam=True)
sam_se_lambda_convnext_b7 = lambda: Model([], 48, "LambdaLayer", "resnet", 7, convnext=True, se=True, sam=True)


def build_model(model, input_shape: tuple, output_size: int) -> keras.Model:
    model = model()
    print(noise_ratio)
    model = model.build_model(input_shape, output_size, )

    return model
