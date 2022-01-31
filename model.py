from network import Model

efficientnet_b0 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 0, efficientv1=True)
efficientnet_b1 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 1, efficientv1=True)
efficientnet_b2 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 2, efficientv1=True)
efficientnet_b3 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 3, efficientv1=True)
efficientnet_b4 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 4, efficientv1=True)
efficientnet_b5 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 5, efficientv1=True)
efficientnet_b6 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 6, efficientv1=True)
efficientnet_b7 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 7, efficientv1=True)

efficientnetv2_b0 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 0, efficientv2=True)
efficientnetv2_b1 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 1, efficientv2=True)

sam_efficientnetv2_b2 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 2, efficientv2=True, sam=True)
sam_efficientnetv2_b3 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 3, efficientv2=True, sam=True)
sam_efficientnetv2_b4 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 4, efficientv2=True, sam=True)

sam_efficientnet_b0 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 0, efficientv1=True, sam=True)
sam_efficientnet_b1 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 1, efficientv1=True, sam=True)
sam_efficientnet_b2 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 2, efficientv1=True, sam=True)
sam_efficientnet_b3 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 3, efficientv1=True, sam=True)
sam_efficientnet_b4 = lambda: Model([], 0, "DepthwiseConv1D", "resnet", 4, efficientv1=True, sam=True)

lambda_efficientnet_b0 = lambda: Model([], 0, "lambdalayer", "resnet", 0, efficientv1=True)
lambda_efficientnet_b1 = lambda: Model([], 0, "lambdalayer", "resnet", 1, efficientv1=True)

sam_lambda_efficientnet_b2 = lambda: Model([], 0, "lambdalayer", "resnet", 2, efficientv1=True, sam=True)
sam_lambda_efficientnet_b3 = lambda: Model([], 0, "lambdalayer", "resnet", 3, efficientv1=True, sam=True)
sam_lambda_efficientnet_b4 = lambda: Model([], 0, "lambdalayer", "resnet", 4, efficientv1=True, sam=True)

dense_efficientnet_b0 = lambda: Model([], 0, "DepthwiseConv1D", "densenet", 0, efficientv1=True)

lambda_densenet_b0 = lambda: Model([], 32, "LambdaLayer", "densenet", 0)

se_lambda_densenet_b0 = lambda: Model([], 32, "LambdaLayer", "densenet", 0, se=True)
se_lambda_densenet_b1 = lambda: Model([], 32, "LambdaLayer", "densenet", 1, se=True)
se_lambda_densenet_b2 = lambda: Model([], 32, "LambdaLayer", "densenet", 2, se=True)
se_lambda_densenet_b3 = lambda: Model([], 32, "LambdaLayer", "densenet", 3, se=True)
se_lambda_densenet_b4 = lambda: Model([], 32, "LambdaLayer", "densenet", 4, se=True)

se_lambda_resnet_b0 = lambda: Model([], 128, "LambdaLayer", "resnet", 0, se=True)
se_lambda_resnet_b1 = lambda: Model([], 128, "LambdaLayer", "resnet", 1, se=True)
se_lambda_resnet_b2 = lambda: Model([], 128, "LambdaLayer", "resnet", 2, se=True)
se_lambda_resnet_b3 = lambda: Model([], 128, "LambdaLayer", "resnet", 3, se=True)

convnext_b0 = lambda: Model([], 32, "DepthwiseConv1D", "resnet", 0, convnext=True)

sam_convnext_b2 = lambda: Model([], 32, "DepthwiseConv1D", "resnet", 2, convnext=True, sam=True)
sam_convnext_b3 = lambda: Model([], 32, "DepthwiseConv1D", "resnet", 3, convnext=True, sam=True)
sam_convnext_b4 = lambda: Model([], 32, "DepthwiseConv1D", "resnet", 4, convnext=True, sam=True)


def build_model(model_name, input_shape: tuple, output_size: int) -> tf.keras.Model:
    model = model_name()
    print(noise_ratio)
    model = model.build_model(input_shape, output_size, )

    return model
