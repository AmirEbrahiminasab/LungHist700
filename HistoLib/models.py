import tensorflow as tf
from tfswin import SwinTransformerLarge224
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (
    ResNet50V2,
    EfficientNetB7,
    ConvNeXtBase
)
import keras_cv


def resnet_model(num_classes, input_shape):
    """
    Creates a model with a ResNet50V2 backbone.
    """
    inputs = layers.Input(input_shape)

    base_model = ResNet50V2(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)

    x = layers.Dense(256, activation='relu', name='prev_dense')(base_model.output)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, 'ResNet50V2'


def efficientnet_b7_model(num_classes, input_shape):
    """
    Creates a model with an EfficientNet-B7 backbone.
    """
    inputs = layers.Input(input_shape)
    base_model = EfficientNetB7(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)

    x = layers.Dense(256, activation='relu', name='prev_dense')(base_model.output)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, 'EfficientNetB7'


def convnext_model(num_classes, input_shape):
    """
    Creates a model with a ConvNeXt-Base backbone.
    """
    inputs = layers.Input(input_shape)

    base_model = ConvNeXtBase(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)

    x = layers.Dense(256, activation='relu', name='prev_dense')(base_model.output)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, 'ConvNeXt'


def vit_model(num_classes, input_shape):
    """
    Creates a model with a Vision Transformer (ViT) backbone using KerasCV.
    """
    inputs = layers.Input(shape=input_shape)

    base_model = keras_cv.models.VisionTransformer.from_preset(
        "vit_base_en_imagenet",
        load_weights=True
    )
    x = base_model(inputs)

    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, 'ViT_KerasCV'


def swin_model(num_classes, input_shape):
    """
    Creates a model with a Swin Transformer Large backbone using KerasCV.
    Fits on T4/P100 (16GB VRAM).
    """
    inputs = layers.Input(shape=input_shape)
    
    base_model = SwinTransformerLarge224(
        include_top=False,  
        pooling='avg'       
    )
    
    x = base_model(inputs)
    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'Swin_Transformer'


def get_model(generator, model_name='ResNet50'):
    """
    Retrieves a specified model by name, configured for the given data generator.
    """
    assert model_name in ['ResNet50', 'EfficientNetB7', 'ConvNeXt', 'ViT_KerasCV', 'Swin_KerasCV']

    num_classes = generator.num_classes
    input_shape = generator[0][0][0].shape

    print(f"Building model: {model_name}")
    print(f"    Input Shape: {input_shape}")
    print(f"    Number of Classes: {num_classes}")

    if model_name == 'ResNet50':
        return resnet_model(num_classes, input_shape)
    elif model_name == 'EfficientNetB7':
        return efficientnet_b7_model(num_classes, input_shape)
    elif model_name == 'ConvNeXt':
        return convnext_model(num_classes, input_shape)
    elif model_name == 'ViT_KerasCV':
        return vit_model(num_classes, input_shape)
    elif model_name == 'Swin_KerasCV':
        return swin_model(num_classes, input_shape)