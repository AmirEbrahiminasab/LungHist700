import tensorflow as tf
from tfswin import SwinTransformerLarge224
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (
    ResNet50V2,
    EfficientNetB3,
    ConvNeXtBase
)
import keras_cv


def resnet_model(num_classes, input_shape):
    """
    Creates a model with a ResNet50V2 backbone.
    """
    inputs = layers.Input(input_shape)
    # GENERATOR GIVES 0...1. ResNetV2 wants -1...1
    # Formula: (x * 255) -> preprocess -> or simple (x - 0.5) * 2
    inputs = layers.Rescaling(scale=1./0.5, offset=-1)(inputs)

    base_model = ResNet50V2(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)

    x = layers.Dense(256, activation='relu', name='prev_dense')(base_model.output)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, 'ResNet50V2'


def efficientnet_b3_model(num_classes, input_shape):
    """
    Creates a model with an EfficientNet-B3 backbone.
    """
    inputs = layers.Input(input_shape)

    base_model = EfficientNetB3(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)

    x = layers.Dense(256, activation='relu', name='prev_dense')(base_model.output)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, 'EfficientNetB3'


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
    inputs = layers.Resizing(224, 224, interpolation='bilinear')(inputs)
    inputs = layers.Rescaling(scale=255.0)(inputs)

    inputs = layers.Lambda(lambda x: tf.cast(x, tf.uint8), name='to_uint8')(inputs)
    
    base_model = SwinTransformerLarge224(
        include_top=False,  
        pooling='avg'       
    )
    
    x = base_model(inputs)
    x = layers.Dense(128, activation='relu', name='prev_dense')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.7, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'Swin_KerasCV'


def get_model(generator, model_name='ResNet50'):
    """
    Retrieves a specified model by name, configured for the given data generator.
    """
    assert model_name in ['ResNet50', 'EfficientNetB3', 'ConvNeXt', 'ViT_KerasCV', 'Swin_KerasCV']

    num_classes = generator.num_classes
    input_shape = generator[0][0][0].shape

    print(f"Building model: {model_name}")
    print(f"    Input Shape: {input_shape}")
    print(f"    Number of Classes: {num_classes}")

    if model_name == 'ResNet50':
        return resnet_model(num_classes, input_shape)
    elif model_name == 'EfficientNetB3':
        return efficientnet_b3_model(num_classes, input_shape)
    elif model_name == 'ConvNeXt':
        return convnext_model(num_classes, input_shape)
    elif model_name == 'ViT_KerasCV':
        return vit_model(num_classes, input_shape)
    elif model_name == 'Swin_KerasCV':
        return swin_model(num_classes, input_shape)