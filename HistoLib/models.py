import tensorflow as tf
import numpy as np
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
    Uses 'nested model' pattern to ensure GradCAM compatibility.
    """
    inputs = layers.Input(input_shape)
    # ResNetV2 expects inputs in [-1, 1]. Generator provides [0, 1].
    x = layers.Rescaling(scale=1./0.5, offset=-1)(inputs)

    # Note: input_tensor is NOT used. We call base_model(x) later.
    base_model = ResNet50V2(
        include_top=False, 
        weights='imagenet', 
        pooling=None, 
        input_shape=input_shape
    )
    # This encapsulates the backbone as a single layer in the graph
    x = base_model(x)
    
    # Manual pooling to ensure feature maps are accessible before this step
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'ResNet50V2'


def efficientnet_b3_model(num_classes, input_shape):
    inputs = layers.Input(input_shape)
    
    # EfficientNet expects [0, 255] usually, but if your generator gives [0,1],
    # and you trained that way, we keep it as is.
    # Note: We do NOT pass input_tensor.
    base_model = EfficientNetB3(
        include_top=False, 
        weights='imagenet', 
        pooling=None, 
        input_shape=input_shape
    )
    
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'EfficientNetB3'


def convnext_model(num_classes, input_shape):
    inputs = layers.Input(input_shape)

    base_model = ConvNeXtBase(
        include_top=False, 
        weights='imagenet', 
        pooling=None, 
        input_shape=input_shape
    )

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'ConvNeXt'


def swin_model(num_classes, input_shape):
    """
    Swin Transformer. 
    Note: GradCAM will likely skip this model as it produces 3D outputs, not 4D.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Swin implementation often requires pooling inside or specific structure
    base_model = SwinTransformerLarge224(
        include_top=False,  
        pooling='avg'       
    )
    
    x = base_model(inputs)
    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'Swin_KerasCV'


def get_model(generator, model_name='ResNet50'):
    assert model_name in ['ResNet50', 'EfficientNetB3', 'ConvNeXt', 'Swin_KerasCV']

    # Resetting seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    num_classes = generator.num_classes
    # Safety check for shape
    try:
        input_shape = generator[0][0][0].shape
    except:
        # Fallback if generator is empty or complex
        input_shape = (224, 224, 3)

    print(f"Building model: {model_name}")
    print(f"    Input Shape: {input_shape}")
    print(f"    Number of Classes: {num_classes}")

    if model_name == 'ResNet50':
        return resnet_model(num_classes, input_shape)
    elif 'EfficientNet' in model_name:
        return efficientnet_b3_model(num_classes, input_shape)
    elif model_name == 'ConvNeXt':
        return convnext_model(num_classes, input_shape)
    elif model_name == 'Swin_KerasCV':
        return swin_model(num_classes, input_shape)
    else:
        # Fallback
        return resnet_model(num_classes, input_shape)