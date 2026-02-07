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

def clean_input_shape(input_shape):
    """
    Ensures input dimensions are multiples of 32 (required for Swin).
    """
    h, w, c = input_shape
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    if new_h != h or new_w != w:
        print(f"Warning: Input shape {input_shape} adjusted to {(new_h, new_w, c)} for Swin compatibility.")
    return (new_h, new_w, c)

def resnet_model(num_classes, input_shape):
    """
    ResNet50V2 backbone.
    """
    inputs = layers.Input(input_shape)
    # ResNetV2 expects [-1, 1].
    x = layers.Rescaling(scale=1./127.5, offset=-1)(inputs)

    base_model = ResNet50V2(
        include_top=False, 
        weights='imagenet', 
        pooling=None, 
        input_shape=input_shape
    )
    
    # We name the backbone layer so GradCAM can find it easily
    x = base_model(x)
    
    # x is now (Batch, H, W, 2048). GradCAM attaches here.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'ResNet50V2'

def efficientnet_b3_model(num_classes, input_shape):
    inputs = layers.Input(input_shape)
    
    # EfficientNetB3 in Keras Apps expects [0, 255] for default weights
    # But if we trained with [0, 1], we should stick to it or rescale if needed.
    # Assuming the standard Keras App usage which includes preprocessing:
    
    base_model = EfficientNetB3(
        include_top=False, 
        weights='imagenet', 
        pooling=None, 
        input_shape=input_shape
    )
    
    x = base_model(inputs)
    
    # x is (Batch, H, W, C). GradCAM attaches here.
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
    Swin Transformer with manual reshaping for GradCAM support.
    """
    # Adjust input shape for Swin (must be divisible by 32)
    input_shape = clean_input_shape(input_shape)
    inputs = layers.Input(shape=input_shape)
    
    # pooling=None ensures we get the sequence output: (Batch, N, C)
    base_model = SwinTransformerLarge224(
        include_top=False,  
        pooling=None       
    )
    
    x = base_model(inputs) 
    # Output is (Batch, NumPatches, Channels). e.g. (B, 49, 1536)
    
    # Reshape back to spatial map (B, H/32, W/32, C)
    # This allows GradCAM to see "where" the activation is.
    h_dim = input_shape[0] // 32
    w_dim = input_shape[1] // 32
    
    x = layers.Reshape((h_dim, w_dim, -1), name="swin_reshape")(x)
    
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, 'Swin_KerasCV'

def get_model(generator, model_name='ResNet50'):
    tf.random.set_seed(42)
    np.random.seed(42)

    num_classes = generator.num_classes
    try:
        input_shape = generator[0][0][0].shape
    except:
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
        return resnet_model(num_classes, input_shape)