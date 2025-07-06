import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (
    ResNet50V2,
    EfficientNetB7,
    ConvNeXtBase
)
from vit_keras import vit


def resnet_model(num_classes, input_shape):
    """
    Creates a model with a ResNet50V2 backbone.

    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input images (H, W, C).

    Returns:
        A tuple containing the Keras Model and the model's name.
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

    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input images (H, W, C).

    Returns:
        A tuple containing the Keras Model and the model's name.
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

    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input images (H, W, C).

    Returns:
        A tuple containing the Keras Model and the model's name.
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
    Creates a model with a Vision Transformer (ViT) backbone.
    NOTE: This model has a different architecture and input requirements.

    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input images (H, W, C).
                             ViT often requires specific input sizes, e.g., (224, 224, 3).

    Returns:
        A tuple containing the Keras Model and the model's name.
    """
    image_size = 224

    inputs = layers.Input(input_shape)
    x = layers.Resizing(image_size, image_size)(inputs)

    vit_base_model = vit.vit_b16(
        image_size=image_size,
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )

    x = vit_base_model(x)
    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, 'ViT'


def get_model(generator, model_name='ResNet50'):
    """
    Retrieves a specified model by name, configured for the given data generator.

    Args:
        generator: A Keras data generator from which num_classes and input_shape are inferred.
        model_name (str): The name of the model to retrieve.

    Returns:
        A tuple containing the compiled Keras Model and the model's name.
    """
    assert model_name in ['ResNet50', 'EfficientNetB7', 'ConvNeXt', 'ViT']

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
    elif model_name == 'ViT':
        return vit_model(num_classes, input_shape)
