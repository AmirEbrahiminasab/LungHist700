import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import tensorflow as tf
from tensorflow import keras

def get_img_array(img_path, size, preprocess_fn=None):
    # Load image
    img = keras.utils.load_img(img_path, target_size=size)
    arr = keras.utils.img_to_array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    
    # Normalize to [0, 1]. Models with Rescaling layers will handle [-1, 1] internally.
    if preprocess_fn is not None:
        arr = preprocess_fn(arr)
    else:
        arr = arr / 255.0
    return arr

def find_target_layer(model):
    """
    Finds the best layer for GradCAM.
    Priorities:
    1. Last Conv2D layer.
    2. Layers named 'resnet', 'efficientnet' (Nested Backbones).
    3. Layers named 'swin_reshape' (Custom Swin fix).
    """
    # 1. Check for custom Swin reshape layer first
    for layer in reversed(model.layers):
        if 'swin_reshape' in layer.name:
            return layer.name

    # 2. Check for nested backbones (common in Transfer Learning)
    # We return the backbone layer itself; GradCAM will use its output.
    for layer in reversed(model.layers):
        # Identify backbones by typical naming conventions
        if any(x in layer.name.lower() for x in ['resnet', 'efficientnet', 'convnext', 'densenet', 'vgg']):
            # Ensure it actually has outputs
            try:
                if isinstance(layer.output, list): continue
                return layer.name
            except: continue

    # 3. Fallback: Search for the last layer with 4D output (Batch, H, W, C)
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name
        except AttributeError:
            continue
            
    raise ValueError("Could not find a suitable 4D target layer for GradCAM.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_target_layer(model)
        print(f"Auto-selected layer for GradCAM: {last_conv_layer_name}")

    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
        )
    except ValueError as e:
        print(f"Error building GradCAM model: {e}")
        return np.zeros(img_array.shape[1:3]), last_conv_layer_name

    with tf.GradientTape() as tape:
        # Cast input to match model expectation (usually float32)
        inputs = tf.cast(img_array, tf.float32)
        
        # Watch the input explicitly (sometimes needed for nested models)
        tape.watch(inputs)
        
        # Forward pass
        # training=False is crucial to disable Dropout/BatchNorm training behavior
        conv_out, preds = grad_model(inputs, training=False)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_out)

    # Vector of weights: mean intensity of the gradient per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    conv_out = conv_out[0]
    
    # Element-wise multiplication and summation
    heatmap = tf.matmul(conv_out, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy(), last_conv_layer_name

def merge_image_mask(im, mk, alpha=0.4, colormap="jet"):
    # Resizing heat map to match image dimensions
    mk = cv2.resize(mk, (im.shape[1], im.shape[0]))
    
    mk = np.clip(mk, 0.0, 1.0)
    im = np.clip(im, 0.0, 1.0)
    
    heat_u8 = np.uint8(255 * mk)
    cmap = mpl.colormaps[colormap]
    
    # Get RGB values from colormap
    colors = cmap(np.arange(256))[:, :3]
    jet_heatmap = colors[heat_u8]
    
    merged = jet_heatmap * alpha + im * (1.0 - alpha)
    return np.clip(merged, 0.0, 1.0)

def generate_gradcam_samples(model, generator, N=6, layer=None, seed=None):
    if seed: np.random.seed(seed)
    
    # Detect Input Size from model input or generator
    try:
        input_shape = model.inputs[0].shape[1:3] # (H, W)
        if input_shape[0] is None: raise ValueError
    except:
        # Fallback to checking the generator image size
        img_ex = generator[0][0][0]
        input_shape = (img_ex.shape[0], img_ex.shape[1])

    target_size = input_shape
    print(f"GradCAM running with Target Size: {target_size}")

    fig, axs = plt.subplots(2, N, figsize=(3 * N, 6))
    
    # Handle single column case
    if N == 1: axs = axs[:, np.newaxis]

    indices = np.random.choice(len(generator.images), N, replace=False)

    for i, idx in enumerate(indices):
        path = generator.images[idx]
        
        # Load and Preprocess
        img_array = get_img_array(path, target_size)
        
        # Original Image for display
        disp_img = img_array[0].copy()

        try:
            # Generate Heatmap
            heatmap, used_layer = make_gradcam_heatmap(img_array, model, last_conv_layer_name=layer)
            
            # Overlay
            merged = merge_image_mask(disp_img, heatmap)
            
            # Prediction
            preds = model.predict(img_array, verbose=0)
            pred_lbl = np.argmax(preds[0])
            real_lbl = np.argmax(generator.labels[idx])
            
            axs[0, i].imshow(disp_img)
            axs[0, i].set_title(f"Real: {real_lbl}\nPred: {pred_lbl}")
            axs[0, i].axis("off")
            
            axs[1, i].imshow(merged)
            axs[1, i].set_title(f"Layer:\n{used_layer[-15:]}", fontsize=8)
            axs[1, i].axis("off")
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            axs[0, i].text(0.5, 0.5, "Error", ha='center')
            axs[1, i].text(0.5, 0.5, str(e)[:20], ha='center')

    plt.tight_layout()
    plt.show()