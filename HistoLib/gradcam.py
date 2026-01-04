import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

# ==============================================================================
# Helper: Load and Preprocess Image
# ==============================================================================
def get_img_array(img_path, size):
    """
    Loads an image and converts it to a Numpy array of shape (1, H, W, C).
    """
    # Load image (returns PIL image)
    img = keras.utils.load_img(img_path, target_size=size)
    # Convert to array
    array = keras.utils.img_to_array(img)
    # Add batch dimension and normalize to [0,1]
    array = np.expand_dims(array, axis=0) / 255.0
    return array

# ==============================================================================
# Core: Make GradCAM Heatmap
# ==============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes the Grad-CAM heatmap for a given image and model.
    Compatible with both Keras 2 and Keras 3.
    """
    # 1. Create a model that maps the input image to the activations
    #    of the last conv layer as well as the output predictions.
    try:
        grad_model = Model(
            inputs=model.inputs, 
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        print(f"Error creating grad_model. Check if '{last_conv_layer_name}' exists.")
        raise e

    # 2. Prepare the input data
    # Keras 3 is strict about inputs. If the model has named inputs, we must use a dict.
    img_tensor = tf.convert_to_tensor(img_array)
    
    # Check if we need to wrap input in a dictionary (Keras 3 fix)
    if isinstance(model.input_names, list) and len(model.input_names) == 1:
        input_data = {model.input_names[0]: img_tensor}
    else:
        input_data = img_tensor

    # 3. Compute the gradient
    with tf.GradientTape() as tape:
        # We must watch the tensor explicitly
        tape.watch(img_tensor)
        
        try:
            # Pass dictionary or tensor based on logic above
            last_conv_layer_output, preds = grad_model(input_data)
        except (KeyError, ValueError):
            # Fallback: sometimes direct tensor passing works when dict fails
            last_conv_layer_output, preds = grad_model(img_tensor)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 4. Process gradients
    # Gradient of the output neuron with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of mean intensity of the gradient over specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Return as numpy array
    return heatmap.numpy()

# ==============================================================================
# Helper: Merge Heatmap with Image
# ==============================================================================
def merge_image_mask(im, mk, alpha=0.3, colormap='jet', mask_thresh=0.5):
    """
    Overlays the heatmap mask (mk) onto the original image (im).
    """
    # Ensure the mask is in [0-1] range
    mk = np.clip(mk, 0, 1)

    # Apply colormap
    heatmap = np.uint8(255 * mk)
    jet = mpl.colormaps[colormap]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Blend images
    # Note: 'im' should be shape (H, W, 3)
    merge_image = jet_heatmap * alpha + im * (1 - alpha)
    
    # Only apply mask where activation is high enough
    merge_image[mk < mask_thresh] = im[mk < mask_thresh]
    
    return merge_image

# ==============================================================================
# Main: Generate Samples
# ==============================================================================
def generate_gradcam_samples(model, generator, N=8, 
                             mask_thresh=0.25, layer='conv5_block3_3_conv'):
    """
    Displays N random samples from the generator with their Grad-CAM heatmaps.
    """
    fig, axs = plt.subplots(3, N, figsize=(25, 8))
    used = set()
    
    # --- CRITICAL FIX: Determine correct input size from the Model ---
    # The original code guessed size from the generator, which causes mismatches in Keras 3.
    try:
        # model.input_shape is typically (None, H, W, C)
        target_h, target_w = model.input_shape[1:3]
        # Handle cases where input might be dynamic (None)
        if target_h is None or target_w is None:
            target_h, target_w = 299, 299 # Default fallback
    except AttributeError:
        # Fallback if model doesn't have standard input_shape property
        target_h, target_w = 299, 299
        
    imsize = (target_h, target_w)
    # ------------------------------------------------------------------

    for i in range(N):    
        # Ensure we pick unique random images
        idx = np.random.randint(0, len(generator.images))
        while idx in used:
            idx = np.random.randint(0, len(generator.images))
        used.add(idx)
        
        # Load image with the CORRECT size for the model
        im = get_img_array(generator.images[idx], imsize)
        
        # 1. Plot Real Image
        axs[0, i].imshow(im[0])
        axs[0, i].axis('off')
        
        # Handle label extraction safely (depends on generator path structure)
        try:
            label_part = generator.images[idx].split('/')[2]
            true_label = generator.labels[idx]
            title_text = f"Real: {label_part} {true_label}"
        except IndexError:
            title_text = f"Real: {generator.labels[idx]}"
        axs[0, i].set_title(title_text, fontsize=12)
        
        # 2. Generate Heatmap
        # Note: We pass the tensor, the model, and the layer name
        heat = make_gradcam_heatmap(im, model, layer)
        
        # Resize heatmap to match image visualization size
        heat = cv2.resize(heat, dsize=(target_w, target_h))
        
        # Plot Heatmap
        axs[1, i].imshow(heat)
        axs[1, i].axis('off')
        
        # Get prediction
        pred_score = model.predict(im, verbose=0)[0]
        pred_class = np.argmax(pred_score)
        axs[1, i].set_title(f"Pred: {pred_class}", fontsize=12)
        
        # 3. Plot Merged
        merged = merge_image_mask(im[0], heat, mask_thresh=mask_thresh)
        axs[2, i].imshow(merged)
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()