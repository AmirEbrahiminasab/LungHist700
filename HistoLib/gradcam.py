from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib as mpl
from tensorflow.keras.models import Model
import cv2


# The following two functions were copied from: 
# https://keras.io/examples/vision/grad_cam/

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Create the gradient model
    # Note: in Keras 3, slicing models like this can be unstable with mixed input types
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Prepare the input strictly as a Tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # 3. Handle Input Naming
    # Keras 3 Functional API requires inputs to exactly match the dictionary keys 
    # of the internal graph if the model has named inputs.
    input_name = model.inputs[0].name
    
    # We construct a dictionary mapping the Input Layer Name -> Image Tensor
    input_data = {input_name: img_tensor}

    # 4. Compute Gradients
    with tf.GradientTape() as tape:
        # We explicitly watch the tensor, not the dictionary
        tape.watch(img_tensor)
        
        try:
            # Try passing the dictionary (standard Keras 3 way)
            last_conv_layer_output, preds = grad_model(input_data)
        except (KeyError, ValueError):
            # Fallback: Try passing the tensor directly 
            # (sometimes works if the dictionary mapping fails)
            last_conv_layer_output, preds = grad_model(img_tensor)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 5. Process Gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)/255
    return array
        
        
def merge_image_mask(im, mk, alpha=0.3, colormap='jet', mask_thresh=0.5):
    """
    Given an image and a mask of the same size, it blends the images together.
    """
    
    # Ensure the mask doesn't have values out of [0-1] range.
    mk = np.clip(mk, 0, 1)

    heatmap = np.uint8(255 * mk)
    jet = mpl.colormaps[colormap]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Blend the images together
    merge_image = jet_heatmap * alpha + im * (1-alpha)
    merge_image[mk<mask_thresh] = im[mk<mask_thresh]
    
    return merge_image
    
        
def generate_gradcam_samples(model, generator, N=8, 
                             mask_thresh=0.25, layer='conv5_block3_3_conv'):
    """
    Show some samples of the test generator. 
    """
    fig, axs = plt.subplots(3,N, figsize=(25,8))
    used = set()   
    for i in range(N):    
        
        # Ensure they are unique
        idx = np.random.randint(0, len(generator.images))
        while idx in used:
            idx = np.random.randint(0, len(generator.images))
        used.add(idx)
        
        x,y,z = generator[0][0][0].shape
        imsize = x,y
        
        im = get_img_array(generator.images[idx], imsize)
        
        # Original image
        axs[0,i].imshow(im[0]);
        axs[0,i].axis('off')
        axs[0,i].set_title(f"Real: {generator.images[idx].split('/')[2]} {generator.labels[idx]}", fontsize=18)
        
        heat = make_gradcam_heatmap(im, model, layer)
        pos_i, pos_j = imsize
        heat = cv2.resize(heat, dsize=(pos_j, pos_i))
        
        # Heatmap
        axs[1,i].imshow(heat)
        axs[1,i].axis('off')
        axs[1,i].set_title(f"Pred: {np.argmax(model.predict(im)[0])}", fontsize=18)
        
        # Merged
        axs[2,i].imshow(merge_image_mask(im[0], heat, mask_thresh=mask_thresh))
        axs[2,i].axis('off')


    plt.tight_layout()
    plt.show()