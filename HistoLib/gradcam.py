import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import tensorflow as tf
from tensorflow import keras

def get_img_array(img_path, size, preprocess_fn=None):
    img = keras.utils.load_img(img_path, target_size=size)
    arr = keras.utils.img_to_array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    if preprocess_fn is not None:
        arr = preprocess_fn(arr)
    else:
        arr = arr / 255.0
    return arr

def _get_single_input_size(model, fallback=(224, 224)):
    try:
        shape = model.inputs[0].shape
        h, w = shape[1], shape[2]
        if h is None or w is None: return fallback
        return int(h), int(w)
    except Exception:
        return fallback

def find_last_connected_conv2d(model):
    """
    Robustly finds the last layer that outputs a 4D feature map (Batch, H, W, C).
    Does NOT recurse into nested models to avoid graph disconnection errors.
    """
    # Iterate backwards through TOP-LEVEL layers only
    for layer in reversed(model.layers):
        try:
            # We look for a 4D tensor output: (Batch, Height, Width, Channels)
            if isinstance(layer.output, tf.Tensor) or isinstance(layer.output, keras.KerasTensor):
                shape = layer.output.shape
                if len(shape) == 4:
                    return layer.name
        except Exception:
            continue
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_connected_conv2d(model)
        
    if last_conv_layer_name is None:
        # Stop silently here; handled in generate loop
        raise ValueError("No 4D spatial layer found.")

    target_layer = model.get_layer(last_conv_layer_name)

    # Build gradient model
    # Note: We use model.inputs and target_layer.output. 
    # Because we only search top-level layers, this graph is always valid.
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        preds = tf.convert_to_tensor(preds)
        
        if len(preds.shape) == 1:
            class_channel = preds
        elif preds.shape[-1] == 1:
            class_channel = preds[:, 0]
        else:
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom

    return heatmap.numpy(), last_conv_layer_name

def merge_image_mask(im, mk, alpha=0.35, colormap="jet", mask_thresh=0.25):
    mk = np.clip(mk, 0.0, 1.0).astype(np.float32)
    im = np.clip(im, 0.0, 1.0).astype(np.float32)
    heat_u8 = np.uint8(255 * mk)
    cmap = mpl.colormaps[colormap]
    colors = cmap(np.arange(256))[:, :3]
    jet_heatmap = colors[heat_u8]
    merged = jet_heatmap * alpha + im * (1.0 - alpha)
    merged[mk < mask_thresh] = im[mk < mask_thresh]
    return np.clip(merged, 0.0, 1.0)

def generate_gradcam_samples(model, generator, N=8, mask_thresh=0.25, layer=None, preprocess_fn=None, seed=None):
    if seed is not None: np.random.seed(seed)
    
    # 1. Check if model is supported (Swin/ViT often fail here)
    if layer is None:
        layer = find_last_connected_conv2d(model)
        
    if layer is None:
        print(f"\n[{'!'*20}]\nSKIPPING GradCAM: No 4D feature map found.\n(This is expected for Swin/ViT or if 'pooling' was not None).\n[{'!'*20}]\n")
        return

    print(f"Generating GradCAM using layer: {layer}")

    target_h, target_w = _get_single_input_size(model)
    imsize = (target_h, target_w)

    fig, axs = plt.subplots(3, N, figsize=(3.2 * N, 9))
    if N == 1: axs = np.array(axs).reshape(3, 1)

    used = set()
    max_len = len(generator.images)
    has_labels = hasattr(generator, "labels")

    for i in range(N):
        idx = np.random.randint(0, max_len)
        while idx in used: idx = np.random.randint(0, max_len)
        used.add(idx)

        path = generator.images[idx]
        im = get_img_array(path, imsize, preprocess_fn=preprocess_fn)

        if preprocess_fn is None:
            disp = im[0].copy()
        else:
            d = im[0].astype(np.float32)
            d = d - d.min()
            d = d / (d.max() + 1e-8)
            disp = d

        # Plot Original
        axs[0, i].imshow(disp)
        axs[0, i].axis("off")
        title = "Real"
        if has_labels: title += f": {np.argmax(generator.labels[idx])}" # simplified label
        axs[0, i].set_title(title, fontsize=11)

        # Plot Heatmap
        try:
            heat, used_layer = make_gradcam_heatmap(im, model, last_conv_layer_name=layer)
            heat = cv2.resize(heat, (target_w, target_h)).astype(np.float32)
            
            axs[1, i].imshow(heat)
            axs[1, i].axis("off")
            
            # Prediction
            pred = model.predict(im, verbose=0)
            pred_idx = np.argmax(pred[0])
            axs[1, i].set_title(f"Pred: {pred_idx}\n{used_layer}", fontsize=8)

            # Plot Overlay
            merged = merge_image_mask(disp, heat, mask_thresh=mask_thresh)
            axs[2, i].imshow(merged)
            axs[2, i].axis("off")
            
        except Exception as e:
            print(f"GradCAM error on img {i}: {e}")
            axs[1, i].text(0.5, 0.5, "Error", ha='center')

    plt.tight_layout()
    plt.show()