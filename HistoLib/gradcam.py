import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import tensorflow as tf
from tensorflow import keras


# ==============================================================================
# Helper: Load and Preprocess Image
# ==============================================================================
def get_img_array(img_path, size, preprocess_fn=None):
    """
    Loads an image and converts it to a NumPy array of shape (1, H, W, C).
    If preprocess_fn is provided (e.g. tf.keras.applications.resnet.preprocess_input),
    it will be applied. Otherwise scales to [0,1].
    """
    img = keras.utils.load_img(img_path, target_size=size)
    arr = keras.utils.img_to_array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    if preprocess_fn is not None:
        arr = preprocess_fn(arr)
    else:
        arr = arr / 255.0
    return arr


# ==============================================================================
# Helpers: Find a connected last Conv2D layer (robust for nested backbones)
# ==============================================================================
def _iter_layers_recursive(model):
    for layer in model.layers:
        yield layer
        if isinstance(layer, keras.Model):
            yield from _iter_layers_recursive(layer)

def _get_single_input_size(model, fallback=(224, 224)):
    """
    Returns (H, W) for the model's first input if possible.
    """
    try:
        shape = model.inputs[0].shape  # (None, H, W, C)
        h, w = shape[1], shape[2]
        if h is None or w is None:
            return fallback
        return int(h), int(w)
    except Exception:
        return fallback

def find_last_connected_conv2d(model):
    """
    Finds the last layer in the TOP LEVEL model that outputs a 4D tensor (Batch, H, W, C).
    This works for ResNet/EfficientNet if pooling=None is used in the backbone.
    """
    # Iterate backwards through layers
    for layer in reversed(model.layers):
        # We look for a layer that has a 4D output shape: (Batch, Height, Width, Channels)
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name
        except AttributeError:
            # Some layers (like multiple outputs) might differ, skip them
            continue
            
    # If no 4D layer found (common in Transformers like Swin/ViT which are 3D)
    return None


# ==============================================================================
# Core: Make Grad-CAM Heatmap (Keras 3 safe)
# ==============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # 1. Auto-detect layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_connected_conv2d(model)
        
    if last_conv_layer_name is None:
        raise ValueError("Could not find a 4D spatial layer (Conv2D-like). "
                         "If this is a Transformer (Swin/ViT), GradCAM requires "
                         "reshaping patches which is not supported by this generic function.")

    target_layer = model.get_layer(last_conv_layer_name)

    # 2. Build Gradient Model
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.outputs[0]]
    )

    # 3. Record Gradients
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


# ==============================================================================
# Helper: Merge Heatmap with Image
# ==============================================================================
def merge_image_mask(im, mk, alpha=0.35, colormap="jet", mask_thresh=0.25):
    """
    Overlays heatmap (mk in [0,1]) on image (im float in [0,1]).
    """
    mk = np.clip(mk, 0.0, 1.0).astype(np.float32)
    im = np.clip(im, 0.0, 1.0).astype(np.float32)

    heat_u8 = np.uint8(255 * mk)
    cmap = mpl.colormaps[colormap]
    colors = cmap(np.arange(256))[:, :3]  # (256,3)
    jet_heatmap = colors[heat_u8]         # (H,W,3)

    merged = jet_heatmap * alpha + im * (1.0 - alpha)
    merged[mk < mask_thresh] = im[mk < mask_thresh]
    return np.clip(merged, 0.0, 1.0)


# ==============================================================================
# Main: Generate Samples
# ==============================================================================
def generate_gradcam_samples(
    model,
    generator,
    N=8,
    mask_thresh=0.25,
    layer=None,                 # pass a layer name OR None to auto-detect
    preprocess_fn=None,         # optional: e.g. tf.keras.applications.resnet.preprocess_input
    seed=None
):
    """
    Displays N random samples from a generator that has generator.images (paths)
    and optionally generator.labels.
    """
    if seed is not None:
        np.random.seed(seed)

    if not hasattr(generator, "images"):
        raise AttributeError("Expected generator to have `.images` (list of image paths).")
    
    try:
        if find_last_connected_conv2d(model) is None:
            print("\n[WARN] Skipping GradCAM: No spatial 4D feature map found (Model might be Swin/ViT).")
            return

        # Determine model input size
        target_h, target_w = _get_single_input_size(model, fallback=(224, 224))
        imsize = (target_h, target_w)

        fig, axs = plt.subplots(3, N, figsize=(3.2 * N, 9))
        if N == 1:
            # keep indexing consistent
            axs = np.array(axs).reshape(3, 1)

        used = set()
        max_len = len(generator.images)

        # Resolve labels if present
        has_labels = hasattr(generator, "labels")

        # If layer not provided, auto-detect once using model + dummy input
        if layer is None:
            layer = find_last_connected_conv2d(model)

        for i in range(N):
            idx = np.random.randint(0, max_len)
            while idx in used:
                idx = np.random.randint(0, max_len)
            used.add(idx)

            path = generator.images[idx]
            im = get_img_array(path, imsize, preprocess_fn=preprocess_fn)  # (1,H,W,C)

            # If preprocess_fn was used, im might not be in [0,1] for visualization
            # Create a display copy:
            if preprocess_fn is None:
                disp = im[0].copy()
            else:
                # best-effort display: min-max to [0,1]
                d = im[0].astype(np.float32)
                d = d - d.min()
                d = d / (d.max() + 1e-8)
                disp = d

            # Row 1: original
            axs[0, i].imshow(disp)
            axs[0, i].axis("off")
            title = "Real"
            if has_labels:
                title += f": {generator.labels[idx]}"
            axs[0, i].set_title(title, fontsize=11)

            # Heatmap
            heat, used_layer = make_gradcam_heatmap(im, model, last_conv_layer_name=layer)
            heat = cv2.resize(heat, (target_w, target_h)).astype(np.float32)

            # Row 2: heatmap + prediction
            axs[1, i].imshow(heat)
            axs[1, i].axis("off")
            pred = model.predict(im, verbose=0)
            pred_arr = np.array(pred).squeeze()
            if pred_arr.ndim == 0:
                pred_text = f"{float(pred_arr):.4f}"
            elif pred_arr.ndim == 1 and pred_arr.shape[0] == 1:
                pred_text = f"{float(pred_arr[0]):.4f}"
            elif pred_arr.ndim == 1:
                pred_text = f"class {int(np.argmax(pred_arr))}"
            else:
                pred_text = f"class {int(np.argmax(pred_arr, axis=-1)[0])}"
            axs[1, i].set_title(f"Pred: {pred_text}\nLayer: {used_layer}", fontsize=10)

            # Row 3: overlay (use disp which is [0,1])
            merged = merge_image_mask(disp, heat, mask_thresh=mask_thresh)
            axs[2, i].imshow(merged)
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n[Error] GradCAM failed for this model: {e}")


# ==============================================================================
# USAGE
# ==============================================================================
# 1) Auto-detect last conv layer (recommended):
# generate_gradcam_samples(model, val_generator, N=8)

# 2) Or explicitly pass a layer name if you know it:
# generate_gradcam_samples(model, val_generator, N=8, layer="conv5_block3_3_conv")

# 3) If you trained with a specific preprocess_input:
# from tensorflow.keras.applications.resnet import preprocess_input
# generate_gradcam_samples(model, val_generator, N=8, preprocess_fn=preprocess_input)
