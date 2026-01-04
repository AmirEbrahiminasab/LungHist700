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

def find_last_connected_conv2d(model, input_size=None):
    """
    Returns the name of the last Conv2D layer that is actually connected
    to model.inputs (Keras 3 can otherwise throw KeyError / graph-disconnected).
    """
    if input_size is None:
        input_size = _get_single_input_size(model)

    # Collect Conv2D candidates from the whole (possibly nested) model
    conv_candidates = [l for l in _iter_layers_recursive(model) if isinstance(l, keras.layers.Conv2D)]
    if not conv_candidates:
        raise ValueError("No Conv2D layers found in the model (Grad-CAM Conv2D version won't work).")

    # Test connectivity from the end backwards
    dummy = tf.zeros((1, input_size[0], input_size[1], 3), dtype=tf.float32)
    last_ok = None
    last_err = None

    for layer in reversed(conv_candidates):
        try:
            test_model = keras.Model(inputs=model.inputs, outputs=layer.output)
            _ = test_model(dummy, training=False)
            last_ok = layer.name
            break
        except Exception as e:
            last_err = e
            continue

    if last_ok is None:
        raise RuntimeError(f"Found Conv2D layers, but none were connected. Last error was: {last_err}")
    return last_ok


# ==============================================================================
# Core: Make Grad-CAM Heatmap (Keras 3 safe)
# ==============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given (1,H,W,C) image array and model.
    Works with Keras 3. Avoids dict-input guessing and avoids model.output pitfalls.
    """
    # Ensure tensor float32
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Auto-pick a safe last conv layer if not provided
    if last_conv_layer_name is None:
        h, w = img_array.shape[1], img_array.shape[2]
        last_conv_layer_name = find_last_connected_conv2d(model, input_size=(h, w))

    # Get the target conv layer object (recursive search by name)
    target_layer = None
    for layer in _iter_layers_recursive(model):
        if layer.name == last_conv_layer_name:
            target_layer = layer
            break
    if target_layer is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found (even in nested models).")

    # Build gradient model (use model.outputs[0] to be safe in multi-output setups)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)

        # Handle binary / single-unit outputs
        # preds shape can be (1,), (1,1), or (1,num_classes)
        preds = tf.convert_to_tensor(preds)
        if len(preds.shape) == 1:
            # (1,) -> treat as single class logit/prob
            class_channel = preds
        elif preds.shape[-1] == 1:
            class_channel = preds[:, 0]
        else:
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    # Gradient of selected class w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_out)
    if grads is None:
        raise RuntimeError(
            "Gradient is None. This can happen if the chosen layer isn't in the forward path "
            "or if the model output isn't differentiable from that layer."
        )

    # Global-average-pool the gradients over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps
    conv_out = conv_out[0]  # (Hc, Wc, channels)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    # Normalize to [0,1]
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
        layer = find_last_connected_conv2d(model, input_size=imsize)

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
