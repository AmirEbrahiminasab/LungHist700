# %%
import numpy as np
import os
import shutil
import tensorflow as tf
import gc

from HistoLib import generator
from HistoLib import utils
from HistoLib import models
from HistoLib import traintest
from HistoLib import gradcam

# %%
np.random.seed(42)
tf.random.set_seed(42)

# %%
utils.dataset_description()

# %% [markdown]
# ## Get images

# %%
resolution = '20x'      # One of ['20x', '40x']

train_generator, val_generator, test_generator, class_names = generator.get_patient_generators(resolution, batch_size=8, 
                                                                                               debug=True,                 # Shows the number of images being used.
                                                                                               reproducible=True           # Use the splits from the original paper.
                                                                                              )

# %%
train_generator.show_generator()

# %%
val_generator.show_generator()

# %% [markdown]
# ## Hyperparameter tuning

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
BATCH_SIZE_CANDIDATES = [8, 16] 

LR_CANDIDATES = [5e-5, 1e-5, 3e-5] 

GLOBAL_BEST_LOSS = float('inf')
GLOBAL_BEST_CFG = {}
RESULTS_LOG = []

def hyper_tuning(TARGET_MODEL_NAME):
    print(f"\n--- Hyper-Tuning for {TARGET_MODEL_NAME} ---")
    for batch_size in BATCH_SIZE_CANDIDATES:
        print(f"\n--- Setting up Generators for Batch Size {batch_size} ---")
        
        train_gen, val_gen, test_gen, classes = generator.get_patient_generators(
            resolution, batch_size=batch_size, debug=False, reproducible=True
        )
        
        class_weights = utils.compute_weights(train_gen)

        for lr in LR_CANDIDATES:
            print(f"\n[TUNING] Model: {TARGET_MODEL_NAME} | Batch: {batch_size} | LR: {lr}")
            
            model, model_name = models.get_model(train_gen, TARGET_MODEL_NAME)
            
            print("  > Stage 1: Warm-up (Head only)")
            model = traintest.make_backbone_trainable(model, trainable=False)
            
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
            
            model.fit(train_gen, 
                    validation_data=val_gen, 
                    epochs=8, 
                    class_weight=class_weights,
                    verbose=0) 
            
            print(f"  > Stage 2: Fine-Tuning with LR {lr}")
            model = traintest.make_backbone_trainable(model, trainable=True)
            
            model = traintest.compile_model(model, num_classes=len(classes), init_lr=lr)
            
            run_name = f"{TARGET_MODEL_NAME}_BS{batch_size}_LR{lr}"
            current_log_dir = traintest.get_logdir(TARGET_MODEL_NAME, base_log=f"/content/drive/MyDrive/tuning/{run_name}")
            
            history = traintest.train_model(
                model, 
                train_gen, 
                val_gen, 
                class_weights, 
                current_log_dir, 
                num_epochs=50,  
                patience=10,    
                patience_lr=5
            )
            
            best_run_val_loss = np.min(history.history['val_loss'])
            best_run_epoch = np.argmin(history.history['val_loss']) + 1
            
            print(f"  > Result: Val Loss {best_run_val_loss:.5f} at Epoch {best_run_epoch}")
            
            RESULTS_LOG.append({
                'batch_size': batch_size,
                'lr': lr,
                'val_loss': best_run_val_loss,
                'log_dir': current_log_dir
            })

            if best_run_val_loss < GLOBAL_BEST_LOSS:
                print(f"  *** NEW BEST MODEL FOUND! (Previous: {GLOBAL_BEST_LOSS:.5f}) ***")
                GLOBAL_BEST_LOSS = best_run_val_loss
                GLOBAL_BEST_CFG = {'batch_size': batch_size, 'lr': lr, 'path': current_log_dir}
                
            tf.keras.backend.clear_session()
            del model

    print("\n\n=== TUNING COMPLETE ===")
    print(f"Best Configuration: Batch {GLOBAL_BEST_CFG.get('batch_size')} | LR {GLOBAL_BEST_CFG.get('lr')}")
    print(f"Best Val Loss: {GLOBAL_BEST_LOSS}")

# %% [markdown]
# ### ResNet50

# %%
hyper_tuning('ResNet50')

# %% [markdown]
# ### EffetiveNetB3

# %%
hyper_tuning('EfficientNetB3')

# %% [markdown]
# ### Swin

# %%
hyper_tuning('Swin_KerasCV')

# %% [markdown]
# ## Train Model

# %%
class_weights = utils.compute_weights(train_generator)

# %%
def train_pipeline(lr, TARGET_MODEL_NAME):
    tf.keras.backend.clear_session()
    gc.collect()
    
    model, model_name = models.get_model(train_generator, TARGET_MODEL_NAME)
            
    print("  > Stage 1: Warm-up (Head only)")
    model = traintest.make_backbone_trainable(model, trainable=False)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    
    model.fit(train_generator, 
                    validation_data=val_generator, 
                    epochs=8, 
                    class_weight=class_weights,
                    verbose=0) 
    
    print(f"  > Stage 2: Fine-Tuning with LR {lr}")
    model = traintest.make_backbone_trainable(model, trainable=True)
    
    model = traintest.compile_model(model, num_classes=len(class_weights), init_lr=lr)

    return model, model_name

# %% [markdown]
# ### ResNet50

# %%
model, model_name = train_pipeline(lr=1e-5, TARGET_MODEL_NAME="ResNet50")

# %%
log_dir = traintest.get_logdir(model_name, base_log="/content/drive/MyDrive/logs/resnet50")

# %%
history = traintest.train_model(model, train_generator, val_generator, class_weights, log_dir)

# %% [markdown]
# ### EfficientNetB7

# %%
model, model_name = train_pipeline(lr=1e-5, TARGET_MODEL_NAME="EfficientNetB7")

# %%
log_dir = traintest.get_logdir(model_name, base_log="/content/drive/MyDrive/logs/EfficientNetB7")

# %%
history = traintest.train_model(model, train_generator, val_generator, class_weights, log_dir)

# %% [markdown]
# ### Swin Transformer

# %%
model, model_name = train_pipeline(lr=3e-5, TARGET_MODEL_NAME="Swin_KerasCV")

# %%
log_dir = traintest.get_logdir(model_name, base_log="/content/drive/MyDrive/logs/Swin")

# %%
history = traintest.train_model(model, train_generator, val_generator, class_weights, log_dir)

# %% [markdown]
# ## Evaluate using test data

# %%
test_generator.show_generator()

# %%
traintest.metrics_and_test(history, model, test_generator, class_names)

# %%
gradcam.generate_gradcam_samples(model, val_generator)


