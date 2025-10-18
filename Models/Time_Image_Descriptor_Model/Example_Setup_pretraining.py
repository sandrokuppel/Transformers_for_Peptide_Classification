import torch
from .Pretraining_Model import MAE_MultiModal_WaveletsRaw_Pretraining

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
"""
Example configuration for input of:
"""
# time series:
max_length_raw = 928            # max length of your time series input (use zero padding to this length)
patch_size_raw = 4               # patch size for time series
seq_length_raw = max_length_raw // patch_size_raw
# image:
image_height = 224
image_width = 224
image_size = image_height * image_width        # image size (has to be image with one channel)
patch_size_image = 14         # patch size for image
seq_length_image = (image_size // patch_size_image**2)
# learning rate schedule
samples_in_dataset = 300000 # number of samples in your dataset
batch_size = 800    
steps_per_epoch = int(len(samples_in_dataset) / int(float(batch_size)))

hp = {
    "filename": 'some_filename',
    "num_epochs": 500,
    "steps_per_epoch": steps_per_epoch,
    'feature_input_dim': 5, 
    "depth": 12,
    'depth_decoder': 4,
    'cross_att_freq': 1,  # frequency of cross attention layers
    "learning_rate": 1e-4,
    'dimension_list': [256, 256],
    'heads_list': [8, 8],
    'hidden_dim_list': [256 * 4, 256 * 4],
    'input_dimension_list': [patch_size_raw, patch_size_image**2],
    "batch_size": 400,
    "patch_size_raw": patch_size_raw,
    "patch_size_image": patch_size_image,
    "num_classes": 42,  # numbers from 0 to 9
    "seq_length_raw": seq_length_raw, 
    'max_length_raw': max_length_raw,
    "seq_length_image": seq_length_image,
    'height': image_height,
    'width': image_width,
    "channels": 2,
    "Pretrained": False,
    "dropout": 0.1,
    'alpha': 0.1,
    "weight_decay": 0,
    "use_catch25_only_rel": True
}

hp_lr= {
    'pct_start': 0.02,   # percent of steps spend in warmup 
    'div_factor': 20,   # initial learning rate is div_factor smaller than max_lr
    'final_div_factor': 15,  # final learning rate is final_div_factor times the initial learning rate
}

hp_mask = {
    "mask_ratio_image": 0.7,     # ratio of masked tokens for image input
    "mask_ratio_raw": 0.5       # ratio of masked tokens for time series input
}

# Initialize model
model = MAE_MultiModal_WaveletsRaw_Pretraining(hp, hp_lr, hp_mask).to(device)

'''
- Write your own dataloader specific to your dataset 
    Use torch.utils.data.StackDataset() to stack your datasets for each modality
    train_ds_rwc = torch.utils.data.StackDataset(train_ds_time, train_ds_image, train_feature)
    Expected returns of the torch Dataset:
        - Time series: time_series_array, label, l      # l is the length of the time series before padding 
        - Image: image_array, label
            -> this hast to be flat patches of the image (eg. for 224x224 image with patch size 14 -> 256 patches of size 14x14 flattened to size 196)
            -> function in helper_functions.py: img_to_patches(prepare_picture_no_batch(x, patch_size))
        - Feature vector: feature_vector_array, label
- Adjust hyperparameters, adjust dimension of feed forward to your feature vector size
- Train as usual using pytorch lightning trainer
- use logger = TensorBoardLogger(path, name=model_name) to log training
'''