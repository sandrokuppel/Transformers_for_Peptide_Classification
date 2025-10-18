import torch
from Transformers_for_peptide_Classification import (
    TPrep,
    CrossViT_Encoder,
    CrossAttention,
    rename_keys,
    remove_prefix,
)
from .Classification_model import MAE_MultiModal_WaveletsRaw_Classification

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
    "filename": "some_filename",
    "num_classes": 42,
    "num_epochs": 500,
    "steps_per_epoch": steps_per_epoch,
    "depth": 12,               # depth of the transformer encoder
    "cross_att_freq": 1,  # frequency of cross attention layers (eg. every layer = 1, every 2nd layer = 2 etc.)
    "learning_rate": 1e-4,
    "dimension_list": [256, 256],  # transformer dimension for each branch
    'feature_input_dim': 5,        # dimension of the feature vector input
    "heads_list": [8, 8],
    "hidden_dim_list": [256 * 4, 256 * 4],
    "input_dimension_list": [patch_size_raw, patch_size_image**2],
    "batch_size": batch_size,
    "patch_size_raw": patch_size_raw,
    "patch_size_image": patch_size_image,
    "seq_length_raw": seq_length_raw,
    "max_length_raw": max_length_raw,
    "seq_length_image": seq_length_image,
    "height": image_height,
    "width": image_width,
    "channels": 2,            # number of input channels (eg. 1 for single channel time series, 2 for wavelet + raw etc.), feature vector does not count into this
    "Pretrained": True,
    "Pretraining": False,
    "dropout": 0.2,
    "label_smoothing": 0.1,  # float between 0 and 1
    "alpha": 0.1,
    "weight_decay": 0.01,
}

hp_lr = {
    "pct_start": 0.02,  # percent of steps spend in warmup
    "div_factor": 20,  # initial learning rate is div_factor smaller than max_lr
    "final_div_factor": 15,  # final learning rate is final_div_factor times the initial learning rate
}

### load pretrained weights ###
checkpoint = torch.load(".../checkpoints/epoch=361-step=59368.ckpt")

model_state_dict = checkpoint["state_dict"]
# get indicidual weights for the classes
encoder_state_dict = {k: v for k, v in model_state_dict.items() if "encoder" in k}
PrepPatches_raw_state_dict = {
    k: v for k, v in model_state_dict.items() if "PrepPatches_raw" in k
}
PrepPatches_image_state_dict = {
    k: v for k, v in model_state_dict.items() if "PrepPatches_image" in k
}
# create classes
encoder = CrossViT_Encoder(hp)
PrepPatches_raw = TPrep(
    input_dimension=hp["input_dimension_list"][0],
    dimension=hp["dimension_list"][0],
    sequence_length=hp["seq_length_raw"],
    Pretrained=hp["Pretrained"],
)
PrepPatches_image = TPrep(
    input_dimension=hp["input_dimension_list"][1],
    dimension=hp["dimension_list"][1],
    sequence_length=hp["seq_length_image"],
    Pretrained=hp["Pretrained"],
)
# rename keys
encoder_state_dict = rename_keys(encoder_state_dict, remove_prefix, "encoder.")
PrepPatches_raw_state_dict = rename_keys(
    PrepPatches_raw_state_dict, remove_prefix, "PrepPatches_raw."
)
PrepPatches_image_state_dict = rename_keys(
    PrepPatches_image_state_dict, remove_prefix, "PrepPatches_image."
)

# load weights
encoder.load_state_dict(encoder_state_dict)
PrepPatches_raw.load_state_dict(PrepPatches_raw_state_dict)
PrepPatches_image.load_state_dict(PrepPatches_image_state_dict)


cross_att_raw_state_dict = {
    k: v for k, v in model_state_dict.items() if "cross_att_raw" in k
}
cross_att_raw_state_dict = rename_keys(
    cross_att_raw_state_dict, remove_prefix, "cross_att_raw."
)
cross_att_wavelet_state_dict = {
    k: v for k, v in model_state_dict.items() if "cross_att_wavelet" in k
}
cross_att_wavelet_state_dict = rename_keys(
    cross_att_wavelet_state_dict, remove_prefix, "cross_att_wavelet."
)
fc1_state_dict = {k: v for k, v in model_state_dict.items() if "fc1" in k}
fc1_state_dict = rename_keys(fc1_state_dict, remove_prefix, "fc1.")
fc2_state_dict = {k: v for k, v in model_state_dict.items() if "fc2" in k}
fc2_state_dict = rename_keys(fc2_state_dict, remove_prefix, "fc2.")

cross_att_raw = CrossAttention(
    256, 256, hp["heads_list"][0], hp["dropout"], add_in_both_latent_space=False
)
cross_att_img = CrossAttention(
    256, 256, hp["heads_list"][1], hp["dropout"], add_in_both_latent_space=False
)
fc1 = torch.nn.Linear(5, 128)
fc2 = torch.nn.Linear(128, 256)
fc3 = torch.nn.Linear(256, hp["num_classes"])
cross_att_raw.load_state_dict(cross_att_raw_state_dict)
cross_att_img.load_state_dict(cross_att_wavelet_state_dict)
fc1.load_state_dict(fc1_state_dict)
fc2.load_state_dict(fc2_state_dict)

# Initialize model
model = MAE_MultiModal_WaveletsRaw_Classification(
    hp,
    hp_lr,
    model_name="Example_Classification_Model",
    encoder=encoder,
    PrepPatches_raw=PrepPatches_raw,
    PrepPatches_image=PrepPatches_image,
    fc1=fc1,
    fc2=fc2,
    cross_att_raw=cross_att_raw,
    cross_att_img=cross_att_img,
).to(device)


'''
- Write your own dataloader specific to your dataset 
    Use torch.utils.data.StackDataset() to stack your datasets for each modality
    train_ds_rwc = torch.utils.data.StackDataset(train_ds_time, train_ds_image, train_feature)
    Expected returns of the torch Dataset:
        - Time series: time_series_array, label
        - Image: image_array, label
            -> this hast to be flat patches of the image (eg. for 224x224 image with patch size 14 -> 256 patches of size 14x14 flattened to size 196)
            -> function in helper_functions.py: img_to_patches(prepare_picture_no_batch(x, patch_size))
        - Feature vector: feature_vector_array, label
- Adjust hyperparameters, adjust dimension of feed forward to your feature vector size
- Train as usual using pytorch lightning trainer
- use logger = TensorBoardLogger(path, name=model_name) to log training
'''