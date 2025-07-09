import pandas as pd

# Constants for all scripts

# Constants for Data
DATASET_BOUNDS = {
        "train" : (pd.to_datetime("2010-07-01"), pd.to_datetime("2011-06-30")),
        "validation" : (pd.to_datetime("2011-07-01"), pd.to_datetime("2012-06-30")),
        "test" : (pd.to_datetime("2012-07-01"), pd.to_datetime("2013-06-30"))
    }
#
# REMARK DIFFERENT THAN DFR BECAUSE OF THE NEW DATALOADER. They produce exaclty the same Data as DFR. 
# Training: First Sample 2010-07-08 12:00:00 Last 2011-06-27 12:00:00
#


# Constants for Surrogate Models
OP_EST_NAMES = [
"prosumption_1_0/surrogate_models/model_2024_07_29_17_10_12.pt",
"prosumption_1_0/surrogate_models/model_2024_07_29_17_15_55.pt",
"prosumption_1_0/surrogate_models/model_2024_07_29_17_46_41.pt",
"prosumption_1_0/surrogate_models/model_2024_07_29_17_51_25.pt",
"prosumption_1_0/surrogate_models/model_2024_07_29_18_49_32.pt",
]

# Constants for Moirai

SIZE = "base"  # model size: choose from {'small', 'base', 'large'}
PDT = 42  # prediction length: any positive integer
CTX = 168  # context length: any positive integer
PATCH_SIZE = 32
FEAT_DYNAMIC_REAL = 0

DATASET_CONFIG = {
    "pdt": PDT,
    "ctx": CTX,
    "patch_size": PATCH_SIZE,
    "feat_dynamic_real": FEAT_DYNAMIC_REAL
}

MORAI_CONFIG = {
    "size": SIZE,
    "pdt": PDT,
    "ctx": CTX,
    "patch_size": PATCH_SIZE,
    "feat_dynamic_real": FEAT_DYNAMIC_REAL
}

# Constants for LoRA

LORA_CONFIG = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["v_proj", "q_proj", "k_proj", "out_proj"]}

# Constants for DoRA

DORA_CONFIG = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["v_proj", "q_proj", "k_proj", "out_proj"], "use_dora":True}



# Constants for Training
# The training buildings low and high are the buildings which are used for global training.
TRAINING_CONFIG = {
    "dataset_bounds": DATASET_BOUNDS,
    "peft_epochs": 1,
    "optimizer": "AdamW",
    "ctx": CTX,
    "pdt": PDT,
    "patch_size": PATCH_SIZE,
    "batch_size": 32,
    "op_est_names": OP_EST_NAMES,
    "training_building_low": 1,
    "training_building_high": 51,
}

