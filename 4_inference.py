import torch
import os
import pandas as pd
import segmentation_models_pytorch as smp
from utils import BioMasstersDatasetS2S1, SentinelModel, inference_agb_2m
import rasterio as rio
import warnings
import json
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

root_dir = os.getcwd() # Change to the folder where you stored preprocessed training data

S1_CHANNELS = {'2S': 8, '2SI': 12, '3S': 12, '4S': 16, '4SI': 24, '6S': 24}
S2_CHANNELS = {'2S': 20, '2SI': 38, '3S': 30, '4S': 40, '4SI': 48, '6S': 60}


df = pd.read_csv(os.path.join(f'./data/train_val_split_96_0.csv'), dtype={"id": str})
X_train, X_val, X_test = (df["id"].loc[df["dataset"] == 0].values,
                          df["id"].loc[df["dataset"] == 1].values,
                          df["id"].loc[df["dataset"] == 2].values)
print(df["dataset"].value_counts())
print("Total Images: ", len(df))



f = open('./data/mean.json')
mean = json.load(f)
f = open('./data/std.json')
std = json.load(f)
f = open('./data/mean_agb.json')
mean_agb = json.load(f)
f = open('./data/std_agb.json')
std_agb = json.load(f)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

def model_inference(suffix, encoder_name, decoder_attention_type, 
                    checkpoint_path_1, checkpoint_path_2, output_val_dir, output_test_dir):

    model = smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=S1_CHANNELS[suffix]+S2_CHANNELS[suffix],
                             decoder_attention_type=decoder_attention_type, classes=1, activation=None)

    s2s1_model_1 = SentinelModel.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path_1, 
                                                      mean_agb=mean_agb, std_agb=std_agb)
    s2s1_model_2 = SentinelModel.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path_2, 
                                                      mean_agb=mean_agb, std_agb=std_agb)


    # Update the batch size here for inference datasets
    val_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S", 
                                     s1_path=f"{root_dir}/train_features_s1_6S", 
                                     agb_path=f"{root_dir}/train_agbm", X=X_val, mean=mean['6S'], std=std['6S'], 
                                     mean_agb=mean_agb, std_agb=std_agb, transform=None)

    # Here, you'd usually create a DataLoader to pass the dataset to the model in batches
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)
    
    # Inference step using the created DataLoader
    inference_agb_2m(s2s1_model_1, s2s1_model_2, val_loader, val_loader, device, mean_agb=mean_agb, std_agb=std_agb, 
                     clamp_threshold=None, preds_agbm_dir=output_val_dir, save_ground_truth=False)

    test_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/test_features_s2_6S", 
                                      s1_path=f"{root_dir}/test_features_s1_6S",
                                      agb_path=None, X=X_test, mean=mean['6S'], std=std['6S'], 
                                      mean_agb=mean_agb, std_agb=std_agb, transform=None)

    # Similarly for the test set, we create a DataLoader with the batch size of 16
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    
    # Inference step for the test set
    inference_agb_2m(s2s1_model_1, s2s1_model_2, test_loader, test_loader, device, mean_agb=mean_agb, std_agb=std_agb, 
                     clamp_threshold=None, preds_agbm_dir=output_test_dir, save_ground_truth=False)

model_inference('6S', "mit_b5", None, 
                os.path.join(root_dir, 'models/se_resnext50_32x4d_6S_None/dbdy005j/checkpoints/loss=0.07437055557966232.ckpt'),
                os.path.join(root_dir, 'models/se_resnext50_32x4d_6S_None/dbdy005j/checkpoints/loss=0.07442700117826462.ckpt'), 
                f"{root_dir}/dd_inf_models/val_dbdy005j_2m",
                f"{root_dir}/dd_inf_models/test_dbdy005j_2m")


