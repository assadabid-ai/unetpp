import os
import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import rasterio as rio
from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import warnings
from torchsummary import summary
from torch.utils.data import DataLoader
import json
# import wandb
from utils import calcuate_mean_std, stratify_data, freeze_encoder, BioMasstersDatasetS2S1, SentinelModel

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
torch.set_printoptions(sci_mode=False)

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

def train_finetuned_model(checkpoint_path, suffix, encoder_name, decoder_attention_type):
    # wandb.finish()

    train_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                       s1_path=f"{root_dir}/train_features_s1_6S",
                                       agb_path=f"{root_dir}/train_agbm", X=X_train, mean=mean['6S'], std=std['6S'], 
                                       mean_agb=mean_agb, std_agb=std_agb, transform=None)

    val_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                     s1_path=f"{root_dir}/train_features_s1_6S",
                                     agb_path=f"{root_dir}/train_agbm", X=X_val, mean=mean['6S'], std=std['6S'], 
                                     mean_agb=mean_agb, std_agb=std_agb, transform=None)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=16, num_workers=8, pin_memory=True)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=16, num_workers=8, pin_memory=True)

    model = smp.UnetPlusPlus(encoder_name=encoder_name, decoder_attention_type=decoder_attention_type,
                             in_channels=S2_CHANNELS['6S']+S1_CHANNELS['6S'], classes=1, activation=None)

    freeze_encoder(model)

    s2s1_model = SentinelModel.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path, 
                                                    mean_agb=mean_agb, std_agb=std_agb,
                                                    lr=0.0005, wd=0.0001)

    # summary(s2s1_model.cuda(), (S2_CHANNELS['6S']+S1_CHANNELS['6S'], 256, 256)) 


#     wandb_logger = WandbLogger(save_dir=f'./models', name=f'{encoder_name}_6S_{decoder_attention_type}', 
#                                project=f'{encoder_name}_6S_{decoder_attention_type}')

    ## Define a trainer and start training:
    on_best_valid_loss = ModelCheckpoint(filename="{epoch}-{valid/loss}", mode='min', save_last=True,
                                         monitor='valid/loss', save_top_k=2)
    on_best_valid_rmse = ModelCheckpoint(filename="{epoch}-{valid/rmse}", mode='min', save_last=True,
                                         monitor='valid/rmse', save_top_k=2)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = [on_best_valid_loss, on_best_valid_rmse, lr_monitor]

    # Initialize a trainer
    trainer = Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=100, 
                      # logger=[wandb_logger], 
                      callbacks=checkpoint_callback)
    # Train the model ⚡
    trainer.fit(s2s1_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    checkpoint_path = r'./models/se_resnext50_32x4d_6S_None/qji032p2/checkpoints/loss=0.07499314099550247.ckpt'
    train_finetuned_model(checkpoint_path, '6S', "mit_b5", None)