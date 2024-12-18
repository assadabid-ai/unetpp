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

df = stratify_data(
    s2_path_train=f"{root_dir}/train_features_s2_6S", 
    agb_path=f"{root_dir}/train_agbm", 
    s2_path_test=f"{root_dir}/test_features_s2_6S", 
    test_size=96, 
    random_state=0
)
df.to_csv(os.path.join(f'./data/train_val_split_96_0.csv'), index=None)

X_train, X_val, X_test = (df["id"].loc[df["dataset"] == 0].tolist(),
                          df["id"].loc[df["dataset"] == 1].tolist(),
                          df["id"].loc[df["dataset"] == 2].tolist())
print(df["dataset"].value_counts())
print("Total Images: ", len(df))

mean_agb, std_agb = calcuate_mean_std(image_dir=f"{root_dir}/train_agbm", train_set=X_train, percent=100, channels=1, 
                                      nodata=None, data='agbm', log_scale=False)

mean, std = {}, {}


S2_PATH = f"{os.path.join(root_dir, 'train_features_s2_6S')}"
S1_PATH = f"{os.path.join(root_dir, 'train_features_s1_6S')}"

mean_s2, std_s2 = calcuate_mean_std(image_dir=S2_PATH, train_set=X_train, percent=5, channels=S2_CHANNELS['6S'], 
                                      nodata=0, data='S2', log_scale=False)
mean_s1, std_s1 = calcuate_mean_std(image_dir=S1_PATH, train_set=X_train, percent=5, channels=S1_CHANNELS['6S'], 
                                      nodata=None, data='S1', log_scale=False)

mean['6S'] = mean_s2 + mean_s1
std['6S'] = std_s2 + std_s1

with open('./data/mean.json', 'w') as f:
    json.dump(mean, f)
with open('./data/std.json', 'w') as f:
    json.dump(std, f)
with open('./data/mean_agb.json', 'w') as f:
    json.dump(mean_agb, f)
with open('./data/std_agb.json', 'w') as f:
    json.dump(std_agb, f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# Empty cache
torch.cuda.empty_cache()
print(torch.version.cuda)
torch.backends.cudnn.benchmark = False

def train_base_model(suffix, encoder_name, encoder_weights, decoder_attention_type):
    # wandb.finish()    
    
    train_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                       s1_path=f"{root_dir}/train_features_s1_6S",
                                       agb_path=f"{root_dir}/train_agbm", X=X_train, mean=mean['6S'], std=std['6S'], 
                                       mean_agb=mean_agb, std_agb=std_agb, 
                                       transform=A.Compose([A.HorizontalFlip(), A.VerticalFlip(), 
                                                            A.RandomRotate90(), A.Transpose(), A.ShiftScaleRotate()]))

    val_set = BioMasstersDatasetS2S1(s2_path=f"{root_dir}/train_features_s2_6S",
                                     s1_path=f"{root_dir}/train_features_s1_6S",
                                     agb_path=f"{root_dir}/train_agbm", X=X_val, mean=mean['6S'], std=std['6S'], 
                                     mean_agb=mean_agb, std_agb=std_agb, transform=None)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=16, num_workers=8, pin_memory=True)
    
    val_loader = DataLoader(val_set, shuffle=False, batch_size=16, num_workers=8, pin_memory=True)

    model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, 
                             decoder_attention_type=decoder_attention_type,
                             in_channels=S2_CHANNELS['6S']+S1_CHANNELS['6S'], classes=1, activation=None)

    s2s1_model = SentinelModel(model, mean_agb=mean_agb, std_agb=std_agb, lr=0.001, wd=0.0001)

    # summary(s2s1_model.cuda(), (S2_CHANNELS['6S']+S1_CHANNELS['6S'], 256, 256)) 

    # wandb_logger = WandbLogger(save_dir=f'./models', name=f'{encoder_name}_6S_{decoder_attention_type}', 
    #                            project=f'{encoder_name}_6S_{decoder_attention_type}')

    ## Define a trainer and start training:
    on_best_valid_loss = ModelCheckpoint(filename="{epoch}-{valid/loss}", mode='min', save_last=True,
                                         monitor='valid/loss', save_top_k=2)
    on_best_valid_rmse = ModelCheckpoint(filename="{epoch}-{valid/rmse}", mode='min', save_last=True,
                                         monitor='valid/rmse', save_top_k=2)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = [on_best_valid_loss, on_best_valid_rmse, lr_monitor]

    # Initialize a trainer
    trainer = Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=200, 
                      # logger=[wandb_logger], 
                      callbacks=checkpoint_callback)
    # Train the model ⚡
    trainer.fit(s2s1_model, train_dataloaders=train_loader, val_dataloaders=val_loader)



if __name__ == '__main__':
    train_base_model('6S', "mit_b5", "imagenet", None)

if __name__ == '__main__':
    checkpoint_path = r'./models/se_resnext50_32x4d_6S_None/qji032p2/checkpoints/loss=0.07499314099550247.ckpt'
    train_finetuned_model(checkpoint_path, '6S', "mit_b5", None)