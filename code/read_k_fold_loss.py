import torch
import glob
import pandas as pd
from model import load_model
from train_it import empty_model
df=pd.DataFrame(columns=['fold','epoch','train_loss','train_energy_loss','train_force_loss','val_loss','val_energy_loss','val_force_loss','learning_rate','trainable_parameter'])
fold_cross_validation=10

model_file=glob.glob('best_fold*_cust_model_checkpoint_epoch_*.pt')[0]
model=load_model(empty_model,model_file)
trainable_parameter=sum(p.numel() for p in model.energy_net.parameters() if p.requires_grad)
for i in range(1,fold_cross_validation+1):
    files=glob.glob(f'best_fold{i}_losses_epoch_*.pt')
    assert len(files)==1
    file=files[0]
    df.loc[i,'fold']=i
    df.loc[i,'epoch']=torch.load(file)['epoch']
    df.loc[i,'train_loss']=torch.load(file)['train_loss']
    df.loc[i,'train_energy_loss']=torch.load(file)['train_energy_loss']
    df.loc[i,'train_force_loss']=torch.load(file)['train_force_loss']
    df.loc[i,'val_loss']=torch.load(file)['val_loss']
    df.loc[i,'val_energy_loss']=torch.load(file)['val_energy_loss']
    df.loc[i,'val_force_loss']=torch.load(file)['val_force_loss']
    df.loc[i,'learning_rate']=torch.load(file)['lr']
    
ave_epoch=df['epoch'].mean()
ave_train_loss=df['train_loss'].mean()
ave_train_energy_loss=df['train_energy_loss'].mean()
ave_train_force_loss=df['train_force_loss'].mean()
ave_val_loss=df['val_loss'].mean()
ave_val_energy_loss=df['val_energy_loss'].mean()
ave_val_force_loss=df['val_force_loss'].mean()
ave_learning_rate=df['learning_rate'].mean()

std_epoch=df['epoch'].std()
std_train_loss=df['train_loss'].std()
std_train_energy_loss=df['train_energy_loss'].std()
std_train_force_loss=df['train_force_loss'].std()
std_val_loss=df['val_loss'].std()
std_val_energy_loss=df['val_energy_loss'].std()
std_val_force_loss=df['val_force_loss'].std()
std_learning_rate=df['learning_rate'].std()

df.loc[fold_cross_validation+1,'fold']='mean'
df.loc[fold_cross_validation+1,'epoch']=ave_epoch
df.loc[fold_cross_validation+1,'train_loss']=ave_train_loss
df.loc[fold_cross_validation+1,'train_energy_loss']=ave_train_energy_loss
df.loc[fold_cross_validation+1,'train_force_loss']=ave_train_force_loss
df.loc[fold_cross_validation+1,'val_loss']=ave_val_loss
df.loc[fold_cross_validation+1,'val_energy_loss']=ave_val_energy_loss
df.loc[fold_cross_validation+1,'val_force_loss']=ave_val_force_loss
df.loc[fold_cross_validation+1,'learning_rate']=ave_learning_rate

df.loc[fold_cross_validation+2,'fold']='std'
df.loc[fold_cross_validation+2,'epoch']=std_epoch
df.loc[fold_cross_validation+2,'train_loss']=std_train_loss
df.loc[fold_cross_validation+2,'train_energy_loss']=std_train_energy_loss
df.loc[fold_cross_validation+2,'train_force_loss']=std_train_force_loss
df.loc[fold_cross_validation+2,'val_loss']=std_val_loss
df.loc[fold_cross_validation+2,'val_energy_loss']=std_val_energy_loss
df.loc[fold_cross_validation+2,'val_force_loss']=std_val_force_loss
df.loc[fold_cross_validation+2,'learning_rate']=std_learning_rate

df.loc[fold_cross_validation+3,'fold']='trainable_parameter'
df.loc[fold_cross_validation+3,'trainable_parameter']=trainable_parameter

df.to_csv('k_fold_loss.csv',index=False)
