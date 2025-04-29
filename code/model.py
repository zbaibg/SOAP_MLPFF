# A standalone script to train the model, copied from the most updated version of the cust_model.ipynb
# Standard library imports
import glob
import os
import re

# Third party import
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Local imports
from dataprocess import get_energy_per_atom, set_device_for_dataprocess,pad
from soap_torch import SoapLayer
from stratified_sampler import StratifiedFixedSampler

# Set device
device_for_model = None
def set_device_for_model(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    global device_for_model
    device_for_model = device

def _get_device_for_model():
    global device_for_model
    if device_for_model is None:
        raise ValueError("Device is not set for the model, please set it by set_device_for_model() first")
    return device_for_model

class EnergyForceModel(nn.Module):
    def __init__(self, hidden_size,allelements,cutoff,n_max,l_max,debug=False):
        super().__init__()
        self.debug = debug
        #self.log_sigma_e = nn.Parameter(torch.tensor(0.0))
        #self.log_sigma_f = nn.Parameter(torch.tensor(0.0))
        self.soaplayer = SoapLayer(
            species=allelements,
            cutoff=cutoff,
            n_max=n_max,
            l_max=l_max,
            device=_get_device_for_model())
        #self.soaplayer = torch.compile(self.soaplayer) if torch.cuda.is_available() else self.soaplayer
        # Neural network to process SOAP features
        self.energy_net = nn.Sequential(
            nn.Linear(self.soaplayer.desc_dim, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
            )
        print('trainable parameters: ',sum(p.numel() for p in self.energy_net.parameters() if p.requires_grad))
    def forward(self, data,cal_force=True):
        positions = data['positions']
        elements = data['elements']
        mask = data['mask']
        batch_size = positions.shape[0]
        atom_num = positions.shape[1]
        
        #(Batch_N*Atom_N,3) 
        soap_features = self.soaplayer(pos=positions,Z=elements, atom_mask=mask)
        soap_dim = self.soaplayer.desc_dim
        #(Batch_N*Atom_N,soap_dim) to (Batch_N,Atom_N)
        atomic_energies = self.energy_net(soap_features.view(batch_size*atom_num,soap_dim)).squeeze(-1)
        atomic_energies = atomic_energies.view(batch_size,atom_num)


        if self.debug:
            print('shape mask', mask.shape)
            print('shape atomic_energies',(atomic_energies.shape))
            print('shape descriptors',(mask.shape))  
        # Sum to get total energy
        atomic_energies = atomic_energies * mask
        #(Batch_N,Atom_N) to (Batch_N)
        total_energy = atomic_energies.sum(dim=1).squeeze(-1)

        if cal_force:
            if self.debug:
                print("input.requires_grad =", positions.requires_grad)
                print("atomic_energies.requires_grad =", atomic_energies.requires_grad)
                print("total_energy.requires_grad =", total_energy.requires_grad)
            force = torch.autograd.grad(total_energy, positions, create_graph=True,grad_outputs=torch.ones_like(total_energy))[0]

            if self.debug:
                print('shape total_energy',(total_energy.shape))
                print('shape input',(positions.shape))
                print('shape force',(force.shape))
            return total_energy, force
        else:
            return total_energy, None

    def loss_function(self,pred_energy, pred_force, true_energy, true_forces,mask,weight_energy,weight_force):
        # The energy loss is normalized by the number of atoms of each molecule, as performed in allegro software
        energy_loss = torch.sqrt(F.mse_loss(pred_energy/mask.sum(dim=1), true_energy/mask.sum(dim=1)))  #(batch)
        force_diff = (pred_force - true_forces) ** 2    # (batch, n_atoms, 3)
        mask_force = mask.unsqueeze(-1)                       # (batch, n_atoms, 1)
        masked_diff = force_diff * mask_force
        force_loss = torch.sqrt(
            masked_diff.sum() / (mask_force.sum() * 3.0)
        ) 

        total_loss = weight_energy * energy_loss +  weight_force * force_loss
        return total_loss,energy_loss,force_loss

def train_model(model,optimizer, train_dataset, test_dataset,batch_size, last_saved_epoch, epochs, weight_energy,weight_force,energy_bins_to_sample_trainset=None):

    train_total_losses = []
    train_energy_losses = []
    train_force_losses = []
    val_total_losses = []
    val_energy_losses = []
    val_force_losses = []
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    if energy_bins_to_sample_trainset is not None:
        energies_train = get_energy_per_atom(train_dataset)
        train_sampler = StratifiedFixedSampler(energies_train, batch_size, energy_bins_to_sample_trainset)
    else:
        train_sampler = None
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=pad)
    val_dataset_loader= torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad)    
    if last_saved_epoch is None:
        last_saved_epoch=0
    for epoch in range(last_saved_epoch+1,epochs):
        assert not os.path.exists(f'cust_model_checkpoint_epoch_{epoch}.pt'),f"Checkpoint for epoch {epoch} exist, please delete it first"
        model.train()
        train_total_loss = 0
        train_energy_loss = 0
        train_force_loss = 0
        
        # Training
        with tqdm(range(len(train_dataset_loader)),desc='Training Batch',leave=True) as pbar_train_batch:
            for batch in train_dataset_loader:
                optimizer.zero_grad()
                # Forward pass
                pred_energy, pred_forces = model(batch)
                true_energy = batch['energy']
                true_forces = batch['forces']
                mask = batch['mask']
                # Calculate loss using defined loss function
                total_loss,energy_loss,force_loss = model.loss_function(pred_energy, pred_forces, true_energy, true_forces,mask,weight_energy,weight_force)
                # Backward pass
                total_loss.backward()
                optimizer.step()
                train_total_loss += total_loss.item()
                train_energy_loss += energy_loss.item()
                train_force_loss += force_loss.item()
                pbar_train_batch.set_postfix(train_loss=total_loss.item(),train_energy_loss=energy_loss.item(),train_force_loss=force_loss.item())
                pbar_train_batch.update(1) 
            avg_train_total_loss = train_total_loss / len(train_dataset_loader)
            avg_train_energy_loss = train_energy_loss / len(train_dataset_loader)
            avg_train_force_loss = train_force_loss / len(train_dataset_loader)
            train_total_losses.append(avg_train_total_loss)
            train_energy_losses.append(avg_train_energy_loss)
            train_force_losses.append(avg_train_force_loss)

            # Validation
        with tqdm(range(len(val_dataset_loader)),desc='Validation Batch',leave=True) as pbar_val_batch:
            if val_dataset_loader is not None:
                model.eval()
                val_total_loss = 0
                val_energy_loss = 0
                val_force_loss = 0
                for batch in val_dataset_loader:
                    # Forward pass
                    pred_energy, pred_forces = model(batch)
                    true_energy = batch['energy']
                    true_forces = batch['forces']
                    mask = batch['mask']
                    # Calculate loss using defined loss function
                    total_loss,energy_loss,force_loss = model.loss_function(pred_energy, pred_forces, true_energy, true_forces,mask,weight_energy,weight_force)
                    val_total_loss += total_loss.item()
                    val_energy_loss += energy_loss.item()
                    val_force_loss += force_loss.item()
                    pbar_val_batch.set_postfix(val_loss=total_loss.item(),val_energy_loss=energy_loss.item(),val_force_loss=force_loss.item())
                    pbar_val_batch.update(1)
                avg_val_total_loss = val_total_loss / len(val_dataset_loader)
                avg_val_energy_loss = val_energy_loss / len(val_dataset_loader)
                avg_val_force_loss = val_force_loss / len(val_dataset_loader)
                val_total_losses.append(avg_val_total_loss)
                val_energy_losses.append(avg_val_energy_loss)
                val_force_losses.append(avg_val_force_loss)
            scheduler.step(avg_val_total_loss)

            # Save model checkpoint after each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_total_loss,
                'train_energy_loss': avg_train_energy_loss,
                'train_force_loss': avg_train_force_loss,
                'val_loss': avg_val_total_loss,
                'val_energy_loss': avg_val_energy_loss,
                'val_force_loss': avg_val_force_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            # Save losses to a separate file for easier access later
            losses = {
                'epoch': epoch,
                'train_loss': avg_train_total_loss,
                'train_energy_loss': avg_train_energy_loss, 
                'train_force_loss': avg_train_force_loss,
                'val_loss': avg_val_total_loss,
                'val_energy_loss': avg_val_energy_loss,
                'val_force_loss': avg_val_force_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            torch.save(losses, f'losses_epoch_{epoch}.pt')
            torch.save(checkpoint, f'cust_model_checkpoint_epoch_{epoch}.pt')
        print(f'Epoch {epoch}: Train Loss: {avg_train_total_loss:.4e}, Train Energy Loss: {avg_train_energy_loss:.4e}, Train Force Loss: {avg_train_force_loss:.4e}, Val Loss: {avg_val_total_loss:.4e}, Val Energy Loss: {avg_val_energy_loss:.4e}, Val Force Loss: {avg_val_force_loss:.4e}, LR: {optimizer.param_groups[0]["lr"]:.4e}')
    return model
def create_or_load_optimizer(model,file_of_parameter=None,lr_for_new_optimizer=0.001):
    '''
    model: the model to be used for creating the optimizer, but the parameter of the optimizer can be loaded from the file_of_parameter
    file_of_parameter: the file of parameter to be loaded, if None, a new optimizer will be created
    lr_for_new_optimizer: the learning rate if a new optimizer is created when file_of_parameter is None
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_for_new_optimizer)
    if file_of_parameter is not None:
        optimizer.load_state_dict(torch.load(file_of_parameter)['optimizer_state_dict'])
    return  optimizer
def set_lr_for_optimizer(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def load_model(empty_model,file_of_parameter):
    empty_model.load_state_dict(torch.load(file_of_parameter)['model_state_dict'])
    return empty_model.to(_get_device_for_model())

def get_the_checkpoint_path_of_latest_epoch():
    checkpoint_files = glob.glob('cust_model_checkpoint_epoch_*.pt')

    # Extract epoch numbers and find the latest
    last_epoch = -1
    if checkpoint_files:
        for f in checkpoint_files:
            match = re.search(r'model_checkpoint_epoch_(\d+)\.pt', f)
            if match:
                epoch = int(match.group(1))
                last_epoch = max(last_epoch, epoch)
                
    if last_epoch>0:
        return f'cust_model_checkpoint_epoch_{last_epoch}.pt',last_epoch
    else:
        return None
    
    
    
def train_it(train_dataset,test_dataset,allelements,hidden_size,cutoff,n_max,l_max,batch_size,epochs,weight_energy,weight_force,initial_lr=0.001,set_lr_for_mid_epoch=False,energy_bins_to_sample_trainset=None):
    set_device_for_dataprocess(_get_device_for_model())
    empty_model = EnergyForceModel(hidden_size=hidden_size,allelements=allelements,cutoff=cutoff,n_max=n_max,l_max=l_max).to(_get_device_for_model())
    if get_the_checkpoint_path_of_latest_epoch() is None:
        optimizer = create_or_load_optimizer(empty_model,lr_for_new_optimizer=initial_lr)
        model=empty_model
        last_epoch=None
    else:
        model=load_model(empty_model,file_of_parameter=get_the_checkpoint_path_of_latest_epoch()[0]).to(_get_device_for_model())
        optimizer = create_or_load_optimizer(model,file_of_parameter=get_the_checkpoint_path_of_latest_epoch()[0],lr_for_new_optimizer=initial_lr)
        last_epoch=get_the_checkpoint_path_of_latest_epoch()[1]
        if set_lr_for_mid_epoch:
            set_lr_for_optimizer(optimizer,initial_lr)
    # Train the model
    model= train_model(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        last_saved_epoch=last_epoch,
        epochs=epochs,
        weight_energy=weight_energy,
        weight_force=weight_force,
        energy_bins_to_sample_trainset=energy_bins_to_sample_trainset
    )
    print("Training completed!")

