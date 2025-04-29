from model import train_it,set_device_for_model,EnergyForceModel
from dataprocess import set_device_for_dataprocess,read_dataset,get_energy_per_atom
import numpy as np
set_device_for_dataprocess()
set_device_for_model()

train_indices_path='bondenergy_train_indices.pt'
test_indices_path='bondenergy_test_indices.pt'
molecule_list_path='bondenergy_molecule_list_cust.pkl'
hidden_size=1024
cutoff=6
n_max=3
l_max=3
batch_size=256
epochs=3000
weight_energy=1
weight_force=0
initial_lr=0.001
set_lr_for_mid_epoch=False
num_bins_to_sample_trainset=10

train_dataset,test_dataset,allelements=read_dataset(train_indices_path,test_indices_path,molecule_list_path)
energies_per_atom=get_energy_per_atom(train_dataset)
min_energy_per_atom=min(energies_per_atom)
max_energy_per_atom=max(energies_per_atom)
print('min_energy_per_atom,max_energy_per_atom',min_energy_per_atom,max_energy_per_atom)
energy_bins_to_sample_trainset=np.linspace(min_energy_per_atom-1e-6,max_energy_per_atom+1e-6,num_bins_to_sample_trainset)
print('train_dataset molecule number',len(train_dataset))
print('train_dataset atom number',sum([len(molecule['elements']) for molecule in train_dataset]))
print('test_dataset molecule number',len(test_dataset))
print('test_dataset atom number',sum([len(molecule['elements']) for molecule in test_dataset]))

#create an empty model for other model to import when loading checkpoints
empty_model=EnergyForceModel(hidden_size,allelements,cutoff,n_max,l_max)

if __name__ == '__main__':

    train_it(train_dataset,
            test_dataset,
            allelements,
            hidden_size,cutoff,
            n_max,
            l_max,
            batch_size,
            epochs,
            weight_energy,
            weight_force,
            initial_lr,
            set_lr_for_mid_epoch,
            energy_bins_to_sample_trainset)
