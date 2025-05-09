import glob
import pickle
import re
from model import train_it,set_device_for_model,EnergyForceModel
from dataprocess import set_device_for_dataprocess,read_dataset,get_energy_per_atom,get_all_elements
import numpy as np
set_device_for_dataprocess()
set_device_for_model()

# Configuration parameters
mode = 'cross-validation' # 'cross-validation' or 'holdout'
fold_of_cross_validation=10 # only used when mode is 'cross-validation'
if mode == 'cross-validation':
    save_unconverged_models=False # if False only save the best model to save space
else:
    save_unconverged_models=True # if True save all models
molecule_list_path='bondenergy_molecule_list_cust.pkl'
hidden_size=1024
cutoff=6
n_max=3
l_max=3
batch_size=64
epochs=3000
weight_energy=1
weight_force=0
initial_lr=0.001
set_lr_for_mid_epoch=False
num_bins_to_sample_trainset=10
early_stopping_patience=30 # None if early_stopping_patience is not used

molecule_list=pickle.load(open(molecule_list_path,'rb'))
allelements=get_all_elements(molecule_list)
empty_model=EnergyForceModel(hidden_size, allelements, cutoff, n_max, l_max)

if __name__ == '__main__':
    if mode == 'cross-validation':
        files=glob.glob('best_fold*_cust_model_checkpoint_epoch_*.pt')
        finished_folds=[]    
        for file in files:
            fold_finish=re.search(r'best_fold(\d+)_cust_model_checkpoint_epoch_(\d+).pt',file).group(1)
            finished_folds.append(int(fold_finish))
        for fold in range(1, fold_of_cross_validation+1):
            if fold in finished_folds:
                continue
            print(f"Running fold {fold}/{fold_of_cross_validation}")
            test_indices_path = f'bondenergy_test_indices_fold{fold}.pt'
            train_indices_path = f'bondenergy_train_indices_fold{fold}.pt'
            
            train_dataset, test_dataset, allelements = read_dataset(train_indices_path, test_indices_path, molecule_list_path)
            energies_per_atom = get_energy_per_atom(train_dataset)
            min_energy_per_atom = min(energies_per_atom)
            max_energy_per_atom = max(energies_per_atom)
            print(f'Fold {fold} - min_energy_per_atom, max_energy_per_atom: {min_energy_per_atom}, {max_energy_per_atom}')
            energy_bins_to_sample_trainset = np.linspace(min_energy_per_atom-1e-6, max_energy_per_atom+1e-6, num_bins_to_sample_trainset)
            print(f'Fold {fold} - train_dataset molecule number: {len(train_dataset)}')
            print(f'Fold {fold} - train_dataset atom number: {sum([len(molecule["elements"]) for molecule in train_dataset])}')
            print(f'Fold {fold} - test_dataset molecule number: {len(test_dataset)}')
            print(f'Fold {fold} - test_dataset atom number: {sum([len(molecule["elements"]) for molecule in test_dataset])}')
            
            # Create an empty model for other model to import when loading checkpoints
            empty_model = EnergyForceModel(hidden_size, allelements, cutoff, n_max, l_max)
            
            train_it(train_dataset,
                    test_dataset,
                    allelements,
                    hidden_size, cutoff,
                    n_max,
                    l_max,
                    batch_size,
                    epochs,
                    weight_energy,
                    weight_force,
                    initial_lr,
                    set_lr_for_mid_epoch,
                    energy_bins_to_sample_trainset,
                    early_stopping_patience,
                    model_save_prefix=f"fold{fold}_")
        print(f"Finished {fold_of_cross_validation} folds")
    else:
        # holdout mode
        train_indices_path='bondenergy_train_indices.pt'
        test_indices_path='bondenergy_test_indices.pt'
        
        train_dataset, test_dataset, allelements = read_dataset(train_indices_path, test_indices_path, molecule_list_path)
        energies_per_atom = get_energy_per_atom(train_dataset)
        min_energy_per_atom = min(energies_per_atom)
        max_energy_per_atom = max(energies_per_atom)
        print('min_energy_per_atom, max_energy_per_atom', min_energy_per_atom, max_energy_per_atom)
        energy_bins_to_sample_trainset = np.linspace(min_energy_per_atom-1e-6, max_energy_per_atom+1e-6, num_bins_to_sample_trainset)
        print('train_dataset molecule number', len(train_dataset))
        print('train_dataset atom number', sum([len(molecule['elements']) for molecule in train_dataset]))
        print('test_dataset molecule number', len(test_dataset))
        print('test_dataset atom number', sum([len(molecule['elements']) for molecule in test_dataset]))

        # Create an empty model for other model to import when loading checkpoints
        train_it(train_dataset,
                test_dataset,
                allelements,
                hidden_size, cutoff,
                n_max,
                l_max,
                batch_size,
                epochs,
                weight_energy,
                weight_force,
                initial_lr,
                set_lr_for_mid_epoch,
                energy_bins_to_sample_trainset,
                early_stopping_patience,
                model_save_prefix='',
                save_unconverged_models=save_unconverged_models)
