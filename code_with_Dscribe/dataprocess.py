# Standard library imports
import pickle
import random

# Third party imports
import ase
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

device_for_dataprocess=None
def set_device_for_dataprocess(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    global device_for_dataprocess
    device_for_dataprocess=device
def _get_device_for_dataprocess():
    global device_for_dataprocess
    if device_for_dataprocess is None:
        raise ValueError("Device is not set. Please call set_device_for_dataprocess() first.")
    return device_for_dataprocess

def read_csv(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize lists to store molecules
    molecule_list = []
    
    positions = []
    forces=[]
    elements=[]
    energy = None

    # Iterate through rows to group atoms into molecules
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        if pd.notna(row['Filename']):  # New molecule starts
            if len(elements)>0:  # Save previous molecule if exists
                molecule_list.append({
                    'forces':np.array(forces),
                    'elements':ase.Atoms(elements, positions=positions).get_atomic_numbers(),
                    'positions':np.array(positions),
                    'energy': energy})
            positions = []
            forces=[]
            elements=[]
            # Store molecule properties
            energy=float(row['SCF_Energy'])
        
        if pd.notna(row['Element']):  # Valid atom row
            positions.append(np.array([
                    float(row['X']),
                    float(row['Y']),
                    float(row['Z'])
                ]))
            forces.append(np.array([
                    float(row['Fx']) if pd.notna(row['Fx']) else 0.0,
                    float(row['Fy']) if pd.notna(row['Fy']) else 0.0,
                    float(row['Fz']) if pd.notna(row['Fz']) else 0.0
                ]))
            elements.append(row['Element'])

    # Add the last molecule
    if len(elements)>0:
        molecule_list.append({
                    'forces':np.array(forces),
                    'elements':ase.Atoms(elements, positions=positions).get_atomic_numbers(),
                    'positions':np.array(positions),
                    'energy': energy})
    return molecule_list
        
def get_all_elements(molecules):
    allelements = set()
    for molecule in molecules:
        allelements.update(molecule['elements'])
    return sorted(list(allelements))

def pad(batch_molecules):
    device=_get_device_for_dataprocess()
    # Get max number of atoms in this batch
    max_atoms = max(len(molecule['elements']) for molecule in batch_molecules)
    # Process and pad data for each molecule in batch
    E_batch_pad=torch.tensor([mol['energy'] for mol in batch_molecules], dtype=torch.float32)
    # Convert numpy arrays to tensors before padding
    force_tensors = [torch.tensor(mol['forces'], dtype=torch.float32) for mol in batch_molecules]
    positions_tensors = [torch.tensor(mol['positions'], dtype=torch.float32,requires_grad=True) for mol in batch_molecules]
    elements_batch_pad=torch.nn.utils.rnn.pad_sequence([torch.tensor(mol['elements'], dtype=torch.int32) for mol in batch_molecules],batch_first=True)
    force_batch_pad=torch.nn.utils.rnn.pad_sequence(force_tensors,batch_first=True)
    positions_batch_pad=torch.nn.utils.rnn.pad_sequence(positions_tensors,batch_first=True)
    mask_batch=[]
    for i, molecule in enumerate(batch_molecules):
        N = len(molecule['elements'])
        # Pad dD_dr: [N, N, 3, D] -> [max_atoms, max_atoms, 3, D]
        # Mask for valid atoms
        mask = torch.zeros(max_atoms, dtype=torch.bool)
        mask[:N] = 1
        mask_batch.append(mask)
    mask_batch=torch.stack(mask_batch)
    return {'energy':E_batch_pad.to(device),'forces':force_batch_pad.to(device),'elements':elements_batch_pad.to(device),'mask':mask_batch.to(device),'positions':positions_batch_pad.to(device)}

def stratified_sampling_by_force(molecule_list, bins,mode='uniform',sample_fraction=None,n_sample=None):
    """
    molecule_list: list of molecules
    forces_list: list of forces corresponding to molecules (could be max/mean force per molecule)
    bins: list of bin edges, e.g., [0, 0.001, 0.005, 0.01, 0.02, 0.05]
    sample_fraction: fraction of each bin to sample, e.g., 0.1 (10%)
    """
    # Step 1: Compute force magnitudes if not yet done
    forces_array = np.array([np.abs(mol['forces']).mean() for mol in molecule_list])  # shape (n_molecules,)
    assert mode in ['uniform','fraction'], "mode must be either 'uniform' or 'fraction'"
    if mode == 'uniform':
        assert sample_fraction is None, "sample_fraction must be provided if mode is 'uniform'"
        assert n_sample is not None, "n_sample must be provided if mode is 'uniform'"
    elif mode == 'fraction':
        assert sample_fraction is not None, "sample_fraction must be provided if mode is 'fraction'"
        assert n_sample is None, "n_sample must be provided if mode is 'fraction'"
    # Step 2: Digitize into bins
    bin_indices = np.digitize(forces_array, bins)

    # Step 3: Stratified sampling
    sampled_molecules = []

    for bin_id in range(1, len(bins)):
        # Select molecules in this bin
        bin_molecules = [mol for mol, idx in zip(molecule_list, bin_indices) if idx == bin_id]
        if mode == 'fraction':
            n_sample = int(len(bin_molecules) * sample_fraction)
        elif mode == 'uniform':
            n_sample = n_sample

        # Randomly sample
        if n_sample > 0 and len(bin_molecules) > 0:
            sampled = random.sample(bin_molecules, min(n_sample, len(bin_molecules)))
            sampled_molecules.extend(sampled)

    return sampled_molecules

def get_all_data(model,train_dataset,test_dataset):
    batch_size=256
    train_dataset_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad)
    val_dataset_loader= torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad)   
    '''if model is None, no prediction is done'''
    # Training set predictions
    if model is not None:
        model.eval()
    train_true_energies_per_atom = []
    train_pred_energies_per_atom = []
    train_true_forces = []
    train_pred_forces = []
    train_true_force_magnitude=[]
    train_pred_force_magnitude=[]
    test_true_energies_per_atom = []
    test_pred_energies_per_atom = []
    test_true_forces = []
    test_pred_forces = []
    test_true_force_magnitude=[]
    test_pred_force_magnitude=[]
    for batch in train_dataset_loader:
        mask=batch['mask']
        if model is not None:
            pred_energy, pred_force = model(batch)
            train_pred_energies_per_atom.extend((pred_energy/mask.sum(dim=1)).detach().cpu().numpy())
            train_pred_forces.extend(pred_force.detach().cpu().numpy()[batch['mask'].cpu().numpy()==1].flatten())
            train_pred_force_magnitude.extend(np.linalg.norm(pred_force.detach().cpu().numpy(), axis=-1)[batch['mask'].cpu().numpy()==1])
        train_true_energies_per_atom.extend((batch['energy']/mask.sum(dim=1)).cpu().numpy())
        train_true_forces.extend(batch['forces'].cpu().numpy()[batch['mask'].cpu().numpy()==1].flatten())
        train_true_force_magnitude.extend(np.linalg.norm(batch['forces'].cpu().numpy(), axis=-1)[batch['mask'].cpu().numpy()==1])
    # Test set predictions  

    for batch in val_dataset_loader:
        if model is not None:
            pred_energy, pred_force = model(batch)
            test_pred_energies_per_atom.extend((pred_energy/batch['mask'].sum(dim=1)).detach().cpu().numpy())
            test_pred_forces.extend(pred_force.detach().cpu().numpy()[batch['mask'].cpu().numpy()==1].flatten())
            test_pred_force_magnitude.extend(np.linalg.norm(pred_force.detach().cpu().numpy(), axis=-1)[batch['mask'].cpu().numpy()==1])
        test_true_energies_per_atom.extend((batch['energy']/batch['mask'].sum(dim=1)).cpu().numpy())
        test_true_forces.extend(batch['forces'].cpu().numpy()[batch['mask'].cpu().numpy()==1].flatten())
        test_true_force_magnitude.extend(np.linalg.norm(batch['forces'].cpu().numpy(), axis=-1)[batch['mask'].cpu().numpy()==1])
    # Convert to numpy arrays
    train_true_energies_per_atom = np.array(train_true_energies_per_atom)
    train_pred_energies_per_atom = np.array(train_pred_energies_per_atom)
    test_true_energies_per_atom = np.array(test_true_energies_per_atom)
    test_pred_energies_per_atom = np.array(test_pred_energies_per_atom)
    train_true_forces = np.array(train_true_forces)
    train_pred_forces = np.array(train_pred_forces)
    test_true_forces = np.array(test_true_forces)
    test_pred_forces = np.array(test_pred_forces)
    train_true_force_magnitude = np.array(train_true_force_magnitude)
    train_pred_force_magnitude = np.array(train_pred_force_magnitude)
    test_true_force_magnitude = np.array(test_true_force_magnitude)
    test_pred_force_magnitude = np.array(test_pred_force_magnitude)
    return train_true_energies_per_atom, train_pred_energies_per_atom, test_true_energies_per_atom, test_pred_energies_per_atom,\
        train_true_forces, train_pred_forces, test_true_forces, test_pred_forces,\
        train_true_force_magnitude, train_pred_force_magnitude, test_true_force_magnitude, test_pred_force_magnitude

def read_dataset(train_indices_path,test_indices_path,molecule_list_path):
    train_indices = torch.load(train_indices_path) 
    test_indices = torch.load(test_indices_path)
    molecule_list = pickle.load(open(molecule_list_path, 'rb'))

    train_dataset = torch.utils.data.Subset(molecule_list, train_indices)
    test_dataset = torch.utils.data.Subset(molecule_list, test_indices)
    allelements=get_all_elements(molecule_list)
    return train_dataset,test_dataset,allelements

def get_energy_per_atom(molecule_list):
    energies_per_atom=[]
    for molecule in molecule_list:
        num_atom = len(molecule['elements'])
        energies_per_atom.append(molecule['energy']/num_atom)
    return energies_per_atom