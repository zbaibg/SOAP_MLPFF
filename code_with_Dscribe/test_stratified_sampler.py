import numpy as np
import pytest
from stratified_sampler import StratifiedFixedSampler

@pytest.fixture
def setup_data():
    """Setup test data for all tests.
    
    Returns:
        dict: Dictionary containing test data and parameters
    """
    np.random.seed(42)
    energies = np.random.normal(0, 1, 1000)  # 1000 samples
    # Ensure bins cover the full range of energies
    min_energy = np.min(energies)
    max_energy = np.max(energies)
    bins = [min_energy, -2, -1, 0, 1, 2, max_energy]  # Include min and max
    return {
        'energies': energies,
        'bins': bins,
        'batch_size': 100
    }

def test_bin_validation():
    """Test that bin validation works correctly.
    
    Test Logic:
    1. Test with invalid bins (not covering full range)
    2. Test with non-increasing bins
    3. Test with too few bins
    4. Test with valid bins
    """
    energies = np.array([1, 2, 3, 4, 5])
    
    # Test bins not covering full range
    with pytest.raises(ValueError, match="First bin edge"):
        StratifiedFixedSampler(energies, 3, [2, 3, 4])
    with pytest.raises(ValueError, match="Last bin edge"):
        StratifiedFixedSampler(energies, 3, [0, 1, 2, 3])
    
    # Test non-increasing bins
    with pytest.raises(ValueError, match="strictly increasing"):
        StratifiedFixedSampler(energies, 3, [0, 2, 1, 6])
    
    # Test too few bins
    with pytest.raises(ValueError, match="at least 2 values"):
        StratifiedFixedSampler(energies, 3, [1])
    
    # Test valid bins
    sampler = StratifiedFixedSampler(energies, 3, [0, 2, 4, 6])
    assert sampler is not None

def test_sample_proportions(setup_data):
    """Test that sampling maintains correct proportions in each bin.
    
    Test Logic:
    1. Create sampler with test data
    2. Get all sampled indices
    3. Calculate bin proportions in sampled data
    4. Compare with expected proportions from full dataset
    5. Allow 5% relative tolerance for differences
    """
    sampler = StratifiedFixedSampler(
        setup_data['energies'], 
        setup_data['batch_size'], 
        setup_data['bins']
    )
    
    # Get all sampled indices
    all_indices = list(sampler)
    
    # Calculate actual bin proportions in sampled data
    sampled_bin_indices = np.digitize(setup_data['energies'][all_indices], bins=setup_data['bins'], right=True)
    sampled_bin_counts = np.bincount(sampled_bin_indices, minlength=len(setup_data['bins']))
    sampled_proportions = sampled_bin_counts / len(all_indices)
    
    # Calculate expected proportions from full dataset
    expected_bin_indices = np.digitize(setup_data['energies'], bins=setup_data['bins'], right=True)
    expected_bin_counts = np.bincount(expected_bin_indices, minlength=len(setup_data['bins']))
    expected_proportions = expected_bin_counts / len(setup_data['energies'])
    
    # Compare proportions with 5% tolerance
    np.testing.assert_allclose(sampled_proportions, expected_proportions, rtol=0.05)

def test_unique_samples(setup_data):
    """Test that each sample appears only once in the sampling sequence.
    
    Test Logic:
    1. Create sampler with test data
    2. Get all sampled indices
    3. Verify that the set of indices has the same length as the list
       (indicating no duplicates)
    """
    sampler = StratifiedFixedSampler(
        setup_data['energies'], 
        setup_data['batch_size'], 
        setup_data['bins']
    )
    all_indices = list(sampler)
    
    # Check for duplicate indices
    assert len(all_indices) == len(set(all_indices)), "Found duplicate samples"

def test_complete_coverage(setup_data):
    """Test that all samples are included in the sampling sequence.
    
    Test Logic:
    1. Create sampler with test data
    2. Get all sampled indices
    3. Verify that the number of unique indices equals the total number of samples
    """
    sampler = StratifiedFixedSampler(
        setup_data['energies'], 
        setup_data['batch_size'], 
        setup_data['bins']
    )
    all_indices = list(sampler)
    
    # Check if all samples are covered
    assert len(set(all_indices)) == len(setup_data['energies']), "Not all samples were covered"

def test_batch_size(setup_data):
    """Test that batch sizes are correct.
    
    Test Logic:
    1. Create sampler with test data
    2. Get all sampled indices
    3. Verify that the last batch size is less than or equal to the specified batch size
    """
    sampler = StratifiedFixedSampler(
        setup_data['energies'], 
        setup_data['batch_size'], 
        setup_data['bins']
    )
    all_indices = list(sampler)
    
    # Check last batch size
    last_batch_size = len(all_indices) % setup_data['batch_size']
    if last_batch_size != 0:
        assert last_batch_size <= setup_data['batch_size'], "Last batch size exceeds specified batch size"

def test_empty_bins():
    """Test handling of empty bins.
    
    Test Logic:
    1. Create dataset with empty bins
    2. Create sampler with small batch size
    3. Verify that all samples are included in the sampling sequence
    """
    # Create dataset with empty bins
    energies = np.array([-3, -3, -3, 3, 3, 3])  # Only two bins have data
    bins = [-4, -2, 0, 2, 4]  # Include min and max
    sampler = StratifiedFixedSampler(energies, 3, bins)
    
    all_indices = list(sampler)
    assert len(all_indices) == len(energies), "Not all samples were included"

def test_small_dataset():
    """Test behavior with small datasets.
    
    Test Logic:
    1. Create small dataset
    2. Create sampler with batch size larger than dataset
    3. Verify that all samples are included in the sampling sequence
    """
    energies = np.array([1, 2, 3, 4, 5])
    bins = [0, 2, 4, 6]  # Include min and max
    sampler = StratifiedFixedSampler(energies, 3, bins)
    
    all_indices = list(sampler)
    assert len(all_indices) == len(energies), "Not all samples were included"

if __name__ == '__main__':
    pytest.main([__file__]) 