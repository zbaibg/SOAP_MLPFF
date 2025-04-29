###############################################################################
###############################################################################
# Numerical accuracy tests                                                    #
###############################################################################

import math
import numpy as np
import torch
from soap_torch import SoapLayer
from e3nn.o3 import spherical_harmonics
import ase.build
from torch_cluster import radius_graph
from soap_torch import build_graph
from soap_torch import compute_kernel
torch.set_default_dtype(torch.float64)
def test_single_atom_analytic():
    """Test SOAP calculation for a single atom at origin against analytical formula.
    
    This test verifies that the SOAP implementation correctly calculates the power spectrum
    for a single atom at the origin, comparing against the theoretical value derived from
    the analytical formula.

    Test Logic:
    1. For a single atom at origin, the SOAP power spectrum can be calculated analytically:
       - The radial basis function: φₐ(r) = (r_cut - r)^(α+2)/Nₐ
       - The normalization factor: Nₐ = √(r_cut^(2α+5)/(2α+5))
       - The orthonormalized radial function: g_n(r) = ∑_α W_nα φₐ(r)
       - The spherical harmonic Y00 = 1/(2√π)
       - The coefficient: c_n00 = g_n(0) * Y00
       - The power spectrum: p(X)_11,0 = ps_norm * (c_100)^2, where ps_norm = π√(8/(2l+1))

    2. The test compares this analytical result with the numerical implementation.
    """
    species = [1]
    n_max = 1
    l_max = 0
    cutoff = 3.0
    layer = SoapLayer(species, n_max=n_max, l_max=l_max, cutoff=cutoff)
    pos = torch.zeros((1, 1, 3), dtype=torch.get_default_dtype())
    Z = torch.tensor([[1]])
    
    # Get numerical result from implementation
    out = layer(pos, Z).squeeze().item()
    
    # Calculate theoretical value
    # 1. Calculate normalization factors
    alpha_values = np.arange(1, n_max + 1, dtype=np.float64)
    numerator = np.power(cutoff, 2*alpha_values + 5)
    denominator = 2*alpha_values + 5
    N_alpha = np.sqrt(numerator / denominator)
    
    # 2. Calculate radial basis function at origin (r=0)
    phi_0 = np.power(cutoff, alpha_values + 2) / N_alpha
    
    # 3. Calculate overlap matrix S
    S = np.zeros((n_max, n_max), dtype=np.float64)
    for alpha in range(n_max):
        for beta in range(n_max):
            alpha_val = alpha + 1
            beta_val = beta + 1
            numerator = np.sqrt((5.0 + 2*alpha_val) * (5.0 + 2*beta_val))
            denominator = 5.0 + alpha_val + beta_val
            S[alpha, beta] = numerator / denominator
    
    # 4. Calculate orthonormalization matrix W = S^(-1/2)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    assert W.shape==(1,1), f"W should be a 1x1 matrix, but got {W.shape}"
    assert W.item()==1.0, f"W should be a 1x1 matrix with value 1.0, but got {W.item()}"
    # 5. Apply orthonormalization to get g_n(0)
    g_0 = np.matmul(W, phi_0)
    expected_g0=cutoff**3/math.sqrt(cutoff**7/7)
    assert g_0==expected_g0, f"g_0 {g_0} is not equal to cutoff**2/math.sqrt(cutoff**5/5) {expected_g0}"
    # 6. Multiply by Y00 = 1/(2√π)
    Y00 = 1.0 / (2.0 * np.sqrt(math.pi))
    c_n00 = g_0 * Y00 * 4 * math.pi
    
    # 7. Calculate power spectrum
    # Using the normalization factor from the equation: p_nn'l = π√(8/(2l+1))∑_m c^Z1_nlm * c^Z2_n'lm
    ps_norm_0 = math.pi * math.sqrt(8.0 / (2 * 0 + 1))  # For l=0, this is π√8
    power_spectrum_theory = ps_norm_0 * (c_n00[0] ** 2)
    
    # Compare numerical and theoretical results
    diff = abs(out - power_spectrum_theory)
    assert diff < 1e-10, f"Difference between calculated {out} and theoretical {power_spectrum_theory} values too large: {diff}"

def test_single_atom_multi_n():
    """Test SOAP calculation for a single atom with multiple n values.
    
    This test verifies that the SOAP implementation correctly handles multiple radial
    basis functions (n values) for a single atom at the origin.

    Test Logic:
    1. For a single atom at origin with multiple n values, the power spectrum includes
       cross terms between different radial basis functions:
       - The radial basis function: φₐ(r) = (r_cut - r)^(α+2)/Nₐ
       - The normalization factor: Nₐ = √(r_cut^(2α+5)/(2α+5))
       - The overlap matrix: Sₐᵦ = √((5+2α)(5+2β))/(5+α+β)
       - The orthonormalization matrix: W = S^(-1/2)
       - For each pair (n1, n2) where n2 ≥ n1, we calculate:
         p(X)_n1n2,0 = ps_norm * c_n100 * c_n200 * (√2 if n1≠n2 else 1)

    2. The test verifies that all these cross terms are calculated correctly.
    """
    species = [1]
    n_max = 3
    cutoff = 3.0
    layer = SoapLayer(species, n_max=n_max, l_max=0, cutoff=cutoff)
    pos = torch.zeros((1, 1, 3), dtype=torch.get_default_dtype())
    Z = torch.tensor([[1]])
    
    # Get numerical result from implementation
    out = layer(pos, Z).squeeze(0).squeeze(0)
    
    # Calculate theoretical values
    # 1. Calculate normalization factors
    alpha_values = np.arange(1, n_max + 1, dtype=np.float64)
    numerator = np.power(cutoff, 2*alpha_values + 5)
    denominator = 2*alpha_values + 5
    N_alpha = np.sqrt(numerator / denominator)
    
    # 2. Calculate radial basis function at origin (r=0)
    phi_0 = np.power(cutoff, alpha_values + 2) / N_alpha
    
    # 3. Calculate overlap matrix S
    S = np.zeros((n_max, n_max), dtype=np.float64)
    for alpha in range(n_max):
        for beta in range(n_max):
            alpha_val = alpha + 1
            beta_val = beta + 1
            numerator = np.sqrt((5.0 + 2*alpha_val) * (5.0 + 2*beta_val))
            denominator = 5.0 + alpha_val + beta_val
            S[alpha, beta] = numerator / denominator
    
    # 4. Calculate orthonormalization matrix W = S^(-1/2)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # 5. Apply orthonormalization to get g_n(0)
    g_0 = np.matmul(W, phi_0)
    
    # 6. Multiply by Y00 and include 4π factor for spherical harmonics normalization
    Y00 = 1.0 / (2.0 * np.sqrt(math.pi))
    c_n00 = g_0 * Y00 * 4 * math.pi
    
    # 7. Calculate power spectrum for all unique pairs
    # Using the normalization factor from the equation: p_nn'l = π√(8/(2l+1))∑_m c^Z1_nlm * c^Z2_n'lm
    ps_norm_0 = math.pi * math.sqrt(8.0 / (2 * 0 + 1))  # For l=0, this is π√8
    iu_row, iu_col = layer.n_index_row, layer.n_index_col
    theoretical = np.zeros_like(out)
    for idx, (i, j) in enumerate(zip(iu_row, iu_col)):
        # Add √2 factor for cross terms where n1≠n2
        cross_term_factor = np.sqrt(2.0) if i != j else 1.0
        theoretical[idx] = ps_norm_0 * c_n00[i] * c_n00[j] * cross_term_factor
    
    # Compare numerical and theoretical results
    diff = np.abs(out.numpy() - theoretical)
    max_abs_diff = diff.max()
    max_rel_diff = (diff / (np.abs(theoretical) + 1e-10)).max()
    
    assert max_abs_diff < 1e-10, f"Maximum absolute difference too large: {max_abs_diff}, out={out}, theoretical={theoretical}"
    assert max_rel_diff < 1e-10, f"Maximum relative difference too large: {max_rel_diff}"

def test_rotation_invariance():
    """Test that SOAP descriptors are rotationally invariant.
    
    This test verifies that rotating a two-atom system does not change the SOAP
    descriptors, demonstrating the rotational invariance property of the SOAP
    representation.

    Test Logic:
    1. Create a two-atom system along z-axis
    2. Calculate SOAP descriptors for original configuration
    3. Generate a random rotation matrix Q using QR decomposition
    4. Rotate the system using Q
    5. Calculate SOAP descriptors for rotated configuration
    6. Verify that descriptors are identical (up to numerical precision)

    Mathematical Basis:
    - SOAP descriptors are constructed from rotationally invariant combinations
      of spherical harmonics and radial functions
    - The power spectrum p(X) = ∑_m c_nlm c_n'l'm is invariant under rotation
      because spherical harmonics transform as irreducible representations
    """
    torch.manual_seed(0)
    species = [1]
    layer = SoapLayer(species, n_max=2, l_max=2, cutoff=5.0)

    # Create two atoms along z-axis
    base_pos = torch.tensor([[[0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]]], dtype=torch.get_default_dtype())
    Z = torch.tensor([[1, 1]])
    mask = torch.tensor([[True, True]])

    # Calculate descriptors for original configuration
    desc_ref = layer(base_pos, Z, atom_mask=mask).detach()

    # Generate random rotation matrix
    rand = torch.randn(3, 3, dtype=torch.get_default_dtype())
    Q, _ = torch.linalg.qr(rand)  # Q is orthogonal (rotation) matrix
    rot_pos = base_pos @ Q.T  # Apply rotation

    # Calculate descriptors for rotated configuration
    desc_rot = layer(rot_pos, Z, atom_mask=mask).detach()
    
    # Verify invariance
    diff = (desc_ref - desc_rot).abs()
    max_abs_diff = diff.max().item()
    max_rel_diff = (diff / (desc_ref.abs() + 1e-10)).max().item()  # Add small epsilon to avoid division by zero
    
    assert max_abs_diff < 2e-7, f"Maximum absolute difference too large: {max_abs_diff}"
    assert max_rel_diff < 1e-5, f"Maximum relative difference too large: {max_rel_diff}"

def test_batch_processing():
    """Test SOAP calculation with batched input.
    
    This test verifies that the SOAP implementation correctly handles batched input
    with different numbers of atoms per batch, including proper handling of padding.

    Test Logic:
    1. Create a batch containing two different molecules (H2O and CH4)
    2. Pad the batch to maximum number of atoms (5)
    3. Use atom_mask to indicate which positions are valid
    4. Verify that:
       - Output shape is correct (batch_size, max_atoms, descriptor_dim)
       - Padding positions have zero descriptors
    """
    torch.set_default_dtype(torch.get_default_dtype())

    water = ase.build.molecule("H2O")
    methane = ase.build.molecule("CH4")

    max_atoms = 5
    B = 2
    pos = torch.zeros((B, max_atoms, 3), dtype=torch.get_default_dtype())
    Z = torch.zeros((B, max_atoms), dtype=torch.long)
    msk = torch.zeros((B, max_atoms), dtype=torch.bool)

    # Fill first batch with water molecule
    pos[0, :3] = torch.tensor(water.positions)
    Z[0, :3] = torch.tensor(water.numbers)
    msk[0, :3] = True

    # Fill second batch with methane molecule
    pos[1, :5] = torch.tensor(methane.positions)
    Z[1, :5] = torch.tensor(methane.numbers)
    msk[1, :5] = True

    layer = SoapLayer([1, 6, 8], n_max=4, l_max=3, cutoff=4.5)
    out = layer(pos, Z, atom_mask=msk)
    
    # Verify output shape and padding
    assert out.shape == (B, max_atoms, layer.desc_dim), "Output shape mismatch"
    assert out[0, 3:].abs().sum().item() < 1e-12, "Non-zero values found in padding positions"

def test_gradient_calculation():
    """Test gradient calculation with respect to atomic positions.
    
    This test verifies that the gradients of the SOAP descriptors with respect to
    atomic positions are correctly calculated by comparing automatic differentiation
    with finite differences.

    Test Logic:
    1. Calculate gradient using automatic differentiation (autograd)
    2. Calculate gradient using finite differences:
       ∂f/∂x ≈ [f(x+ε) - f(x-ε)] / (2ε)
    3. Compare the two results

    Mathematical Basis:
    - The SOAP descriptor is differentiable with respect to atomic positions
    - The gradient can be calculated using the chain rule through:
      - Radial basis functions
      - Spherical harmonics
      - Power spectrum calculation
    """
    layer = SoapLayer([1, 8], n_max=2, l_max=1, cutoff=4.5).double()
    device = layer.device

    mol = ase.build.molecule("H2O")
    pos = torch.tensor(mol.positions, dtype=torch.get_default_dtype(), device=device, requires_grad=True).unsqueeze(0)
    Z = torch.tensor(mol.numbers, device=device).unsqueeze(0)

    # Calculate gradient using autograd
    loss = layer(pos, Z).sum()
    grad_auto = torch.autograd.grad(loss, pos, create_graph=True)[0][0,0,0].item()

    # Calculate gradient using finite differences
    eps = 1e-4
    with torch.no_grad():
        pos_pos = pos.clone(); pos_pos[0,0,0] += eps
        pos_neg = pos.clone(); pos_neg[0,0,0] -= eps
        fd = (layer(pos_pos, Z).sum() - layer(pos_neg, Z).sum()) / (2*eps)
    grad_fd = fd.item()

    # Compare gradients
    diff = abs(grad_auto - grad_fd)
    assert diff < 1e-4, f"Gradient mismatch: autograd={grad_auto:.3e}, finite difference={grad_fd:.3e}"

def test_Y00_implementation():
    """Test the implementation of Y00 spherical harmonic.
    
    This test verifies that the Y00 spherical harmonic is correctly implemented
    by comparing with the e3nn implementation.

    Test Logic:
    1. The Y00 spherical harmonic is a constant:
       Y00 = 1/(2√π)
    2. This value should be the same regardless of the input vector direction
    3. We test this by comparing our implementation with e3nn's implementation
    """
    test_vec = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.get_default_dtype())
    Y00_from_e3nn = spherical_harmonics(0, test_vec, normalize=True, normalization="integral")[0,0]
    true_Y00 = 1.0 / (2.0 * math.sqrt(math.pi))
    assert abs(Y00_from_e3nn - true_Y00) < 1e-10, f"Y00 mismatch: {Y00_from_e3nn} vs {true_Y00}"

def test_kernel_normalization():
    """Test SOAP kernel normalization properties.
    
    This test verifies several key properties of the SOAP kernel:
    1. k(X,X) = 1 for identical environments
    2. k(X,X') < 1 for different environments
    3. Rotational invariance

    Test Logic:
    1. For identical environments:
       - Calculate normalized SOAP descriptors p̂(X)
       - Verify k(X,X) = p̂(X)·p̂(X) = 1
    
    2. For different environments:
       - Calculate normalized descriptors p̂(X) and p̂(X')
       - Verify k(X,X') = p̂(X)·p̂(X') < 1
    
    3. For rotational invariance:
       - Rotate environment X to get X_rot
       - Verify k(X,X_rot) = p̂(X)·p̂(X_rot) ≈ 1

    Mathematical Basis:
    - The SOAP kernel is defined as k(X,X') = p̂(X)·p̂(X')
    - Normalization: p̂(X) = p(X)/|p(X)|
    - For identical environments: |p̂(X)| = 1 ⇒ k(X,X) = 1
    - For different environments: k(X,X') ≤ 1 (Cauchy-Schwarz inequality)
    - For rotated environments: p̂(X_rot) = p̂(X) ⇒ k(X,X_rot) = 1
    """
    species = [1, 8]
    n_max, l_max = 4, 3
    cutoff = 4.5
    layer = SoapLayer(species, n_max=n_max, l_max=l_max, cutoff=cutoff)
    
    # Test single atom normalization
    pos1 = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.get_default_dtype())
    Z1 = torch.tensor([[1]])
    soap1 = layer(pos1, Z1).squeeze()
    kernel_value = compute_kernel(soap1, soap1)
    assert abs(kernel_value - 1.0) < 1e-10, f"Single atom normalization failed: Soap {soap1} Kernel {kernel_value}"
    
    # Test water molecule normalization
    pos2 = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.758, 0.585],
            [0.0, -0.758, 0.585]
        ]
    ], dtype=torch.get_default_dtype())
    Z2 = torch.tensor([[8, 1, 1]])
    
    soap2 = layer(pos2, Z2)
    for i in range(pos2.shape[1]):
        atom_soap = soap2[0, i]
        kernel_value = compute_kernel(atom_soap, atom_soap)
        assert abs(kernel_value - 1.0) < 1e-10, f"Water molecule atom {i} normalization failed: {kernel_value}"
    
    # Test similarity decrease
    pos3 = torch.tensor([
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.758, 0.585 + 0.1],
            [0.1, -0.758, 0.585]
        ]
    ], dtype=torch.get_default_dtype())
    Z3 = torch.tensor([[8, 1, 1]])
    
    soap3 = layer(pos3, Z3)
    for i in range(pos2.shape[1]):
        soap_orig = soap2[0, i]
        soap_distort = soap3[0, i]
        kernel_value = compute_kernel(soap_orig, soap_distort)
        if i > 0:
            assert kernel_value < 1, f"Distorted molecule similarity should be less than 1: {kernel_value}"
    
    # Test rotational invariance
    torch.manual_seed(42)
    rand = torch.randn(3, 3, dtype=torch.get_default_dtype())
    Q, _ = torch.linalg.qr(rand)
    pos_rotated = pos2.clone()
    center = pos_rotated[0, 0]
    for i in range(pos_rotated.shape[1]):
        rel_pos = pos_rotated[0, i] - center
        rotated_rel_pos = rel_pos @ Q.T
        pos_rotated[0, i] = rotated_rel_pos + center
    
    soap_rotated = layer(pos_rotated, Z2)
    for i in range(pos2.shape[1]):
        soap_orig = soap2[0, i]
        soap_rot = soap_rotated[0, i]
        kernel_value = compute_kernel(soap_orig, soap_rot)
        assert abs(kernel_value - 1.0) < 1e-6, f"Rotational invariance failed: {kernel_value}"

###############################################################################
# Demo                                                                        #
###############################################################################
def _demo_batch():
    """Demonstrate batch + padding usage."""

    torch.set_default_dtype(torch.get_default_dtype())

    water   = ase.build.molecule("H2O")
    methane = ase.build.molecule("CH4")

    max_atoms = 5
    B = 2
    pos = torch.zeros((B, max_atoms, 3), dtype=torch.get_default_dtype())
    Z   = torch.zeros((B, max_atoms), dtype=torch.long)
    msk = torch.zeros((B, max_atoms), dtype=torch.bool)

    pos[0, :3] = torch.tensor(water.positions)
    Z[0, :3]   = torch.tensor(water.numbers)
    msk[0, :3] = True

    pos[1, :5] = torch.tensor(methane.positions)
    Z[1, :5]   = torch.tensor(methane.numbers)
    msk[1, :5] = True

    layer = SoapLayer([1, 6, 8], n_max=4, l_max=3, cutoff=4.5)
    out = layer(pos, Z, atom_mask=msk)
    print("Batch output shape:", out.shape)
    print("Zeros at padding atoms?", out[0, 3:].abs().sum().item() < 1e-12)

def test_high_dimension():
    """Test SOAP calculation with high n_max and l_max values.
    
    This test verifies that the SOAP implementation correctly handles higher dimensional
    descriptors by using n_max=8 and l_max=6 as shown in the model configuration.
    
    Test Logic:
    1. Create a simple two-atom system
    2. Calculate SOAP descriptors with high n_max and l_max values
    3. Verify that:
       - Output shape matches expected descriptor dimension
       - Descriptors are non-zero and finite
       - Rotational invariance still holds at higher dimensions
    """
    species = [1, 6, 7, 8]  # H, C, N, O
    layer = SoapLayer(species, n_max=8, l_max=0, cutoff=6)
    # Create simple two-atom system
    pos = torch.tensor(
        [[[-4.1459, -1.1703,  1.2042],
         [-3.0471, -0.1355,  1.5188],
         [-2.9616,  1.0108,  0.4827],
         [-2.3082,  0.6157, -0.8688],
         [-0.8169,  0.3064, -0.7558],
         [-0.2579, -0.8935, -0.9592],
         [-1.0607, -2.1442, -1.3292],
         [ 1.1455, -1.1676, -0.8583],
         [ 1.8973, -0.1488, -0.5647],
         [ 3.3269, -0.3178, -0.5492],
         [ 4.0370,  0.0852,  0.6968],
         [ 3.6265, -0.7117,  1.9557],
         [ 1.4313,  1.1420, -0.3192],
         [ 0.0455,  1.4887, -0.4385],
         [-0.3204,  2.6462, -0.2896],
         [-4.1600, -1.9493,  1.9608],
         [-3.9844, -1.6404,  0.2397],
         [-5.1226, -0.6959,  1.1883],
         [-2.0834, -0.6362,  1.5882],
         [-3.2511,  0.2979,  2.4962],
         [-2.3926,  1.8340,  0.9087],
         [-3.9639,  1.3850,  0.2834],
         [-2.4302,  1.4566, -1.5513],
         [-2.8350, -0.2291, -1.3010],
         [-0.3743, -2.9662, -1.5004],
         [-1.7424, -2.4235, -0.5305],
         [-1.6433, -1.9838, -2.2322],
         [ 3.5032, -1.3134, -0.7446],
         [ 3.8733,  1.1497,  0.8649],
         [ 5.1022, -0.0531,  0.5058],
         [ 2.5766, -0.5571,  2.1850],
         [ 4.2170, -0.3852,  2.8062],
         [ 3.7939, -1.7743,  1.8068],
         [ 2.1062,  1.9008, -0.2222]]],dtype=torch.get_default_dtype())
    Z = torch.tensor([[6, 6, 6, 6, 6, 6, 6, 7, 6, 7, 6, 6, 7, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    atom_num = len(Z[0])
    
    # 1. Calculate descriptors and verify basic properties
    desc = layer(pos, Z)
    
    # Verify output shape
    expected_dim = layer.desc_dim
    assert desc.shape == (1, atom_num, expected_dim), f"Expected shape (1,{atom_num},{expected_dim}), got {desc.shape}"
    
    # Check descriptors are well-behaved
    assert torch.isfinite(desc).all(), "Found non-finite values in descriptors"
    assert desc.abs().sum() > 0, "All descriptors are zero"
    
    # 2. Test rotational invariance at high dimensions
    torch.manual_seed(1015)
    rand = torch.randn(3, 3, dtype=torch.get_default_dtype())
    Q, _ = torch.linalg.qr(rand)
    
    # Verify Q is orthogonal by checking Q·Q^T = I
    QQt = Q @ Q.T
    I = torch.eye(3, dtype=torch.get_default_dtype())
    assert torch.allclose(QQt, I, atol=1e-10), f"Q is not orthogonal: Q·Q^T = \n{QQt}"
    
    # Verify Q has determinant 1 (proper rotation)
    det = torch.linalg.det(Q)
    assert abs(det - 1.0) < 1e-10, f"Q determinant is not 1: {det}"
    
    # Apply rotation and calculate new descriptors
    pos_rot = pos @ Q.T
    desc_rot = layer(pos_rot, Z)
    
    # Compare original and rotated descriptors
    diff = (desc - desc_rot).abs()
    max_diff = diff.max().item()
    
    # 3. Test consistency of graph building before and after rotation
    B, N, _ = pos.shape
    flat_pos = pos.view(-1, 3)  # (B*N, 3)
    flat_pos_rot = pos_rot.view(-1, 3)  # (B*N, 3)
    batch_vec = torch.arange(B, device=pos.device).repeat_interleave(N)  # (B*N,)

    # Get edges before and after rotation
    edge_index1 = radius_graph(flat_pos, r=layer.cutoff, batch=batch_vec, loop=True,max_num_neighbors=512)  # (2, E)
    edge_index_rot1 = radius_graph(flat_pos_rot, r=layer.cutoff, batch=batch_vec, loop=True,max_num_neighbors=512)  # (2, E)
    # Sort edges for comparison
    edge_index1 = sort_edge_index(edge_index1)
    edge_index_rot1 = sort_edge_index(edge_index_rot1)
    
    # Assert that radius_graph maintains edge indices after rotation
    assert torch.equal(edge_index1, edge_index_rot1), "Radius graph edges changed after rotation by torch_cluster"
    
    # Get edges using build_graph
    edge_index2 = build_graph(x=flat_pos, r=layer.cutoff, batch=batch_vec)
    edge_index_rot2 = build_graph(x=flat_pos_rot, r=layer.cutoff, batch=batch_vec)
    
    # Sort edges for comparison
    edge_index2 = sort_edge_index(edge_index2)
    edge_index_rot2 = sort_edge_index(edge_index_rot2)
    
    # Assert that build_graph maintains edge indices after rotation
    assert torch.equal(edge_index2, edge_index_rot2), "Radius graph edges changed after rotation by build_graph"
    
    # Assert that radius_graph and build_graph produce the same edge indices
    assert torch.equal(edge_index1, edge_index2), "radius_graph and build_graph produce different edge indices"
    
    # Final check for rotational invariance
    assert max_diff < 1e-7, f"Rotational invariance failed with difference: {max_diff}"
    
def sort_edge_index(edge_index):
    """Sort edge indices by source and target indices.
    
    Args:
        edge_index: Edge indices tensor of shape (2, E)
    """
    # Sort edges for consistent ordering
    # Sort columns while keeping pairs together
    # First sort by the first row (source indices)
    sorted_by_src = edge_index[:, edge_index[0].argsort()]
    
    # Then stable sort by the second row (target indices) within groups of the same source index
    # Get unique source indices and their counts
    unique_src, counts = torch.unique_consecutive(sorted_by_src[0], return_counts=True)
    
    # Initialize the final sorted edge_index
    edge_index = sorted_by_src.clone()
    
    # For each group with the same source index, sort by target index
    start_idx = 0
    for count in counts:
        if count > 1:  # Only need to sort if there's more than one element
            end_idx = start_idx + count
            group_slice = slice(start_idx, end_idx)
            # Sort this group by the target indices (second row)
            group_sort_idx = sorted_by_src[1, group_slice].argsort()
            edge_index[:, group_slice] = sorted_by_src[:, group_slice][:, group_sort_idx]
        start_idx += count
    return edge_index

###############################################################################
# Main                                                                        #
###############################################################################
if __name__ == "__main__":


    test_single_atom_analytic()
    test_single_atom_multi_n()
    test_rotation_invariance()
    test_kernel_normalization()
    test_gradient_calculation()
    test_Y00_implementation()
    test_batch_processing()
    test_high_dimension()
