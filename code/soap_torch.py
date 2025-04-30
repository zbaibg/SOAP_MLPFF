import math
from itertools import combinations_with_replacement
from typing import List, Optional
import numpy as np

import torch
from torch import nn
from e3nn.o3 import spherical_harmonics
import numpy as np
#np.set_printoptions(threshold=np.inf)  # threshold设为无穷大

try:
    from torch_cluster import radius_graph  # optional, for fast neighbour search
    _HAS_CLUSTER = True
except ImportError:
    _HAS_CLUSTER = False

###############################################################################
# FastSoapLayer – radius‑graph version (O(E) instead of O(N²))               #
###############################################################################

class SoapLayer(nn.Module):
    """Vectorised GPU SOAP layer using **torch_cluster.radius_graph**.

    *Neighbour list* is built in O(E) where E≈ρN, avoiding full N² matrices.
    Falls back to dense method if *torch_cluster* is unavailable.
    """

    def __init__(self, species: List[int], n_max: int = 8, l_max: int = 4,
                 cutoff: float = 5.0, device: str | torch.device = "cpu"):
        super().__init__()
        self.S, self.n_max, self.l_max = len(species), n_max, l_max  # S: number of species
        self.cutoff = float(cutoff)
        self.device = torch.device(device)

        # constant buffers --------------------------------------------------
        self.register_buffer("species_tensor", torch.tensor(species, dtype=torch.long))  # (S,)
        self.register_buffer("pair_ids", torch.tensor(list(combinations_with_replacement(range(self.S), 2)), dtype=torch.long))  # (P, 2)
        #number of species pairs
        self.P = self.pair_ids.size(0)  # P = S*(S+1)/2

        pref = [math.pi * math.sqrt(8.0 / (2 * l + 1)) for l in range(l_max + 1)]
        self.register_buffer("ps_norm", torch.tensor(pref, dtype=torch.get_default_dtype()))  # (l_max+1,)

        # Calculate radial normalization factor N_α = sqrt(r_cut^(2α+5)/(2α+5))
        alpha_values = torch.arange(1, n_max + 1, dtype=torch.get_default_dtype())  # (n_max,)
        norm_factor = torch.sqrt(cutoff**(2*alpha_values + 5) / (2*alpha_values + 5))  # (n_max,)
        self.register_buffer("norm_factor", norm_factor)
        
        self.register_buffer("_cutcoef", torch.tensor(math.pi / cutoff))  # scalar

      
      # Calculate orthonormalization matrix W and register as buffer
        self.register_buffer("W", self._calculate_orthonormalization_matrix(n_max))

        iu = torch.triu_indices(n_max, n_max)  # (2, n_max*(n_max+1)/2)
        iu_no_diag = torch.triu_indices(n_max, n_max, offset=1)  # (2, n_max*(n_max-1)/2)
        self.register_buffer("n_index_row", iu[0]); self.register_buffer("n_index_col", iu[1])
        self.register_buffer("n_index_row_no_diag", iu_no_diag[0]); self.register_buffer("n_index_col_no_diag", iu_no_diag[1])
        self.desc_dim = self.P * iu.shape[1] * (l_max + 1)

        self.to(self.device)


    # ------------------------------------------------------------------
    def _calculate_orthonormalization_matrix(self, n_max: int) -> torch.Tensor:
        # Use float64 for higher precision calculation
        S = np.zeros((n_max, n_max), dtype=np.float64)
        for a in range(1, n_max + 1):
            for b in range(1, n_max + 1):
                # S_αβ = sqrt((5+2α)(5+2β))/(5+α+β) as shown in the image
                S[a-1, b-1] = np.sqrt((5 + 2*a) * (5 + 2*b)) / (5 + a + b)
        
        # Calculate W = S^(-1/2) for orthonormalization using float64
        # W matrix is used to transform the basis functions to make them orthonormal
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        D_sqrt_inv = np.diag(eigenvalues**(-0.5))
        W = eigenvectors @ D_sqrt_inv @ eigenvectors.conj().T
        W = torch.from_numpy(W)
        # Convert back to default dtype for consistency with rest of model
        return W.to(torch.get_default_dtype())
    def _radial(self, r: torch.Tensor, l: int) -> torch.Tensor:
        # r: (E,) -> (E, n_max)
        # Implement the new radial basis function: φₐ(r) = (r_cut - r)^(α+2)/N_α
        
        alpha_values = torch.arange(1, self.n_max + 1, device=r.device, dtype=r.dtype)  # (n_max,)
        powers = alpha_values + 2  # (n_max,)
        
        # Calculate (r_cut - r)^(α+2) for each α value
        dr = (self.cutoff - r).unsqueeze(-1)  # (E, 1)
        basis = dr.pow(powers)  # (E, n_max)
        
        # Normalize by dividing by N_α
        basis = basis / self.norm_factor  # (E, n_max)
        
        # Apply cutoff (zero outside cutoff radius)
        basis = basis * (r < self.cutoff).float().unsqueeze(-1)  # (E, n_max)
        
        # Apply orthonormalization using W matrix directly in the radial function
        # basis has shape (E, n_max), W has shape (n_max, n_max)
        # For g_n(r) = Σ W_nα φₐ(r), we need to do (basis @ W.T) which gives W·φ in matrix form
        return basis @ self.W.to(basis.device).T  # (E, n_max)

    # ------------------------------------------------------------------
    def forward(self, pos: torch.Tensor, Z: torch.Tensor, *, atom_mask: Optional[torch.Tensor] = None):
        # pos: (B,N,3), Z: (B,N), atom_mask: (B,N)
        if pos.device.type != self.device.type or Z.device.type != self.device.type:
            raise RuntimeError("pos/Z not on {}".format(self.device))

        B, N, _ = pos.shape
        if atom_mask is None:
            atom_mask = (Z != 0)  # (B,N)
        atom_mask = atom_mask.to(dtype=torch.bool, device=self.device) # (B,N)

        # flatten with batch id -----------------------------------------
        flat_pos = pos.view(-1, 3)  # (B*N, 3)
        flat_Z = Z.view(-1)  # (B*N,)
        flat_mask = atom_mask.view(-1)  # (B*N,)
        batch_vec = torch.arange(B, device=self.device).repeat_interleave(N)  # (B*N,)

        active_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)  # (N_act,) namely (B*N,)
        pos_a = flat_pos[active_idx]  # (N_act, 3)
        Z_a = flat_Z[active_idx]  # (N_act,)
        bid_a = batch_vec[active_idx]  # (N_act,)
        N_act = pos_a.size(0)

        # ----------------------------------------------------------------
        # Build neighbour list (edges) via radius_graph or dense fallback
        # ----------------------------------------------------------------
        if _HAS_CLUSTER:
            edge_index = radius_graph(x=pos_a, r=self.cutoff, batch=bid_a, loop=True, max_num_neighbors=512)  # (2, E)
            #edge_index = sort_edge_index(edge_index)
            
            src, dst = edge_index  # (E,), (E,)
            dvec = pos_a[dst] - pos_a[src]  # (E, 3)
            dist = dvec.norm(dim=1)  # (E,)
        else:
            # dense fallback (memory heavy for large N)
            edge_index = build_graph(x=pos_a, r=self.cutoff, batch=bid_a)
            #edge_index = sort_edge_index(edge_index)

            # Sort edges for consistent ordering
            src, dst = edge_index  # (E,), (E,)
            dvec = pos_a[dst] - pos_a[src]  # (E, 3)
            dist = dvec.norm(dim=1)  # (E,))

        species_src = Z_a[src]  # (E,)

        # allocate coefficient tensor ----------------------------------
        max_m = 2 * self.l_max + 1
        c = pos_a.new_zeros((N_act, self.S, self.n_max, self.l_max + 1, max_m))  # (N_act, S, n_max, l_max+1, 2l_max+1)

        # 统一处理所有边 (包括self edges和neighbor edges) ----------------
        # 计算所有边的球谐函数
        Y_ls = [
            # the integral of the theta and phi of all space is 4*pi, this is included here.
            spherical_harmonics(l, dvec, normalize=True, normalization="integral") * 4 * math.pi # (E, 2l+1)
        
            for l in range(self.l_max + 1)
        ]
        #print('c0',c.cpu().numpy())
        
        for l, Y_l in enumerate(Y_ls):
            g_l = self._radial(dist, l)  # (E, n_max)
            contrib = torch.einsum("en,em->enm", g_l, Y_l)  # (E, n_max, 2l+1)
            for s_idx in range(self.S):
                m = species_src == self.species_tensor[s_idx]  # (E,)
                if m.any():
                    c[:, s_idx, :, l, : 2 * l + 1].scatter_add_(dim=0, index=dst[m].unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_max, 2*l+1), src=contrib[m])
                    #c[dst[m], s_idx, :, l, : 2 * l + 1] += contrib[m]   # (N_act, S, n_max, l_max+1, 2l_max+1)
        

        # ------------------------- ---------------------------------------
        # Power spectrum                                                
        # ----------------------------------------------------------------
        desc_act = pos_a.new_zeros((N_act, self.desc_dim))  # (N_act, desc_dim)
        offset = 0
        for (s1, s2) in self.pair_ids:
            c1, c2 = c[:, s1], c[:, s2]  # (N_act, n_max, l_max+1, max_m)
            for l in range(self.l_max + 1):
                ms = slice(0, 2 * l + 1)
                blk = self.ps_norm[l] * torch.einsum("bnm,bpm->bnp", c1[:, :, l, ms], c2[:, :, l, ms])  # (N_act, n_max, n_max)
                if s1 != s2:
                    blk *= math.sqrt(2.0) # normalization for only half s1,s2 matrix is kept
                blk[:,self.n_index_row_no_diag,self.n_index_col_no_diag] *= math.sqrt(2.0) # normalization for only half n1,n2 matrix is kept
                blk = blk[:, self.n_index_row, self.n_index_col].reshape(N_act, -1)  # (N_act, n_pairs) 
                D = blk.size(1)
                desc_act[:, offset : offset + D] = blk
                offset += D

        # scatter back ---------------------------------------------------
        out_flat = pos.new_zeros((B * N, self.desc_dim))  # (B*N, desc_dim)
        out_flat[active_idx] = desc_act
        return out_flat.view(B, N, self.desc_dim)  # (B, N, desc_dim)


def build_graph(x, r, batch):
    """Build edge indices using dense matrix operations.
    
    Args:
        x: Atom positions tensor of shape (N_act, 3)
        r: cutoff radius
        batch: Batch indices tensor of shape (N_act,)
    
    Returns:
        tuple: (src, dst) containing:
            - src: Source atom indices (E,)
            - dst: Destination atom indices (E,)
    """
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # (N_act, N_act, 3)
    dist_full = diff.norm(dim=-1)  # (N_act, N_act)
    same = batch.unsqueeze(0) == batch.unsqueeze(1)  # (N_act, N_act)
    within = (dist_full < r) & same  # (N_act, N_act)
    src, dst = within.nonzero(as_tuple=True)  # (E,), (E,)
    return torch.stack([src, dst], dim=0)

def compute_kernel(soap1, soap2, zeta=1):
    """Compute normalized SOAP kernel between two SOAP vectors.
    
    Args:
        soap1: First SOAP vector
        soap2: Second SOAP vector
        zeta: Kernel exponent parameter (default=1)
        
    Returns:
        Normalized kernel value K^SOAP(p,p') = (p·p'/√(p·p p'·p'))^ζ
    """
    dot_product = torch.dot(soap1, soap2)
    norm_soap1 = torch.norm(soap1)
    norm_soap2 = torch.norm(soap2)
    normalized_dot = dot_product / (norm_soap1 * norm_soap2)
    return normalized_dot.pow(zeta).item()
