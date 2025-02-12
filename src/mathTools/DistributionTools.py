import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.optimize import lsq_linear

from src.mathTools.SpecialMatrix import SpecialMatrix

class DistributionTools():
    def __init__():
        pass
    
    @staticmethod
    def establishMatchingWeight(sam_tr: np.array, sam_te: np.array, n_bin: int = 10) -> np.array:
        """
        PRE: sam_tr and sam_te are the training and test samples; imp is the importance vector.
        POST: Returns a nonnegative weight vector (length=num_sam_tr) obtained via NNLS so that for each
              feature and for each of n_bin equally‐spaced bins (in the sorted order of sam_tr[:,feat]),
              the sum over the bin equals the target bin sum computed from the original sorted weights.
        """
        num_sam_tr, n_feat = sam_tr.shape

        # Compute sorted weights for each feature (memory-friendly per-column processing)
        weights_sorted = np.zeros((num_sam_tr, n_feat), dtype=float)
        for i in range(n_feat):
            sorted_tr = np.sort(sam_tr[:, i])
            sorted_te = np.sort(sam_te[:, i])
            # Compute bin differences via searchsorted (skip first index)
            lo = np.searchsorted(sorted_te, sorted_tr[:-1], side="right")
            hi = np.searchsorted(sorted_te, sorted_tr[1:], side="left")
            weights_sorted[1:, i] = np.abs(hi - lo)
            weights_sorted[0, i] = 0
            # Optional smoothing via convolution with a uniform window
            window = np.ones(sam_te.shape[0]) / sam_te.shape[0]
            weights_sorted[:, i] = np.convolve(weights_sorted[:, i], window, mode='same')
            # Scale to preserve total mass
            weights_sorted[:, i] *= (num_sam_tr / weights_sorted[:, i].sum())

        # Build the sparse constraint matrix A_sparse and target vector b.
        # For each feature, we impose that in the sorted order (using permutation p),
        # the sum over each of n_bin bins equals the corresponding sum from weights_sorted.
        rows, cols, data, b_vals = [], [], [], []
        row_counter = 0
        for i in range(n_feat):
            p = np.argsort(sam_tr[:, i])           # Sorting permutation for feature i
            sorted_w = weights_sorted[:, i]         # Already in sorted order
            # Split sorted indices into n_bin (nearly) equal bins
            bins = np.array_split(np.arange(num_sam_tr), n_bin)
            for bin_idx in bins:
                # For row corresponding to feature i and this bin, mark ones in columns given by p[bin_idx]
                indices = p[bin_idx]
                rows.extend([row_counter] * len(indices))
                cols.extend(indices.tolist())
                data.extend(np.ones(len(indices)))
                # The target is the sum of sorted_w over these bin positions
                b_vals.append(sorted_w[bin_idx].sum())
                row_counter += 1

        A_sparse = coo_matrix((data, (rows, cols)), shape=(n_bin * n_feat, num_sam_tr)).tocsr()
        b = np.array(b_vals)

        # Solve the nonnegative least-squares problem: min ||A_sparse*x - b||_2  s.t. x >= 0.
        # lsq_linear handles bounds and works efficiently with sparse A.
        sol = lsq_linear(A_sparse, b, bounds=(0, num_sam_tr), lsmr_tol='auto', verbose=0)
        res = sol.x
        
        del A_sparse, b, rows, cols, data, b_vals, weights_sorted, lo, hi, p, bins, sorted_te, sorted_tr
        
        res = sol.x
        res = res * num_sam_tr / res.sum()
        return res