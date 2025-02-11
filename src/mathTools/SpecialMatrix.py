import numpy as np
import scipy.sparse as sp

class SpecialMatrix():
    def __init__():
        pass
    
    @staticmethod
    def boundedDiagonal_oneNetCol(n: int, k: int):
        """Builds a banded diagonal matrix (CSR) with column sums = 1."""
        # Offsets
        offs = range(-k, k + 1)

        # Each column c will have 1 + min(c, k) + min(n-1-c, k) ones
        # We'll create each diagonal's data directly.
        diags = []
        for offset in offs:
            # How many elements in this diagonal?
            length = n - abs(offset)
            # Fill with 1.0
            diag_data = np.ones(length, dtype=float)
            # Column indices for these diagonal entries
            start_col = max(0, -offset)
            end_col   = start_col + length
            # Normalize each position by count of ones in that column
            for local_i, col in enumerate(range(start_col, end_col)):
                # # of ones in column = 1 + min(col, k) + min((n-1-col), k)
                diag_data[local_i] /= (1 + min(col, k) + min(n - 1 - col, k))
            diags.append(diag_data)

        return sp.diags(diagonals=diags, offsets=offs, shape=(n, n), format='csr')
    
    def boundedDiagonal_oneNetCol_apply(x: np.ndarray, k: int) -> np.ndarray:
        """
        Equivalent of A_sparse @ x, where A_sparse is from boundedDiagonal_oneNetCol_apply but without building/storing A.
        Each column c in A has 1/(1 + min(c,k) + min(n-1-c,k)) on rows [c-k ... c+k].
        """
        n = len(x)
        y = np.zeros_like(x)
        for c in range(n):
            denom = 1 + min(c, k) + min(n - 1 - c, k)  # how many ones in col c
            val = x[c] / denom
            start = max(0, c - k)
            end = min(n, c + k + 1)
            y[start:end] += val
        return y
    
    @staticmethod
    def boundedDiagonal_zeroNetCol(n: int, k: int):
        """
        PRE: n is the size of the matrix. k is the number of off-diagonals on each side.
        POST: Returns a sparse matrix with 2*k+1 diagonals. The main diagonal is 1.0.
        Desc:
            The matrix has 2*k+1 diagonals. Every column adds up to zero.
        """

        # --- Banded form ---
        # Create an array with 2*k+1 rows (one for each diagonal) and n columns.
        D = np.zeros((2*k + 1, n), dtype=float)
        D[k, :] = 1.0  # Set the main diagonal to 1.0

        # Fill in the off-diagonals:
        for offset in range(1, k + 1):
            D[k + offset, :n - offset] = 1.0  # Lower diagonal (offset below main)
            D[k - offset, offset:] = 1.0        # Upper diagonal (offset above main)

        # Normalize each column: every 1.0 becomes 1/(number of ones in that column)
        D /= (D == 1).sum(axis=0)

        # Subtract 1.0 from the main diagonal (located at row index k)
        D[k, :] -= 1.0

        # --- Sparse matrix using scipy.sparse ---
        # Define the offsets for the diagonals
        offs = range(-k, k + 1)

        # Build the list of diagonals for the sparse matrix.
        # For each offset, the correct slice of D is taken:
        diags = [
            D[k - off, max(0, off): n + min(0, off)]
            for off in offs
        ]

        # Create the sparse matrix in CSR format.
        return sp.diags(diags, offs, shape=(n, n), format='csr')
    