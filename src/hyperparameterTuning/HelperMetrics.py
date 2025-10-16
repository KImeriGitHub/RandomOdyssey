from typing import Iterable, Union
import numpy as np
import polars as pl

class HelperMetrics:
    def __init__():
        pass

    @staticmethod
    def geometric_mean_safe(arr: Iterable[float]) -> float:
        """Compute a numerically stable geometric mean."""
        arr = np.asarray(list(arr), dtype=float)
        if arr.size == 0:
            return float("nan")
        minv = np.min(arr)
        shift = -minv + 1e-9 if minv <= 0 else 0.0
        return float(np.exp(np.mean(np.log(arr + shift))))

    @staticmethod
    def collapse_sl_tp(arr: np.ndarray, 
                        sl: Union[float, np.ndarray], 
                        tp: Union[float, np.ndarray]) -> np.ndarray:
        """
        For each row i:
        - if any arr[i, j] < sl[i] -> out[i] = sl[i]
        - elif any arr[i, j] > tp[i] -> out[i] = tp[i]
        - else out[i] = arr[i, -1]
        SL takes precedence if both occur.

        sl and tp can be scalars or length-n arrays.
        """
        arr = np.asarray(arr)
        n = arr.shape[0]

        # Normalize sl/tp to (n,) arrays (support scalar or length-n)
        if np.ndim(sl) == 0:
            slv = np.full(n, sl, dtype=arr.dtype)
        else:
            slv = np.asarray(sl).ravel()

        if np.ndim(tp) == 0:
            tpv = np.full(n, tp, dtype=arr.dtype)
        else:
            tpv = np.asarray(tp).ravel()

        below_sl = (arr < slv[:, None]).any(axis=1)
        above_tp = (arr > tpv[:, None]).any(axis=1)
        return np.where(below_sl, slv, np.where(above_tp, tpv, arr[:, -1]))

    @staticmethod
    def establish_datesMat(dates: pl.Series) -> np.ndarray:
        """
        Build a dense (D x N) one-hot 'dates matrix' where each column corresponds
        to a sample and each row to a unique date (in order of first appearance).
        Entry (d, n) == 1 iff sample n belongs to date-row d; else 0.

        Parameters
        ----------
        dates : array-like of length N
            Labels that group columns by date. Can be any hashable values.

        Returns
        -------
        M : np.ndarray, shape (D, N), dtype=np.uint8
        """
        dates = dates.to_numpy()
        # unique values, positions of first occurrence, and inverse indices
        uniq, idx_first, inv = np.unique(dates, return_index=True, return_inverse=True)
        order = np.argsort(idx_first)                    # order of first appearance
        # map original inverse codes (0..D-1 in sorted-uniq space) to "appearance order"
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size, dtype=order.dtype)
        row_ids = remap[inv]                             # length N, row index per column

        D, N = order.size, dates.size
        M = np.zeros((D, N), dtype=int)
        M[row_ids, np.arange(N)] = 1
        return M

    @staticmethod
    def evaluate_mask(mask: np.ndarray, dates: pl.Series, y: np.ndarray) -> float:
        """
        Compute the (equal-weight-per-date) geometric mean of y over the columns selected by mask.
        If a date has no selected columns, it is ignored. If no dates remain, returns 1.0.
        If mask selects nothing, returns 1.0.

        Parameters
        ----------
        mask  : np.ndarray of bool, shape (N,)
        dates : array-like of length N (passed to establish_datesMat_numpy)
        y     : np.ndarray of nonnegative/positive values, shape (N,)

        Returns
        -------
        float
            Geometric mean with equal weight per date (across dates that have ≥1 selected sample).
        """
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return 1.0

        M = HelperMetrics.establish_datesMat(dates)[:, mask]     # (D, K)
        y_sel = np.asarray(y, dtype=float)[mask]
        logy = np.log(np.clip(y_sel, 1e-8, None))

        rowsum = M.sum(axis=1)                           # counts per date (over selected cols)
        valid = rowsum > 0
        if not valid.any():
            return 1.0

        perdate_logmean = (M[valid] @ logy) / rowsum[valid]
        return float(np.exp(perdate_logmean.mean()))
