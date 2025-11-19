from typing import Iterable, Union
from matplotlib import dates
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
    def collapse_sl_tp(
        arr: np.ndarray, 
        arr_low: np.ndarray,
        arr_high: np.ndarray,
        arr_open: np.ndarray,
        sl: Union[float, np.ndarray, None], 
        tp: Union[float, np.ndarray, None],
        sl_max: float = 0.995,
        tp_min: float = 1.005,
        spread_cost: float = 0.001,
        commission: float = 0.0000
    ) -> np.ndarray:
        """
        Collapse each row of a time series using time-ordered stop-loss / take-profit.

        For each row i with columns j = 0..T-1, find the earliest column where a threshold is crossed.
        Within a given column j,  priority is given in the following order (SL before TP):
        1) SL by low (non-gap):   arr_open[i,j] >= sl[i] and arr_low[i,j]  <  sl[i]    -> return sl[i] - spread_cost
        2) SL by open gap:        arr_open[i,j] <= sl[i]                               -> return arr_open[i,j]
        3) TP by high (non-gap):  arr_open[i,j] <= tp[i] and arr_high[i,j] >  tp[i]    -> return tp[i] - spread_cost
        4) TP by open gap:        arr_open[i,j] >= tp[i]                               -> return arr_open[i,j]        
        If no threshold is ever crossed, return arr[i, -1].
        Commission is applied to all executed prices/results.

        Notes
        -----
        - “First hit wins”: the earliest column index j with any hit determines the outcome.
        On the same column, SL checks are evaluated before TP checks (SL wins ties).
        - `sl` and `tp` may be scalars or shape (N,) (optionally (N,1)); they broadcast across columns.
        - Assumes a long trade convention: sl < 1.0, tp > 1.0, arrays ≈ 1.0 scale.
        - `sl_max` and `tp_min` are applied by clipping: sl := min(sl, sl_max); tp := max(tp, tp_min).
        - Fully vectorized; O(N·T). Output dtype matches `arr.dtype`.
        - arr_open[:,0] is fully 1.0. (entry prices)

        Parameters
        ----------
        arr : np.ndarray, shape (N, T)
            Row-wise time series of prices/returns.
        arr_low : np.ndarray, shape (N, T)
            Row-wise time series of lows (same shape as `arr`).
        arr_high : np.ndarray, shape (N, T)
            Row-wise time series of highs (same shape as `arr`).
        arr_open : np.ndarray, shape (N, T)
            Row-wise time series of opens (same shape as `arr`).
        sl : float or np.ndarray, shape (N,) or (N,1)
            Stop-loss level(s); broadcast across columns.
        tp : float or np.ndarray, shape (N,) or (N,1)
            Take-profit level(s); broadcast across columns.
        sl_max : float, optional
            Maximum allowed stop-loss level; values above are clipped (default: 0.990).
        tp_min : float, optional
            Minimum allowed take-profit level; values below are clipped (default: 1.005).
        spread_cost : float, optional
            Per-trade spread_cost applied to the executed fill/result (e.g., subtract from return; default: 0.005).
        commission : float, optional
            Per-trade commission applied to the executed fill/result (e.g., subtract from return; default: 0.0008).

        Returns
        -------
        np.ndarray, shape (N,)
            Collapsed per-row values according to the first threshold hit, or `arr[i, -1]` if no hit.

        """
        # Ensure 2D; cast (N,) -> (N,1)
        arr      = arr[:, None]      if arr.ndim == 1      else np.asarray(arr)
        arr_low  = arr_low[:, None]  if arr_low.ndim == 1  else np.asarray(arr_low,  dtype=arr.dtype)
        arr_high = arr_high[:, None] if arr_high.ndim == 1 else np.asarray(arr_high, dtype=arr.dtype)
        arr_open = arr_open[:, None] if arr_open.ndim == 1 else np.asarray(arr_open, dtype=arr.dtype)

        N, T = arr.shape
        
        has_sl = sl is not None
        has_tp = tp is not None

        if has_sl:
            if np.ndim(sl) == 0:
                sl = np.full((N,1), sl, dtype=arr.dtype)
            else:
                sl = np.asarray(sl, dtype=arr.dtype).reshape(N,1)

        if has_tp:
            if np.ndim(tp) == 0:
                tp = np.full((N,1), tp, dtype=arr.dtype)
            else:
                tp = np.asarray(tp, dtype=arr.dtype).reshape(N,1)

        # Guards
        if has_sl: sl = np.clip(sl, None, sl_max)
        if has_tp: tp = np.clip(tp, tp_min, None)

        # Fast path: single column
        if T == 1:
            low0, high0, last = arr_low[:,0], arr_high[:,0], arr[:,0].astype(arr.dtype).copy()
            m_sl = np.zeros(N, dtype=bool)
            m_tp = np.zeros(N, dtype=bool)
            if has_sl:
                slv = sl[:,0]
                m_sl = (low0  < slv)          # non-gap SL
            if has_tp:
                tpv = tp[:,0]
                m_tp = (high0 > tpv)          # non-gap TP
                m_tp = m_tp & (~m_sl) if has_sl else m_tp          # SL has priority over TP

            out = last
            if has_tp:
                out[m_tp] = tpv[m_tp] - spread_cost
            if has_sl: # !!! SL has priority over TP
                out[m_sl] = slv[m_sl] - spread_cost
            out *= (1.0 - commission) * (1.0 - commission)
            return out

        # Masks per column (broadcast (N,1) vs (N,T))
        sl_low_hit  = np.zeros_like(arr_open, dtype=bool) if not has_sl else ((arr_open >= sl) & (arr_low  <  sl))   # 1) SL by low (non-gap)
        sl_gap_hit  = np.zeros_like(arr_open, dtype=bool) if not has_sl else (arr_open <= sl)                         # 2) SL by open gap
        tp_high_hit = np.zeros_like(arr_open, dtype=bool) if not has_tp else ((arr_open <= tp) & (arr_high > tp))     # 3) TP by high (non-gap)
        tp_gap_hit  = np.zeros_like(arr_open, dtype=bool) if not has_tp else (arr_open >= tp)                         # 4) TP by open gap
        
        sl_any  = sl_low_hit | sl_gap_hit
        tp_any  = tp_high_hit | tp_gap_hit
        any_hit = sl_any | tp_any

        # First hit column j
        big = T
        first_j = np.where(any_hit.any(1), any_hit.argmax(1), big)
        idx = np.arange(N)
        has_hit = first_j < T

        # Helper: take mask value at first_j
        def take_at_first(mask: np.ndarray) -> np.ndarray:
            out = np.zeros(N, dtype=bool)
            if T and has_hit.any():
                out[has_hit] = np.take_along_axis(mask[has_hit], first_j[has_hit, None], axis=1).ravel()
            return out

        sl_at_first = take_at_first(sl_any)
        tp_at_first = has_hit & (~sl_at_first)

        sl_low_at_first  = take_at_first(sl_low_hit)  & sl_at_first
        sl_gap_at_first  = take_at_first(sl_gap_hit)  & sl_at_first
        tp_high_at_first = take_at_first(tp_high_hit) & tp_at_first
        tp_gap_at_first  = take_at_first(tp_gap_hit)  & tp_at_first

        out = arr[:, -1].astype(arr.dtype).copy()

        first_open = np.zeros(N, dtype=arr.dtype)
        first_open[has_hit] = arr_open[idx[has_hit], first_j[has_hit]]

        if tp_at_first.any():
            tp_price = np.empty(N, dtype=arr.dtype)
            if tp_high_at_first.any():
                tp_price[tp_high_at_first] = tp[tp_high_at_first, 0] - spread_cost
            if tp_gap_at_first.any():
                tp_price[tp_gap_at_first]  = first_open[tp_gap_at_first]
            out[tp_at_first] = tp_price[tp_at_first]
        
        if sl_at_first.any():  # SL has priority over TP
            sl_price = np.empty(N, dtype=arr.dtype)
            # Only fill indices we will read from
            if sl_low_at_first.any():
                sl_price[sl_low_at_first] = sl[sl_low_at_first, 0] - spread_cost
            if sl_gap_at_first.any():
                sl_price[sl_gap_at_first] = first_open[sl_gap_at_first]
            out[sl_at_first] = sl_price[sl_at_first]

        out *= (1.0 - commission) * (1.0 - commission)
        return out

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
        codes = dates.cast(pl.Int64).to_physical().to_numpy()  # 0..D-1
        N = codes.size
        D = (codes.max() + 1) if N else 0

        M = np.zeros((D, N), dtype=np.uint8)
        M[codes, np.arange(N)] = 1
        return M

    @staticmethod
    @DeprecationWarning
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

    @staticmethod
    def evaluate_mask_nullonempty(mask: np.ndarray, dates: pl.Series, y: np.ndarray) -> float:
        """
        Compute the geometric mean across dates of the *arithmetic* mean of y within each date,
        using equal weight per date. If a date has no selected columns, it is ignored.
        If no dates remain, returns 1.0. If mask selects nothing, returns 1.0.

        Parameters
        ----------
        mask  : np.ndarray of bool, shape (N,)
        dates : array-like of length N
        y     : np.ndarray of nonnegative/positive values, shape (N,)

        Returns
        -------
        float
            Geometric mean (equal weight per date) across dates that have ≥1 selected sample.
        """
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return 1.0

        # Integer codes for dates (same coding logic as establish_datesMat)
        codes = dates.cast(pl.Int64).to_physical().to_numpy()

        # Apply mask
        codes_sel = codes[mask]
        y_sel = np.asarray(y, dtype=float)[mask]

        if codes_sel.size == 0:
            return 1.0

        # Aggregate per date using arithmetic mean intraday
        counts_per_date = np.bincount(codes_sel)                         # number of selected samples per date
        sum_y_per_date = np.bincount(codes_sel, weights=y_sel)           # sum of y per date

        valid = counts_per_date > 0
        if not valid.any():
            return 1.0

        # Arithmetic mean per date
        perdate_mean = sum_y_per_date[valid] / counts_per_date[valid]

        # Geometric mean across dates of these per-date arithmetic means
        perdate_mean_clipped = np.clip(perdate_mean, 1e-8, None)
        log_means = np.log(perdate_mean_clipped)
        return float(np.exp(log_means.mean()))    
    
    @staticmethod
    def evaluate_mask_oneonempty(mask: np.ndarray, dates: pl.Series, y: np.ndarray) -> float:
        """
        Compute the (equal-weight-per-date) geometric mean of y over the columns selected by mask.
        If mask misses a date, that date is counted with 1.0.

        Parameters
        ----------
        mask  : np.ndarray of bool, shape (N,)
        dates : array-like of length N
        y     : np.ndarray of nonnegative/positive values, shape (N,)

        Returns
        -------
        float
            Geometric mean with equal weight per date (empty dates are evaluated to 1.0).
        """
        dateframe = dates.to_frame("date")
        meta_ext = dateframe.with_columns(pl.Series("y_tar", y))
        meta_ext_masked = meta_ext.filter(mask)

        # Group means on filtered data
        perday_eval_masked = (
            meta_ext_masked
            .group_by("date")
            .agg(pl.col("y_tar").mean().alias("y_mean"))
        )

        # All dates you care about (e.g. from the original meta_tr)
        all_dates = (
            dateframe
            .select("date")
            .unique()
            .sort("date")
        )

        # Left-join and force missing dates to 1.0
        perday_eval_masked_full = (
            all_dates
            .join(perday_eval_masked, on="date", how="left")
            .with_columns(pl.col("y_mean").fill_null(1.0))
        )

        gmean = np.exp(perday_eval_masked_full.select(pl.col("y_mean").log()).mean())
        return float(gmean)