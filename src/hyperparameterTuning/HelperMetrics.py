from typing import Iterable, Union
from matplotlib import dates
import numpy as np
import polars as pl

import logging
logger = logging.getLogger(__name__)

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
    def collapse_sl_tp_dep(
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
    def collapse_sl_tp(
        arr: np.ndarray, 
        arr_low: np.ndarray,
        arr_high: np.ndarray,
        arr_open: np.ndarray,
        sl: np.ndarray, 
        tp: np.ndarray,
        spread_cost: float = 0.000,
        commission: float = 0.0000
    ) -> np.ndarray:
        """
        Summary
        -----
        Collapse per-step OHLC, stop-loss, and take-profit information into a single
        exit price per path, assuming long positions with SL priority over TP and
        per-side transaction costs.

        Notes
        -----
        - arr inputs are 2D arrays of shape (N, T) or will be cast to (N,1), where N is the number
           of paths and T is the number of time steps. 
        - For each path, the first time a stop-loss (SL) or take-profit (TP) is hit
          determines the exit price; subsequent hits are ignored.
        - SL has priority over TP if both are touched within the same bar.
        - Non-gap fills (in-bar hits) are priced at the SL/TP level minus `spread_cost`.
        - Gap exits (where the open is already beyond SL/TP) are filled at the bar
          open and do **not** incur `spread_cost`, reflecting MOO/MOC executions.
        - If no SL/TP is hit for a path, the exit price defaults to the last value
          in `arr` along the time axis.
        - A proportional `commission` is applied per side, so the final exit price
          is multiplied by `(1 - commission) ** 2`.

        Parameters
        -----
        arr : np.ndarray
            Base price series of shape (N, T), typically close prices, used to
            derive the default exit (last column) when no SL/TP is hit.
        arr_low : np.ndarray
            Low prices for each bar, shape (N, T).
        arr_high : np.ndarray
            High prices for each bar, shape (N, T).
        arr_open : np.ndarray
            Open prices for each bar, shape (N, T).
        sl : np.ndarray
            Stop-loss levels for each bar, shape (N, T) or (N,) or float.
            Assumes long positions with SL below the entry/open.
        tp : np.ndarray
            Take-profit levels for each bar, shape (N, T) or (N,) or float.
            Assumes long positions with TP above the entry/open.
        spread_cost : float, optional
            Per-exit spread cost applied to non-gap hits (SL/TP touched within
            the bar range), expressed in price units. Defaults to 0.0.
        commission : float, optional
            Proportional commission rate per side (e.g. 0.0005 for 5 bps). Applied
            multiplicatively on both entry and exit as `(1 - commission) ** 2`.
            Defaults to 0.0.

        Returns
        -----
        out : np.ndarray
            1D array of shape (N,) containing the final exit price per path,
            after applying SL/TP logic, spread costs (for non-gap hits), and
            commissions.
        """
        # Ensure 2D
        arr      = np.asarray(arr)
        arr_low  = np.asarray(arr_low,  dtype=arr.dtype)
        arr_high = np.asarray(arr_high, dtype=arr.dtype)
        arr_open = np.asarray(arr_open, dtype=arr.dtype)
        sl       = np.asarray(sl,       dtype=arr.dtype)
        tp       = np.asarray(tp,       dtype=arr.dtype)
        
        if np.ndim(arr) == 1:
            arr = arr[:, None]
            arr_low  = arr_low[:, None]
            arr_high = arr_high[:, None]
            arr_open = arr_open[:, None]
        
        N, T = arr.shape
        
        if np.ndim(sl) == 0:
            sl = np.full((N, T), sl, dtype=arr.dtype)
        if np.ndim(tp) == 0:
            tp = np.full((N, T), tp, dtype=arr.dtype)
        
        if np.ndim(sl) == 1:
            sl = np.repeat(sl[:, None], T, axis=1)
        if np.ndim(tp) == 1:
            tp = np.repeat(tp[:, None], T, axis=1)
            
        assert arr_low.shape == (N, T)
        assert arr_high.shape == (N, T)
        assert arr_open.shape == (N, T)
        assert sl.shape == (N, T)
        assert tp.shape == (N, T)
        
        out = arr[:,-1].copy()
        mask_hit = np.zeros((N,), dtype=bool)
        for t in range(T):
            m_sl_low  = (arr_open[:, t] > sl[:, t]) & (arr_low[:, t]  <= sl[:, t])    # SL by low (non-gap)
            m_sl_gap  = (arr_open[:, t] <= sl[:, t])                                 # SL by open gap
            m_tp_high = (arr_open[:, t] < tp[:, t]) & (arr_high[:, t] >= tp[:, t])    # TP by high (non-gap)
            m_tp_gap  = (arr_open[:, t] >= tp[:, t])                                 # TP by open gap

            new_hits = ~mask_hit  # rows not hit yet
            
            m_sl = (m_sl_low | m_sl_gap) 

            sl_low_hits  = new_hits & m_sl_low
            sl_gap_hits  = new_hits & m_sl_gap
            tp_high_hits = new_hits & (m_tp_high & (~m_sl))  # SL has priority over TP
            tp_gap_hits  = new_hits & (m_tp_gap  & (~m_sl))  # SL has priority over TP

            if sl_low_hits.any():
                out[sl_low_hits]  = sl[sl_low_hits, t] - spread_cost
            if sl_gap_hits.any():
                out[sl_gap_hits]  = arr_open[sl_gap_hits, t] # no costs due to MOO and MOC trades
            if tp_high_hits.any():
                out[tp_high_hits] = tp[tp_high_hits, t] - spread_cost
            if tp_gap_hits.any():
                out[tp_gap_hits]  = arr_open[tp_gap_hits, t]  # no costs due to MOO and MOC trades

            mask_hit |= (sl_low_hits | sl_gap_hits | tp_high_hits | tp_gap_hits)

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
            logger.error("UNSORTED DATES: Mask selects nothing; returning 1.0")
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
    def evaluate_mask_oneonempty(mask: np.ndarray, dates: pl.Series, y: np.ndarray, gmean_onday: bool = False) -> float:
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
        if mask is None or not mask.any():
            return 1.0
        
        if not dates.is_sorted():
            return 0.0
        dateframe = dates.to_frame("date")
        meta_ext = dateframe.with_columns(pl.Series("y_tar", y))
        meta_ext_masked = meta_ext.filter(mask)

        # Group means on filtered data
        if gmean_onday:
            perday_eval_masked = meta_ext_masked.group_by("date").agg(
                pl.col("y_tar").log().mean().exp().alias("y_mean")
            )
        else:
            perday_eval_masked = meta_ext_masked.group_by("date").agg(pl.col("y_tar").mean().alias("y_mean"))
        

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