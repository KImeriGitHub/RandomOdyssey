import numpy as np

from src.hyperparameterTuning.HelperMetrics import HelperMetrics

class HelperSLTP:
    def __init__(self):
        pass
    
    @staticmethod
    def optimize_sl_tp(
        arr: np.ndarray, 
        arr_low: np.ndarray,
        arr_high: np.ndarray,
        arr_open: np.ndarray,
        *,
        n_grid: int = 15,            # number of quantile points
        sl_max: float = 0.995,       # SL upper cap
        tp_min: float = 1.005,       # TP lower cap
        spread_cost: float = 0.001,
        commission: float = 0.0000,
    ) -> tuple[float, float, dict]:
        """
        Returns (best_sl, best_tp, info) maximizing mean(out) where:
            out = collapse_sl_tp(..., sl, tp, ...)

        SL candidates: [-inf, quantiles(arr[:,-1]) clipped to <= sl_max, sl_max]
        TP candidates: [tp_min, quantiles(arr[:,-1]) clipped to >= tp_min, +inf)

        Notes:
        - -inf for SL and +inf for TP are treated as "no stop" (passed as None).
        - Pairs violating common-sense constraints are skipped:
            if finite(sl) and sl >= 1.0  -> skip
            if finite(tp) and tp <= 1.0  -> skip
            if both finite and sl >= tp  -> skip
        """
        # --- candidate grids from quantiles of the last column ---
        last = np.concatenate((arr_low[:, -1], arr_high[:, -1]))
        squish_power = 1.3
        squish_pwr_inv = 1.0 / squish_power
        qs_sl = np.linspace(0.001**squish_pwr_inv, 0.495**squish_pwr_inv, n_grid)**squish_power
        qs_tp = 1.0 - qs_sl[::-1]

        qvals_sl = np.quantile(last, qs_sl)
        # SL: (-inf .. sl_max]
        stretch_factor = 0.99
        sl_cands = np.unique(
            np.concatenate((
                qvals_sl[qvals_sl <= sl_max]*stretch_factor,
                np.array([sl_max])
            ))
        )

        # TP: [tp_min .. +inf)
        qvals_tp = np.quantile(last, qs_tp)
        stretch_factor = 1.01
        tp_cands = np.unique(
            np.concatenate((
                np.array([tp_min]),
                qvals_tp[qvals_tp >= tp_min]*stretch_factor,
            ))
        )

        def _collapse(sl_val, tp_val):
            # interpret infinities as "no stop"
            sl_arg = np.min(last) if not np.isfinite(sl_val) else float(sl_val)
            tp_arg = np.max(last) if not np.isfinite(tp_val) else float(tp_val)
            return HelperMetrics.collapse_sl_tp(
                arr, arr_low, arr_high, arr_open,
                sl=sl_arg, tp=tp_arg, spread_cost=spread_cost, commission=commission
            )

        best = (-np.inf, None, None, None)  # (mean, sl, tp, std)

        for sl_val in sl_cands:
            for tp_val in tp_cands:
                out = _collapse(sl_val, tp_val)
                m = float(np.exp(np.mean(np.log(out))))
                if m > best[0]:
                    best = (m, float(sl_val), float(tp_val), float(np.std(out)))

        if best[1] is None:
            # fallback: no valid pair found -> use "no SL/TP"
            out = _collapse(-np.inf, np.inf)
            return -np.inf, np.inf, {"mean": float(np.mean(out)), "std": float(np.std(out)), "stage": "fallback"}

        mean_star, sl_star, tp_star, std_star = best
        return sl_star, tp_star, {
            "mean": mean_star,
            "std": std_star,
            "params": {
                "n_grid": n_grid, "sl_max": sl_max, "tp_min": tp_min,
                "spread_cost": spread_cost, "commission": commission
            }
        }
    
    @staticmethod
    def perfect_sl_tp(
        arr: np.ndarray, 
        arr_low: np.ndarray,
        arr_high: np.ndarray,
        arr_open: np.ndarray, # keeping for consistency, not used
        *,
        tp_min: float = 1.005,
        tp_buffer_pct: float = 0.05,
        sl_max: float = 0.995,
        sl_buffer_pct: float = 0.05,
        sl_min: float | None = 0.8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute "perfect" row-wise stop-loss (SL) and take-profit (TP) levels
        from normalized OHLC time series.

        Method
        ------
        1. Take-profit:
           - For each row, find the maximum high in `arr_high`.
           - Enforce a minimum TP of `tp_min`.
           - If this max high exceeds `tp_min` plus its buffer
             `(tp_min - 1) * tp_buffer_pct`, set TP to the reached level
             minus its own buffer. Otherwise, fall back to `tp_min`.
        2. Stop-loss:
           - For each row, locate the index of the maximum close in `arr`.
           - Compute the running minimum of `arr_low` and read the minimum
             up to that index.
           - If the TP threshold was crossed, set SL to this minimum minus
             its buffer `(1 - min_price) * sl_buffer_pct`.
           - If not crossed, set SL to the last close in `arr` minus its
             buffer `(1 - last_close) * sl_buffer_pct`.
        3. Clamp SL between `sl_min` and `sl_max`, and TP to be at least
           `tp_min`.

        Parameters
        ----------
        arr : np.ndarray, shape (N, T)
            Row-wise time series of normalized close prices.
        arr_low : np.ndarray, shape (N, T)
            Row-wise time series of normalized low prices.
        arr_high : np.ndarray, shape (N, T)
            Row-wise time series of normalized high prices.
        arr_open : np.ndarray, shape (N, T)
            Row-wise time series of normalized open prices (unused).
        tp_min : float
            Minimum TP level to consider.
        tp_buffer_pct : float
            Fractional buffer applied to TP: (tp - 1) * tp_buffer_pct.
        sl_max : float
            Upper cap for SL (prevents SL above ~1).
        sl_buffer_pct : float
            Fractional buffer applied to SL: (1 - price) * sl_buffer_pct.
        sl_min : float | None
            Lower cap for SL (prevents excessively deep stops).

        Returns
        -------
        sl : np.ndarray, shape (N,)
            Row-wise stop-loss levels.
        tp : np.ndarray, shape (N,)
            Row-wise take-profit levels.
        """

        # --- Helpers
        def btp(val, pct):  # Buffer for take-profit
            return (val - 1.0) * pct

        def bsl(val, pct):  # Buffer for stop-loss
            return (1.0 - val) * pct

        # --- Take-profit ---
        arr_high_max = np.max(arr_high, axis=1)
        tp = np.maximum(arr_high_max, tp_min)

        tp_min_buffer_adj = tp_min + btp(tp_min, tp_buffer_pct)
        high_above_tpminbuffer = arr_high_max >= tp_min_buffer_adj

        # If we crossed the threshold, pull TP down by its buffer
        tp = np.where(high_above_tpminbuffer, tp - btp(tp, tp_buffer_pct), tp_min)

        # --- Stop-loss ---
        arr_argmax = np.argmax(arr, axis=1) 
        cummin = np.minimum.accumulate(arr_low, axis=1)  # (N, T)
        mins_to_argmax = cummin[np.arange(arr.shape[0]), arr_argmax]  # (N,)

        # If TP-threshold crossed: set SL below the min up to argmax
        sl_cross = mins_to_argmax - bsl(mins_to_argmax, sl_buffer_pct)

        # If not crossed: set SL below the last CLOSE price
        last = arr[:, -1]
        sl_else = last - bsl(last, sl_buffer_pct)

        sl = np.where(high_above_tpminbuffer, sl_cross, sl_else)

        # --- Safety clamps ---
        sl = np.clip(sl, sl_min, sl_max)
        tp = np.clip(tp, tp_min, None)

        return sl, tp
    
    def _stat_sl_step(
        self,
        ytr_step,
        ytr_last,
        ytr_open,
        sl_min=0.7,
        sl_max=1.5
    ) -> float:
        """
        ytr_step is mapped to ytr_last. Then sort ytr_step =: x.
        Through the mapping we get z := Mapped(ytr_last) and o := Mapped(ytr_open).

        Given arrays x,z,o (same length), we:
            1. Sort x ascending and apply the same permutation to z and o.
            2. For each index idx, compute:
                score(idx) = sum(np.minimum(x[idx], o[:idx+1])) - sum(z[:idx+1])
            3. Find idx that maximizes score(idx).

        The unconstrained optimal stop loss level would be:
            sl_raw = x[idx]

        Returns
        -------
        float
            The capped optimal stop loss level sl, clipped.
        """
        ytr_step = np.asarray(ytr_step)
        ytr_last = np.asarray(ytr_last)
        ytr_open = np.asarray(ytr_open)
        x = ytr_step.astype(float)
        z = ytr_last.astype(float)
        o = ytr_open.astype(float)
        
        n = len(x)
        if n <= 1: return sl_min

        # sort by x ascending
        asort = np.argsort(x)
        x = x[asort]
        z = z[asort]
        o = o[asort]

        # cumulative sums of z
        z_sum = np.cumsum(z)
        
        # left_sum[i] = sum(min(x[i], o[:i+1]))
        left_sum = np.array([np.minimum(x[i], o[:i+1]).sum() for i in range(n)])
        
        # score(idx) = sum(np.minimum(x[idx], o[:idx+1])) - sum(z[:idx+1])
        scores = left_sum - z_sum

        best_idx = int(np.argmax(scores))
        
        sl_end = float(np.clip(x[best_idx], sl_min, sl_max))
        return sl_end
    
    def _stat_tp_step(
        self,
        ytr_step, 
        ytr_last, 
        ytr_open,
        tp_min=0.7,
        tp_max=2.0
    ) -> float:
        """
        ytr_step is mapped to ytr_last. Then sort ytr_step =: x.
        Through the mapping we get z := Mapped(ytr_last) values.

        Given arrays z and x (same length), we:
            1. Sort x ascending and apply the same permutation to z.
            2. For each index idx, compute:
                score(idx) = sum(np.maximum(x[idx], o[idx:])) - sum(z[idx:])
            3. Find idx that maximizes score(idx).

        The unconstrained optimal take profit level would be:
            tp_raw = x[idx]

        Returns
        -------
        float
            The floored optimal take profit level tp, clipped.
        """
        ytr_step = np.asarray(ytr_step)
        ytr_last = np.asarray(ytr_last)
        ytr_open = np.asarray(ytr_open)
        x = ytr_step.astype(float)
        z = ytr_last.astype(float)
        o = ytr_open.astype(float)
        
        n = len(x)
        if n <= 1: return tp_max
        
        # sort by x ascending
        asort = np.argsort(x)
        x = x[asort]
        z = z[asort]
        o = o[asort]

        # z_sum[i] = sum(z[i:])
        z_sum = np.cumsum(z[::-1])[::-1]

        # left_sum[i] = sum(max(x[i], o[idx:]))
        left_sum = np.array([np.maximum(x[i], o[i:]).sum() for i in range(n)])

        # score(idx) = sum(np.maximum(x[idx], o[idx:])) - sum(z[idx:])
        scores = left_sum - z_sum

        best_idx = int(np.argmax(scores))
        
        tp_end = float(np.clip(x[best_idx], tp_min, tp_max))
        return tp_end
      
    @staticmethod
    def calc_conditional(
        ytr_close: np.ndarray,
        ytr_low: np.ndarray,
        ytr_high: np.ndarray,
        ytr_open: np.ndarray, # keeping for consistency, not used
        nS_te: int,
        *,
        sl_min: float = 0.7,
        sl_max: float = 1.5,
        tp_min: float = 0.7,
        tp_max: float = 2.0,
        sl_init: float | None = None,
        tp_init: float | None = None,
        include_live_mask: bool = True,
    ) -> tuple[np.ndarray, ...]:
        nS_tr, _ = ytr_close.shape
        
        # Init step
        if sl_init is not None:
            sl_vec_0 = sl_init
        else:
            sl_vec_0 = HelperSLTP()._stat_sl_step(ytr_low[:,0], ytr_close[:,-1], ytr_open[:,0], sl_min=sl_min, sl_max=0.995)
        
        if tp_init is not None:
            tp_vec_0 = tp_init
        else:
            tp_vec_0 = HelperSLTP()._stat_tp_step(ytr_high[:,0], ytr_close[:,-1], ytr_open[:,0], tp_min=1.005, tp_max=tp_max)
        
        m_l = np.ones(nS_tr, dtype=bool)
        m_l_low = ytr_low[:,0] > sl_vec_0
        m_l_high = ytr_high[:,0] < tp_vec_0
        m_l = m_l & m_l_low & m_l_high if include_live_mask else m_l
        
        if m_l.sum() <= 2:
            m_l = np.ones(nS_tr, dtype=bool)
            
        # intermed steps
        sl_vec = [sl_vec_0]
        tp_vec = [tp_vec_0]
        for idx in range(ytr_close.shape[1]-1):
            m_l_loop = m_l.copy() if include_live_mask else m_l
            ytr_low_step = ytr_low[m_l_loop][:,idx]
            ytr_high_step = ytr_high[m_l_loop][:,idx]
            ytr_open_step = ytr_open[m_l_loop][:,idx]
            ytr_close_last = ytr_close[m_l_loop][:,-1]
            
            sl_step = HelperSLTP()._stat_sl_step(ytr_low_step, ytr_close_last, ytr_open_step, sl_min=sl_min, sl_max=sl_max)
            sl_vec.append(sl_step)
        
            tp_step = HelperSLTP()._stat_tp_step(ytr_high_step, ytr_close_last, ytr_open_step, tp_min=tp_min, tp_max=tp_max)
            tp_vec.append(tp_step)
            
            if include_live_mask:
                m_l_loop_low  = ytr_low_step > sl_step if include_live_mask else m_l_loop
                m_l_loop_high = ytr_high_step < tp_step if include_live_mask else m_l_loop
                
                m_l[m_l_loop] = m_l_loop_low & m_l_loop_high
            
                if m_l.sum() <= 2:
                    m_l = np.ones(nS_tr, dtype=bool)
        
        # final arrays
        sl_vec = np.array(sl_vec)
        sl_tr_mat = np.repeat(sl_vec[np.newaxis, :], nS_tr, axis=0)
        sl_te_mat = np.repeat(sl_vec[np.newaxis, :], nS_te, axis=0)
        tp_vec = np.array(tp_vec)
        tp_tr_mat = np.repeat(tp_vec[np.newaxis, :], nS_tr, axis=0)
        tp_te_mat = np.repeat(tp_vec[np.newaxis, :], nS_te, axis=0)
        
        return sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat
    
    @staticmethod
    def replicate(
        sl_vec: np.ndarray,
        tp_vec: np.ndarray,
        Xtr_tree: np.ndarray,
        Xte_tree: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        assert np.ndim(sl_vec) == 1, "sl_vec must be 1-dimensional"
        assert np.ndim(tp_vec) == 1, "tp_vec must be 1-dimensional"
        
        sl_vec = np.array(sl_vec)
        tp_vec = np.array(tp_vec)
        
        sl_tr_mat = np.repeat(sl_vec[np.newaxis, :], Xtr_tree.shape[0], axis=0)
        sl_te_mat = np.repeat(sl_vec[np.newaxis, :], Xte_tree.shape[0], axis=0)
        
        tp_tr_mat = np.repeat(tp_vec[np.newaxis, :], Xtr_tree.shape[0], axis=0)
        tp_te_mat = np.repeat(tp_vec[np.newaxis, :], Xte_tree.shape[0], axis=0)
        
        return sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat

    @staticmethod
    def sltp_clustered(
        ytr_close: np.ndarray,
        ytr_low: np.ndarray,
        ytr_high: np.ndarray,
        ytr_open: np.ndarray,
        Xtr_feats: np.ndarray,
        Xte_feats: np.ndarray,
        n_clusters: int = 5,
        *,
        sl_min: float = 0.7,
        sl_max: float = 1.5,
        tp_min: float = 0.7,
        tp_max: float = 2.0,
        sl_init: float | None = None,
        tp_init: float | None = None,
        include_live_mask: bool = True,
    ) -> tuple[np.ndarray, ...]:
        # --- case n_clusters == 0: no clustering, single global SL/TP ---
        if n_clusters <= 0:
            sl_tr, sl_te, tp_tr, tp_te = HelperSLTP.calc_conditional(
                ytr_close = ytr_close,
                ytr_low   = ytr_low,
                ytr_high  = ytr_high,
                ytr_open  = ytr_open,
                nS_te     = Xte_feats.shape[0],
                sl_min=sl_min, sl_max=sl_max, tp_min=tp_min, tp_max=tp_max,
                sl_init=sl_init, tp_init=tp_init,
                include_live_mask=include_live_mask,
            )
            return sl_tr, sl_te, tp_tr, tp_te
        
        # ---- helper functions ----
        def _zscore(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
            sig = np.where(sig == 0.0, 1.0, sig)
            return (X - mu) / sig
        
        def _assign_by_edges(score: np.ndarray, edges: np.ndarray) -> np.ndarray:
            # 0..n_clusters-1
            return np.digitize(score, edges[1:-1], right=True)
        
        def _quantile_edges(score: np.ndarray, n_clusters: int) -> np.ndarray:
            q = np.linspace(0.0, 1.0, n_clusters + 1)
            edges = np.quantile(score, q)
            # enforce strictly increasing edges to keep digitize well-defined
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    eps = 1e-12 * (abs(edges[i - 1]) + 1.0)
                    edges[i] = edges[i - 1] + eps
            return edges
        
        # ---- cluster masks (train edges, apply to test) ----
        mu = Xtr_feats.mean(axis=0)
        sig = Xtr_feats.std(axis=0, ddof=0)

        ztr = _zscore(Xtr_feats, mu, sig)
        score_tr = ztr.mean(axis=1)

        edges = _quantile_edges(score_tr, n_clusters)
        cid_tr = _assign_by_edges(score_tr, edges)

        zte = _zscore(Xte_feats, mu, sig)
        score_te = zte.mean(axis=1)
        cid_te = _assign_by_edges(score_te, edges)

        masks_tr = [(cid_tr == k) for k in range(n_clusters)]
        masks_te = [(cid_te == k) for k in range(n_clusters)]

        # ---- per-cluster calc_conditional then scatter back ----
        n_tr = ytr_close.shape[0]
        n_te = Xte_feats.shape[0]
        n_y  = ytr_close.shape[1]

        sl_tr = np.full((n_tr, n_y), sl_min, dtype=float)
        tp_tr = np.full((n_tr, n_y), tp_max, dtype=float)
        sl_te = np.full((n_te, n_y), sl_min, dtype=float)
        tp_te = np.full((n_te, n_y), tp_max, dtype=float)

        for k in range(n_clusters):
            mtr, mte = masks_tr[k], masks_te[k]
            if not mtr.any() and not mte.any():
                continue

            _sl_tr, _sl_te, _tp_tr, _tp_te = HelperSLTP.calc_conditional(
                ytr_close = ytr_close[mtr],
                ytr_low   = ytr_low[mtr],
                ytr_high  = ytr_high[mtr],
                ytr_open  = ytr_open[mtr],
                nS_te     = np.sum(mte),
                sl_min=sl_min, sl_max=sl_max, tp_min=tp_min, tp_max=tp_max,
                sl_init=sl_init, tp_init=tp_init,
                include_live_mask=include_live_mask,
            )

            if mtr.any():
                sl_tr[mtr] = _sl_tr
                tp_tr[mtr] = _tp_tr
            if mte.any():
                sl_te[mte] = _sl_te
                tp_te[mte] = _tp_te

        return sl_tr, sl_te, tp_tr, tp_te