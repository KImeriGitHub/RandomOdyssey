import numpy as np

class HelperSLTP:
    def __init__(self):
        pass
    
    def _calc_sl_zero(
        self,
        ytr_low,
        ytr_last, 
        sl_min=0.8,
        sl_max=0.995
    ) -> float:
        """
        ytr_low is mapped to ytr_last. Then sort ytr_low =: x.
        Through the mapping we get z := Mapped(ytr_last).

        Given arrays x,z (same length), we:
            1. Sort x ascending and apply the same permutation to z.
            2. For each index idx, compute:
                score(idx) = x[idx] * (idx+1) - sum(z[:idx+1])
            3. Find idx that maximizes score(idx).

        The unconstrained optimal stop loss level would be:
            sl_raw = x[idx]

        The returned stop loss is capped above by sl_max:
            sl = min(sl_raw, sl_max)

        Returns
        -------
        float
            The capped optimal stop loss level sl.
        """
        ytr_low = np.asarray(ytr_low)
        ytr_last = np.asarray(ytr_last)
        x = ytr_low.astype(float)
        z = ytr_last.astype(float)
        
        n = len(x)
        if n <= 1: return sl_min

        # sort by x ascending
        asort = np.argsort(x)
        x = x[asort]
        z = z[asort]

        # cumulative sums of z
        z_sum = np.cumsum(z)

        n_vec = np.arange(1, n + 1)

        # score(idx) = x[idx] * (idx+1) - sum(z[:idx+1])
        scores = x * n_vec - z_sum

        best_idx = int(np.argmax(scores))
        
        sl_end = float(np.clip(x[best_idx], sl_min, sl_max))
        return sl_end

    def _calc_tp_zero(
        self,
        ytr_high,
        ytr_last,
        tp_min=1.005,
        tp_max=2.0
    ) -> float:
        """
        ytr_high is mapped to ytr_last. Then sort ytr_high =: x.
        Through the mapping we get z := Mapped(ytr_last).

        Given arrays x,z (same length), we:
            1. Sort x ascending and apply the same permutation to z.
            2. For each index idx (0-based), compute:
                score(idx) = x[idx] * (n-idx) - sum(z[idx:])
            3. Find idx that maximizes score(idx).

        The unconstrained optimal take profit level would be:
            tp_raw = x[idx]

        The returned take profit is floored below by tp_min:
            tp = max(tp_raw, tp_min)
        """
        ytr_high = np.asarray(ytr_high)
        ytr_last = np.asarray(ytr_last)

        x = ytr_high.astype(float)
        z = ytr_last.astype(float)
        
        n = len(x)
        if n <= 1: return tp_max

        # sort by x ascending
        asort = np.argsort(x)
        x = x[asort]
        z = z[asort]

        # z_sum[i] = sum(z[i:])
        z_sum = np.cumsum(z[::-1])[::-1]

        # n_vec[idx] = n - idx
        n_vec = np.arange(1, n + 1)[::-1]

        # score(idx) = x[idx] * (n-idx) - sum(z[idx:])
        scores = x * n_vec - z_sum

        best_idx = int(np.argmax(scores))
        
        tp_end = float(np.clip(x[best_idx], tp_min, tp_max))
        return tp_end
    
    def _calc_sl_step(
        self,
        ytr_step,
        ytr_last,
        sl_min=0.7,
        sl_max=1.5
    ) -> float:
        """
        ytr_step is mapped to ytr_last. Then sort ytr_step =: x.
        Through the mapping we get z := Mapped(ytr_last).

        Given arrays x,z (same length), we:
            1. Sort x ascending and apply the same permutation to z.
            2. For each index idx, compute:
                score(idx) = x[idx] * (idx+1) - sum(z[:idx+1])
            3. Find idx that maximizes score(idx).

        The unconstrained optimal stop loss level would be:
            sl_raw = x[idx]

        The returned stop loss is capped above by sl_max:
            sl = min(sl_raw, sl_max)

        Returns
        -------
        float
            The capped optimal stop loss level sl.
        """
        ytr_step = np.asarray(ytr_step)
        ytr_last = np.asarray(ytr_last)
        x = ytr_step.astype(float)
        z = ytr_last.astype(float)
        
        n = len(x)
        if n <= 1: return sl_min

        # sort by x ascending
        asort = np.argsort(x)
        x = x[asort]
        z = z[asort]

        # cumulative sums of z
        z_sum = np.cumsum(z)

        n_vec = np.arange(1, n + 1)

        # score(idx) = x[idx] * (idx+1) - sum(z[:idx+1])
        scores = x * n_vec - z_sum

        best_idx = int(np.argmax(scores))
        
        sl_end = float(np.clip(x[best_idx], sl_min, sl_max))
        return sl_end
    
    def _calc_tp_step(
        self,
        ytr_step, 
        ytr_last, 
        tp_min=0.7,
        tp_max=2.0
    ) -> float:
        """
        ytr_step is mapped to ytr_last. Then sort ytr_step =: x.
        Through the mapping we get z := Mapped(ytr_last) values.

        Given arrays z and x (same length), we:
            1. Sort x ascending and apply the same permutation to z.
            2. For each index idx, compute:
                score(idx) = x[idx] * (n-idx) - sum(z[idx:])
            3. Find idx that maximizes score(idx).

        The unconstrained optimal take profit level would be:
            tp_raw = x[idx]

        The returned take profit is floored below by tp_min:
            tp = max(tp_raw, tp_min)

        Returns
        -------
        float
            The floored optimal take profit level tp.
        """
        ytr_step = np.asarray(ytr_step)
        ytr_last = np.asarray(ytr_last)
        x = ytr_step.astype(float)
        z = ytr_last.astype(float)
        
        n = len(x)
        if n <= 1: return tp_max

        # z_sum[i] = sum(z[i:])
        z_sum = np.cumsum(z[::-1])[::-1]

        # n_vec[idx] = n - idx
        n_vec = np.arange(1, n + 1)[::-1]

        # score(idx) = x[idx] * (n-idx) - sum(z[idx:])
        scores = x * n_vec - z_sum

        best_idx = int(np.argmax(scores))
        
        tp_end = float(np.clip(x[best_idx], tp_min, tp_max))
        return tp_end
      
    @staticmethod
    def calc_conditional(
        ytr_close: np.ndarray,
        ytr_low: np.ndarray,
        ytr_high: np.ndarray,
        ytr_open: np.ndarray, # keeping for consistency, not used
        Xtr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        sl_min: float = 0.7,
        sl_max: float = 1.5,
        tp_min: float = 0.7,
        tp_max: float = 2.0,
        sl_init: float | None = None,
        tp_init: float | None = None,
        include_live_mask: bool = True,
    ) -> tuple[np.ndarray, ...]:
        # Init step
        if sl_init is not None:
            sl_vec_0 = sl_init
        else:
            sl_vec_0 = HelperSLTP()._calc_sl_zero(ytr_low[:,0], ytr_close[:,-1], sl_min=sl_min, sl_max=0.995)
        
        if tp_init is not None:
            tp_vec_0 = tp_init
        else:
            tp_vec_0 = HelperSLTP()._calc_tp_zero(ytr_high[:,0], ytr_close[:,-1], tp_min=1.005, tp_max=tp_max)
        
        m_l = np.ones(Xtr_tree.shape[0], dtype=bool)
        m_l_low = ytr_low[:,0] > sl_vec_0
        m_l_high = ytr_high[:,0] < tp_vec_0
        m_l = m_l & m_l_low & m_l_high if include_live_mask else m_l
        
        if m_l.sum() <= 2:
            m_l = np.ones(Xtr_tree.shape[0], dtype=bool)
            
        # intermed steps
        sl_vec = [sl_vec_0]
        tp_vec = [tp_vec_0]
        for idx in range(ytr_close.shape[1]-1):
            m_l_loop = m_l.copy() if include_live_mask else m_l
            ytr_low_step = ytr_low[m_l_loop][:,idx]
            ytr_high_step = ytr_high[m_l_loop][:,idx]
            ytr_close_last = ytr_close[m_l_loop][:,-1]
            
            sl_step = HelperSLTP()._calc_sl_step(ytr_low_step, ytr_close_last, sl_min=sl_min, sl_max=sl_max)
            sl_vec.append(sl_step)
        
            tp_step = HelperSLTP()._calc_tp_step(ytr_high_step, ytr_close_last, tp_min=tp_min, tp_max=tp_max)
            tp_vec.append(tp_step)
            
            if include_live_mask:
                m_l_loop_low  = ytr_low_step > sl_step if include_live_mask else m_l_loop
                m_l_loop_high = ytr_high_step < tp_step if include_live_mask else m_l_loop
                
                m_l[m_l_loop] = m_l_loop_low & m_l_loop_high
            
                if m_l.sum() <= 2:
                    m_l = np.ones(Xtr_tree.shape[0], dtype=bool)
        
        # final arrays
        sl_vec = np.array(sl_vec)
        sl_tr_mat = np.repeat(sl_vec[np.newaxis, :], Xtr_tree.shape[0], axis=0)
        sl_te_mat = np.repeat(sl_vec[np.newaxis, :], Xte_tree.shape[0], axis=0)
        tp_vec = np.array(tp_vec)
        tp_tr_mat = np.repeat(tp_vec[np.newaxis, :], Xtr_tree.shape[0], axis=0)
        tp_te_mat = np.repeat(tp_vec[np.newaxis, :], Xte_tree.shape[0], axis=0)
        
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
