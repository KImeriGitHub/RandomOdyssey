import re
import numpy as np
import polars as pl
import datetime
from typing import Optional
import torch
from tqdm import tqdm
import random
from src.mathTools.DistributionTools import DistributionTools

import logging
logger = logging.getLogger(__name__)

class FilterSamples:
    treeblacklist_keywords = ["Category", "Seasonal", "lag"]  # "FeatureGroup", "Lag"]
    
    default_params= {
        "FilterSamples_days_to_train_end": 365,

        "FilterSamples_q_up": 0.90,

        "FilterSamples_lincomb_lr": 0.018601,
        "FilterSamples_lincomb_epochs": 150,
        "FilterSamples_lincomb_probs_noise_std": 0.221060,
        "FilterSamples_lincomb_subsample_ratio": 0.335864,
        "FilterSamples_lincomb_sharpness": 1.528495,
        "FilterSamples_lincomb_featureratio": 0.5,
        "FilterSamples_lincomb_itermax": 6,
        "FilterSamples_lincomb_show_progress": True,
        "FilterSamples_lincomb_init_toprand": 1,
        "FilterSamples_lincomb_batch_size": 2**12,

        "FilterSamples_cat_over20": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_taylor_horizon_days": 20,
        "FilterSamples_taylor_roll_window_days": 20,
        "FilterSamples_taylor_weight_slope": 0.2
    }
    
    def __init__(self, 
            Xtree_train: np.ndarray, 
            ytree_train: np.ndarray, 
            treenames: list[str],
            Xtree_test: np.ndarray, 
            meta_train: pl.DataFrame,
            meta_test: pl.DataFrame | None = None,
            ytree_test: np.ndarray | None = None,
            params: dict | None = None
        ):
        self.Xtree_train = Xtree_train
        self.ytree_train = ytree_train
        self.Xtree_test = Xtree_test
        self.ytree_test = ytree_test
        self.treenames = treenames
        self.samples_dates_train = meta_train['date']
        self.samples_dates_test = meta_test['date'] if meta_test is not None else None
        self.closeprices_train = meta_train['Close'].to_numpy()
        self.closeprices_test = meta_test['Close'].to_numpy()
        self.adjcloseprices_train = meta_train['AdjClose'].to_numpy()
        self.adjcloseprices_test = meta_test['AdjClose'].to_numpy()

        self.doTest = True
        if self.ytree_test is None or np.any(np.isnan(self.ytree_test)):
            self.doTest = False
        if self.doTest and self.samples_dates_test is None:
            # If it is None we assume a single date for all test samples
            self.samples_dates_test = pl.Series(
                name="sample_dates_test",
                values=[datetime.date(year=2100, month=1, day=1)] * self.Xtree_test.shape[0],
                dtype=pl.Date
            )
        else:
            self.samples_dates_test = self.samples_dates_test
            
        self.params = {**self.default_params, **(params or {})}
        
        self.__simple_tests()

    def __simple_tests(self) -> None:
        """
        Perform simple tests on the training and test datasets.
        """
        if self.Xtree_train.shape[1] != len(self.treenames):
            logger.error("Number of features in training data does not match the number of tree names.")
        
        if len(self.samples_dates_train) != self.Xtree_train.shape[0]:
            logger.error("Number of sample dates does not match the number of training samples.")
        if not self.samples_dates_train.is_sorted():
            logger.error("Sample dates are not sorted. Please sort them before proceeding.")
        if not all(isinstance(date, datetime.date) for date in self.samples_dates_train):
            logger.error("Sample dates should be pandas Timestamp objects.")
        if self.doTest and not all(isinstance(date, datetime.date) for date in self.samples_dates_test):
            logger.error("Sample dates for test set should be pandas Timestamp objects.")
        if self.closeprices_train is not None and self.closeprices_train.shape[0] != self.Xtree_train.shape[0]:
            logger.error("Close prices do not match the training samples.")
        if self.closeprices_test is not None and self.closeprices_test.shape[0] != self.Xtree_test.shape[0]:
            logger.error("Close prices do not match the test samples.")
        if self.adjcloseprices_train is not None and self.adjcloseprices_train.shape[0] != self.Xtree_train.shape[0]:
            logger.error("Adjusted close prices do not match the training samples.")
        if self.adjcloseprices_test is not None and self.adjcloseprices_test.shape[0] != self.Xtree_test.shape[0]:
            logger.error("Adjusted close prices do not match the test samples.")

    def separate_treefeatures(self) -> list[bool]:
        """
        Separate features and target variable.
        """
        return np.array([
            (True if not any(sub in s for sub in self.treeblacklist_keywords) else False) 
                for s in self.treenames
        ])
    
    def categorical_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create masks for categorical features in the training and test sets.
        """
        mask_train = np.ones(self.Xtree_train.shape[0], dtype=bool)
        mask_test = np.ones(self.Xtree_test.shape[0], dtype=bool)

        # Apply filter according to something like "FilterSamples_cat_over17.5"
        for k, v in self.params.items():
            m = re.fullmatch(r'FilterSamples_cat_over(\d+(?:\.\d+)?)', k)
            if v and m:
                th = float(m.group(1))
                mask_train &= (self.closeprices_train > th)
                mask_test &= (self.closeprices_test > th)
                break  # drop this if you want to apply multiple thresholds

        if self.params.get("FilterSamples_cat_posOneYearReturn", False):
            mask_train &= (
                (pl.Series(self.adjcloseprices_train) / pl.Series(self.adjcloseprices_train).shift(255)) > 1
            ).fill_null(False).to_numpy()
            mask_test &= (
                (pl.Series(self.adjcloseprices_test) / pl.Series(self.adjcloseprices_test).shift(255)) > 1
            ).fill_null(False).to_numpy()

        if self.params.get("FilterSamples_cat_posFiveYearReturn", False):
            mask_train &= (
                (pl.Series(self.adjcloseprices_train) / pl.Series(self.adjcloseprices_train).shift(5*255)) > 1
            ).fill_null(False).to_numpy()
            mask_test &= (
                (pl.Series(self.adjcloseprices_test) / pl.Series(self.adjcloseprices_test).shift(5*255)) > 1
            ).fill_null(False).to_numpy()

        return mask_train, mask_test

    def lincomb_masks(self) -> tuple[np.ndarray, np.ndarray]:
        # Starting Analysis
        score_true_train = self.evaluate_mask(np.ones(self.Xtree_train.shape[0], dtype=bool),
            dates=self.samples_dates_train, 
            y=self.ytree_train
        )
        logger.info(f"FilterSamples: (train) mean of y values {score_true_train}.")
        if self.doTest:
            score_true_test = self.evaluate_mask(np.ones(self.Xtree_test.shape[0], dtype=bool),
                dates=self.samples_dates_test, 
                y=self.ytree_test
            )
            logger.info(f"FilterSamples: (test) mean of y values {score_true_test}.")
        
        # Lin comb mask
        logger.debug(f"FilterSamples: Starting Lincomb")
        lincomb_mask_train, lincomb_mask_test = self.__mask_lincomb_featurereduce(
            A_lincomb = self.Xtree_train,
            y_lincomb = self.ytree_train,
            Atest_lincomb = self.Xtree_test,
            dates_train = self.samples_dates_train,
            dates_test = self.samples_dates_test if self.doTest else None,
            ytest_lincomb = self.ytree_test if self.doTest else None
        )
        
        # Ending Analysis
        score_train = self.evaluate_mask(
            mask=lincomb_mask_train, 
            dates=self.samples_dates_train, 
            y=self.ytree_train
        )
        logger.info(f"FilterSamples: Final score (train): {score_train:.4f}")
        
        if self.doTest:
            score_test = self.evaluate_mask(
                mask=lincomb_mask_test, 
                dates=self.samples_dates_test, 
                y=self.ytree_test
            )
            logger.info(f"FilterSamples: Final score (test): {score_test:.4f}")
        
        return lincomb_mask_train, lincomb_mask_test if self.doTest else None
    
    def get_recent_training_mask(self, dates_train: pl.Series) -> np.ndarray:
        """
        Generates a boolean mask indicating which training samples fall within a recent date range.
        The range is determined by the most recent date in `dates_train` and a configurable number of days
        (`FilterSamples_days_to_train_end`) prior to that date. If the parameter is not set or is non-positive,
        the range includes all available dates.
        """
        unique_dates = dates_train.unique().sort()
        last_Date: datetime.date = unique_dates[-1]
        days_to_train_end = self.params.get("FilterSamples_days_to_train_end", -1)
        days_to_train_end_mod = days_to_train_end if days_to_train_end > 0 else len(unique_dates) - 1
        first_day = last_Date - datetime.timedelta(days = days_to_train_end_mod)
        return (
            (self.samples_dates_train >= first_day) 
            & (self.samples_dates_train <= last_Date)
        ).fill_null(False).to_numpy()
        
    def __mask_lincomb_featurereduce(self, 
            A_lincomb: np.ndarray, 
            y_lincomb: np.ndarray,
            Atest_lincomb: np.ndarray, 
            dates_train: pl.Series,
            dates_test: Optional[pl.Series] = None,
            ytest_lincomb: Optional[np.ndarray] = None
        ) -> tuple[np.ndarray, np.ndarray]:
        itermax = self.params["FilterSamples_lincomb_itermax"]
        feature_quot = self.params["FilterSamples_lincomb_featureratio"]**(1/itermax)
        lincomb_fmask = np.array(self.separate_treefeatures())
        q_lincomb = self.params["FilterSamples_q_up"]

        mask_dates_reduced = self.get_recent_training_mask(dates_train)

        # Main Loop
        for i in range(itermax):
            logger.debug(f"  FilterSamples: Lincomb Iteration {i}/{itermax} running.")
            dates_recent = dates_train.filter(pl.Series(mask_dates_reduced))
            lincomb = self.ml_train_lincomb(
                A=A_lincomb[mask_dates_reduced][:, lincomb_fmask],
                v=y_lincomb[mask_dates_reduced],
                dates=dates_recent,
                maximize=True,
                upper_quantile=True,
            )
            w_recent = A_lincomb[mask_dates_reduced][:, lincomb_fmask] @ lincomb
            quant_w_recent = np.quantile(w_recent, q_lincomb)
            mask_train_recent = w_recent > quant_w_recent
            score_recent = self.evaluate_mask(
                mask=mask_train_recent, 
                dates=dates_recent, 
                y=y_lincomb[mask_dates_reduced]
            )
            
            lincomb_mask_train = (A_lincomb[:, lincomb_fmask] @ lincomb) > quant_w_recent
            
            #Analysis
            logger.info(f"  Mean target (train): {score_recent}")
            logger.debug(f"  Number of features selected: {np.sum(lincomb_fmask)}")
            ddiff = np.diff(np.sort(np.where(mask_train_recent)[0]))
            logger.info(f"  Max distance between idces: {max(ddiff)}")
            logger.debug(f"  Mean distance between idces: {np.mean(ddiff)}")
            logger.debug(f"  Median distance between idces: {np.median(ddiff)}")
            
            # Update test mask
            w_test = Atest_lincomb[:, lincomb_fmask] @ lincomb
            lincomb_mask_test = w_test > np.quantile(w_recent, q_lincomb)
            if self.doTest:
                score_test = self.evaluate_mask(
                    mask=lincomb_mask_test, 
                    dates=dates_test, 
                    y=ytest_lincomb
                )
                logger.info(f"  Mean target (test): {score_test}")
                
                # Print number of days which are not covered in percentage
                dates_Mat = self.establish_datesMat(dates_test, device="cpu").to_dense().numpy()
                zeroday_perc = (dates_Mat[:, lincomb_mask_test].sum(axis=1) < 0.5).sum() / dates_Mat.shape[0]
                logger.info(f"  Fraction of days with no coverage by test mask: {np.mean(zeroday_perc):.2%}")
                
            #Analysis
            logger.debug(f"  w quantile     : {np.quantile(w_recent, q_lincomb)}")
            logger.debug(f"  w_test quantile: {np.quantile(w_test, q_lincomb)}")
            
            lincomb_argsort = np.argsort(np.abs(lincomb))
            
            # Print top features
            top_features = np.array(self.treenames)[lincomb_fmask][lincomb_argsort][::-1]
            for j, feat in enumerate(top_features[:10]):
                logger.debug(f"    Top-{j} feature: {feat}")
            
            # Update lincomb mask
            if i < (itermax - 1):
                lincomb_fmask_loop = lincomb_argsort >= int((len(lincomb_argsort)-1) * (1-feature_quot))
                lincomb_fmask[lincomb_fmask] = lincomb_fmask_loop
            
        return lincomb_mask_train, lincomb_mask_test
    
    def evaluate_mask(self, mask: np.ndarray, dates: pl.Series, y: np.ndarray) -> float:
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                return np.nan  # or 0.0

            M = self.establish_datesMat(dates, device="cpu").to_dense().numpy()  # (D, N)
            M = M[:, mask]                               # (D, K)
            logy = np.log(np.clip(y[mask].astype(float), 1e-12, None))  # safe log

            rowsum = M.sum(axis=1)                       # counts per date
            valid = rowsum > 0
            if not valid.any():
                return 1.0

            perdate_logmean = (M[valid] @ logy) / rowsum[valid]   # log GM per date

            # OPTION A: equal weight per date (matches your intent)
            return float(np.exp(perdate_logmean.mean()))
    
    def establish_datesMat(self, dates: pl.Series, device: str) -> torch.Tensor:
        """
        Establish a sparse matrix of dates for use in the training process.
        """
        date_counts_np = dates.value_counts(sort=False, name="count").select("count").to_numpy().squeeze()
        date_counts = torch.tensor(date_counts_np, dtype=torch.long, device=device)  # (N,)
        n_udates = date_counts_np.size  # number of unique dates
        N = int(date_counts.sum().item()) # total number of samples
        
        dates_offsets = torch.cat([
            torch.tensor([0], device=device),
            date_counts.cumsum(0)[:-1]
        ])
        row_idx = torch.repeat_interleave(
            torch.arange(n_udates, device=device),
            date_counts
        )
        col_idx = torch.cat([
            torch.arange(s, s + length, device=device) 
            for s, length in zip(dates_offsets, date_counts)
        ])
        indices = torch.stack([row_idx, col_idx], dim=0)   # shape [2, total_ones]
        values  = torch.ones(indices.size(1), device=device)

        datesMat_sparse = torch.sparse_coo_tensor(
            indices,                       # row_idx (0…D-1), col_idx (0…N-1)
            values,
            (n_udates, N)
        ).coalesce()
        
        return datesMat_sparse
    
    def ml_init_lincomb(self,
            A_tensor: torch.Tensor,
            v_tensor: torch.Tensor,
            q: float,
            datesMat_sparse: torch.Tensor,
            maximize: bool = True,
            upper_quantile: bool = True,
            device: Optional[str] = None
        ) -> tuple[float, int]:
        D = A_tensor.shape[1]

        upper = bool(upper_quantile)
        score_t = torch.empty(D, device=device, dtype=torch.float64)

        v_log = torch.log(v_tensor.clamp_min(1e-8))  # safe log

        for i in range(D):
            w = A_tensor[:, i]  # (N,)

            with torch.no_grad():
                thr  = torch.quantile(w, q)
                keep = (w > thr) if upper else (w < thr)     # bool mask
                keep = keep.to(v_log.dtype)                  # [N]

            # counts and log-sums per date, only from kept items
            sum_p  = datesMat_sparse.matmul(keep)                 # (D,)
            sum_pv = datesMat_sparse.matmul(keep * v_log)         # (D,)

            valid = sum_p > 0
            if valid.any():
                perdate_log_gm = sum_pv[valid] / sum_p[valid]     # log GM per date
                m_all = perdate_log_gm.mean().exp()
                score_t[i] = m_all.to(dtype=torch.float64)
            else:
                score_t[i] = float(1.0)

        score = score_t.cpu().numpy()

        # Select from the top toprand scores by randomly picking one
        toprand = self.params.get("FilterSamples_lincomb_init_toprand", 1)
        topidx = random.randint(1, toprand) if toprand > 0 else 0
        
        # Sort scores and get the best one
        argsort_score = np.argsort(score)
        if maximize:
            best_idx   = int(argsort_score[-topidx])
            best_score = float(score[argsort_score[-topidx]])
        else:
            best_idx   = int(argsort_score[topidx-1])
            best_score = float(score[argsort_score[topidx-1]])
            
        return best_score, best_idx

    def ml_train_lincomb(self,
            A: np.ndarray[np.float32],
            v: np.ndarray[np.float32],
            dates: pl.Series,
            maximize: bool = True,
            upper_quantile: bool = True,
            device: Optional[str] = None,
        ) -> np.ndarray[np.float32]:
        
        q=self.params["FilterSamples_q_up"]
        lr=self.params["FilterSamples_lincomb_lr"]
        epochs=self.params["FilterSamples_lincomb_epochs"]
        probs_noise_std=self.params["FilterSamples_lincomb_probs_noise_std"]
        subsample_ratio=self.params["FilterSamples_lincomb_subsample_ratio"]
        sharpness=self.params["FilterSamples_lincomb_sharpness"]
        show_progress= self.params.get("FilterSamples_lincomb_show_progress", True)
        
        # Determine device
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        extreme_sign = -1.0 if maximize else 1.0
        upper_sign = 1.0 if upper_quantile else -1.0
        
        # Convert inputs to torch tensors
        A_tensor = torch.tensor(A, dtype=torch.float32, device=device)  # (N, D)
        v_tensor: torch.Tensor     = torch.tensor(v, dtype=torch.float32, device=device)  # (N,)
        v_log_tensor: torch.Tensor = torch.tensor(np.log(v + 1e-8), dtype=torch.float32, device=device)  # (N,)  # for the geometric mean

        datesMat_sparse = self.establish_datesMat(dates, device=device)  # (D, N)
        
        # Initialize a as the best one-hot vector
        best_score, best_idx = self.ml_init_lincomb(
            A_tensor=A_tensor,
            v_tensor=v_tensor,
            q=q,
            datesMat_sparse=datesMat_sparse,
            maximize=maximize,
            upper_quantile=upper_quantile,
            device=device
        )
        
        # initialize a as that best one-hot
        logger.info(f"FilterSamples: Best init score {best_score:.4f}")
        
        D = A_tensor.shape[1]
        a = torch.nn.Parameter(torch.zeros(D, device=device))
        with torch.no_grad():
            a[best_idx] = 1.0
        # Optimizer
        optimizer = torch.optim.Adam([a], lr=lr, weight_decay=1e-3)

        # Training loop

        # Only instantiate tqdm when showing progress
        if show_progress:
            pbar = tqdm(range(epochs), desc="Epochs")
            iterator = pbar
        else:
            iterator = range(epochs)
        
        # --- batching params ---
        batch_size = self.params.get("FilterSamples_lincomb_batch_size", 2**12)  # adjust
        N, D = A_tensor.shape
        num_dates = datesMat_sparse.shape[0]

        # map sample -> date (once)
        if "sample_to_date" not in locals() or sample_to_date is None:
            coo = datesMat_sparse.coalesce()
            cols = coo.indices()[1]    # sample ids (size nnz)
            rows = coo.indices()[0]    # date ids   (size nnz)
            sample_to_date = torch.empty(N, dtype=torch.long, device=device)
            sample_to_date[cols] = rows

        for epoch in iterator:
            optimizer.zero_grad()

            # -------- Pass 1: compute w and the global quantile (no grad) --------
            with torch.no_grad():
                w_full = torch.empty(N, device=device)
                for s in range(0, N, batch_size):
                    e = min(s + batch_size, N)
                    w_full[s:e] = A_tensor[s:e].matmul(a)  # (e-s,)
                thresh = torch.quantile(w_full, q)

            # -------- Pass 2: streamed loss with grad --------
            probs_perdate_raw = torch.zeros(num_dates, device=device)  # will become grad-connected
            mat_pv            = torch.zeros(num_dates, device=device)

            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)

                w_b = A_tensor[s:e].matmul(a)  # (e-s,) — grad path preserved

                logits_b = (upper_sign * sharpness * (w_b - thresh) +
                            torch.randn_like(w_b) * probs_noise_std).clamp(-20, 20)

                # subsample
                mask_b = (torch.rand_like(logits_b).lt(subsample_ratio)).float()
                probs_sub_b = torch.sigmoid(logits_b) * mask_b / subsample_ratio  # (e-s,)

                # per-date accumulation (functional, keeps grad)
                dates_b = sample_to_date[s:e]  # (e-s,)
                # index_add (out-of-place) creates grad-linked tensors
                delta_raw = torch.index_add(torch.zeros(num_dates, device=device), 0, dates_b, probs_sub_b)
                delta_pv  = torch.index_add(torch.zeros(num_dates, device=device), 0, dates_b,
                                            probs_sub_b * v_log_tensor[s:e])

                probs_perdate_raw = probs_perdate_raw + delta_raw
                mat_pv            = mat_pv            + delta_pv

            perdate_mean_all = mat_pv / (probs_perdate_raw + 1e-8)
            nonneg_mask = probs_perdate_raw > 1e-8
            mean_term = perdate_mean_all[nonneg_mask].mean()

            loss = -mean_term if extreme_sign < 0 else mean_term
            loss.backward()
            optimizer.step()

            if show_progress:
                pbar.set_postfix(mean_perdate_v=f"{mean_term.item():.4f}")
                
        # Return optimized `a` as numpy array
        return a.detach().cpu().numpy()
    
    def taylor_feature_masks(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Selects a single feature by maximizing a Taylor score computed on the
        rolling mean of the per-day target under a recent-window mask.

        Score: x0 + (t/2) * x0', where
        - x0  = rolling mean at the last training day
        - x0' = slope (per day) of the rolling mean near the end
        - t   = horizon in days (param: FilterSamples_taylor_horizon_days; defaults to window)
        """
        # --- Params & setup
        q_up = self.params.get("FilterSamples_q_up", 0.9)
        weight_slope = self.params.get("FilterSamples_taylor_weight_slope", 0.5)
        roll_w = int(self.params.get("FilterSamples_taylor_roll_window_days", 20))
        roll_w = max(2, roll_w)
        iter_feats = np.array(self.separate_treefeatures())
        feat_idx = np.where(iter_feats)[0]

        # Starting Analysis
        score_true_train = self.evaluate_mask(
            np.ones(self.Xtree_train.shape[0], dtype=bool),
            dates=self.samples_dates_train, y=self.ytree_train
        )
        logger.info(f"FilterSamples/Taylor: (train) mean of y values {score_true_train}.")
        if self.doTest:
            score_true_test = self.evaluate_mask(
                np.ones(self.Xtree_test.shape[0], dtype=bool),
                dates=self.samples_dates_test, y=self.ytree_test
            )
            logger.info(f"FilterSamples/Taylor: (test) mean of y values {score_true_test}.")

        # Recent window over training
        mask_recent_train = self.get_recent_training_mask(self.samples_dates_train)
        dates_recent = self.samples_dates_train.filter(pl.Series(mask_recent_train))
        y_recent = self.ytree_train[mask_recent_train]
        A_recent = self.Xtree_train[mask_recent_train][:, iter_feats]

        # Day x Sample matrix (recent)
        dates_mat_recent = self.establish_datesMat(dates_recent, device="cpu").to_dense().numpy()
        n_days_recent = dates_mat_recent.shape[0]

        # Horizon t (days)
        t_h = int(self.params.get("FilterSamples_taylor_horizon_days", roll_w))
        t_h = max(1, min(t_h, n_days_recent))

        best_feat = None
        best_score = -np.inf
        best_thresh = None

        logger.debug("FilterSamples/Taylor: scoring features via rolling-mean Taylor approximation.")

        for k, f in enumerate(feat_idx):
            vals = A_recent[:, k]  # single feature over recent samples
            if not np.any(np.isfinite(vals)):
                continue

            # Threshold on recent quantile
            try:
                thresh = np.quantile(vals[np.isfinite(vals)], q_up)
            except Exception:
                continue

            sel_recent = (vals >= thresh)
            if sel_recent.sum() == 0:
                continue

            # Per-day mean target under selection
            # counts: (# selected samples per day), sums: (sum of y per day over selected)
            counts = dates_mat_recent[:, sel_recent].sum(axis=1).astype(float)
            sums = (dates_mat_recent[:, sel_recent] * y_recent[sel_recent]).sum(axis=1).astype(float)
            daily_mean = np.divide(
                sums, counts,
                out=np.full_like(counts, np.nan, dtype=float),
                where=counts > 0
            )

            # Rolling mean (smooth) on per-day means
            s = pl.Series(daily_mean)
            s = s.to_frame().select(pl.all().fill_nan(None)).to_series()
            rm = s.rolling_mean(window_size=roll_w, min_periods=min(max(2, roll_w//2), n_days_recent))\
                .fill_null(strategy="forward").fill_null(strategy="backward").to_numpy()

            if not np.isfinite(rm[-1]):
                # still too many NaNs → skip
                continue

            x0 = float(rm[-1])

            # Slope near the end via OLS on last W points of rolling mean
            W = min(roll_w, len(rm))
            y_seg = rm[-W:].astype(float)
            x_seg = np.arange(W, dtype=float)
            x_seg_c = x_seg - x_seg.mean()
            y_seg_c = y_seg - y_seg.mean()
            denom = (x_seg_c ** 2).sum()
            slope = float((x_seg_c @ y_seg_c) / (denom + 1e-12))  # per-day slope

            score = x0 + weight_slope * t_h * slope

            logger.debug(
                f"  Taylor score feat[{f}] {self.treenames[f]}: x0={x0:.6f}, "
                f"slope={slope:.6f}, t={t_h}, score={score:.6f}"
            )

            if score > best_score:
                best_score = score
                best_feat = f
                best_thresh = thresh

        if best_feat is None:
            logger.warning("FilterSamples/Taylor: No valid feature found. Falling back to selecting all.")
            mask_train = np.ones(self.Xtree_train.shape[0], dtype=bool)
            mask_test = np.ones(self.Xtree_test.shape[0], dtype=bool) if self.doTest else None
            return mask_train, mask_test

        logger.info(
            f"FilterSamples/Taylor: Selected feature '{self.treenames[best_feat]}' "
            f"with score {best_score:.6f} and threshold {best_thresh:.6g} (q={q_up})."
        )

        # Final masks (use training recent-derived threshold on full sets)
        vals_tr = self.Xtree_train[:, best_feat]
        thr_tr  = np.nanquantile(vals_tr, q_up, method="lower")
        mask_train = np.isfinite(vals_tr) & (vals_tr >= thr_tr)

        vals_te = self.Xtree_test[:, best_feat]
        thr_te  = np.nanquantile(vals_te, q_up, method="lower")
        mask_test = np.isfinite(vals_te) & (vals_te >= thr_te)

        # Ending Analysis
        score_train = self.evaluate_mask(mask=mask_train, dates=self.samples_dates_train, y=self.ytree_train)
        logger.info(f"FilterSamples/Taylor: Final score (train): {score_train:.4f}")

        if self.doTest:
            score_test = self.evaluate_mask(mask=mask_test, dates=self.samples_dates_test, y=self.ytree_test)
            logger.info(f"FilterSamples/Taylor: Final score (test): {score_test:.4f}")

            # Coverage on test
            dates_mat_test = self.establish_datesMat(self.samples_dates_test, device="cpu").to_dense().numpy()
            zeroday_perc = (dates_mat_test[:, mask_test].sum(axis=1) < 0.5).sum() / dates_mat_test.shape[0]
            logger.info(f"FilterSamples/Taylor: Fraction of days with no coverage by test mask: {np.mean(zeroday_perc):.2%}")

        return mask_train, mask_test