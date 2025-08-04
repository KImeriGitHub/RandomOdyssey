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

        "FilterSamples_lincomb_q_up": 0.90,
        "FilterSamples_lincomb_lr": 0.018601,
        "FilterSamples_lincomb_epochs": 15000,
        "FilterSamples_lincomb_probs_noise_std": 0.221060,
        "FilterSamples_lincomb_subsample_ratio": 0.335864,
        "FilterSamples_lincomb_sharpness": 1.528495,
        "FilterSamples_lincomb_featureratio": 0.5,
        "FilterSamples_lincomb_itermax": 6,
        "FilterSamples_lincomb_show_progress": True,
        "FilterSamples_lincomb_init_toprand": 1
    }
    
    def __init__(self, 
            Xtree_train: np.ndarray, 
            ytree_train: np.ndarray, 
            treenames: list[str],
            Xtree_test: np.ndarray, 
            samples_dates_train: pl.Series,
            samples_dates_test: Optional[pl.Series] = None,
            ytree_test: np.ndarray = None,
            params: Optional[dict] = None
        ):
        self.Xtree_train = Xtree_train
        self.ytree_train = ytree_train
        self.Xtree_test = Xtree_test
        self.ytree_test = ytree_test
        self.treenames = treenames
        self.samples_dates_train = samples_dates_train

        self.doTest = True
        if self.ytree_test is None:
            self.doTest = False
        if self.doTest and samples_dates_test is None:
            # If it is None we assume a single date for all test samples
            self.samples_dates_test = pl.Series(
                name="sample_dates_test",
                values=[datetime.date(year=2100, month=1, day=1)] * self.Xtree_test.shape[0],
                dtype=pl.Date
            )
        else:
            self.samples_dates_test = samples_dates_test
            
        self.params = {**self.default_params, **(params or {})}
        
        self.days_to_train_end = self.params.get("FilterSamples_days_to_train_end", -1)
        
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
    
    def separate_treefeatures(self) -> list[bool]:
        """
        Separate features and target variable.
        """
        return np.array([
            (True if not any(sub in s for sub in self.treeblacklist_keywords) else False) 
                for s in self.treenames
        ])
        
    def establish_weights(self,
            sam_tr: np.ndarray,
            sam_te: np.ndarray,
            f_idx: int = None
    ) -> np.ndarray:
        """
        Establish matching weights for training and test samples.
        """
        if f_idx is None:
            ksDistance = DistributionTools.ksDistance(
                sam_tr=sam_tr,
                sam_te=sam_te,
            )
            argsort = np.argsort(ksDistance)
            f_idx = argsort[-1]
        
        weights = DistributionTools.establishMatchingWeight(
            sam_tr = sam_tr[:, f_idx],
            sam_te = sam_te[:, f_idx],
            n_bin = 10,
            minbd = 0.8
        )
        return weights
    
    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        
        ## Lin comb mask
        logger.debug(f"FilterSamples: Starting Lincomb")
        lincomb_mask_train, lincomb_mask_test = self.mask_lincomb_featurereduce(
            A_lincomb = self.Xtree_train,
            y_lincomb = self.ytree_train,
            Atest_lincomb = self.Xtree_test,
            dates_train = self.samples_dates_train,
            dates_test = self.samples_dates_test if self.doTest else None,
            ytest_lincomb = self.ytree_test if self.doTest else None
        )
        
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
        
        return lincomb_mask_train, lincomb_mask_test, score_train, score_test if self.doTest else None
        
    def mask_lincomb_featurereduce(self, 
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
        q_lincomb = self.params["FilterSamples_lincomb_q_up"]
        
        # Reducing to recent days training data
        unique_dates = dates_train.unique().sort()
        last_Date: datetime.date = unique_dates[-1]
        days_to_train_end_mod = self.days_to_train_end if self.days_to_train_end > 0 else len(unique_dates) - 1
        first_day = last_Date - datetime.timedelta(days = days_to_train_end_mod)
        mask_dates_reduced = (
            (self.samples_dates_train >= first_day) 
            & (self.samples_dates_train <= last_Date)
        ).fill_null(False).to_numpy()
        
        # weights UNDER CONSTRUCTION
        #weights = self.establish_weights(
        #    sam_tr=A_lincomb[:, lincomb_fmask],
        #    sam_te=Atest_lincomb[:, lincomb_fmask]
        #)
        #weight_mask = weights > 0.5
        
        for i in range(itermax):
            logger.debug(f"  FilterSamples: Lincomb Iteration {i}/{itermax} running.")
            dates_recent = dates_train.filter(mask_dates_reduced)
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
        dates_Mat = self.establish_datesMat(dates, device="cpu").to_dense().numpy()
        
        dates_Mat = dates_Mat[:, mask]  # Filter the matrix by the mask
        y_mask = y[mask]  # Filter the target variable by the mask
        
        perdate_mean = ((dates_Mat @ y_mask) / (np.sum(dates_Mat, axis=1) + 1e-8))
        non_zero_entries = np.sum(perdate_mean > 1e-7)
        
        return np.sum(perdate_mean) / (non_zero_entries + 1e-8)
        
    
    def establish_datesMat(self, dates: pl.Series, device: str) -> torch.Tensor:
        
        """
        Establish a sparse matrix of dates for use in the training process.
        This method is not used in the current implementation but is kept for reference.
        """
        date_counts_np = dates.value_counts(sort=False, name="count").select("count").to_numpy().squeeze()
        date_counts = torch.tensor(date_counts_np, dtype=torch.long, device=device)  # (N,)
        n_udates = date_counts.size(0)  # number of unique dates
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

        # Evaluate each basis vector
        upper_sign = 1.0 if upper_quantile else -1.0
        score = [None] * D
        for i in range(D):
            a_basis = torch.zeros(D, device=device)
            a_basis[i] = 1.0

            with torch.no_grad():
                w      = A_tensor.matmul(a_basis)           # (N,)
                thresh = torch.quantile(w, q)               # scalar
                # if upper:  mask = w>thresh;  else: mask = w<thresh
                mask   = torch.where(upper_sign * w > upper_sign * thresh, 1.0, 0.0)

                sum_p     = datesMat_sparse.matmul(mask)            # (D,)
                sum_pv    = datesMat_sparse.matmul(mask * v_tensor) # (D,)
                perdate_m = sum_pv / (sum_p + 1e-7)                 # (D,)
                score[i]  = perdate_m.mean().item()

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
        
        q=self.params["FilterSamples_lincomb_q_up"]
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
        v_tensor: torch.Tensor = torch.tensor(v, dtype=torch.float32, device=device)  # (N,)

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
        a = torch.nn.Parameter(torch.ones(D, dtype=torch.float32, device=device)*0.1)
        a.data[best_idx] = 1.0
        # Optimizer
        optimizer = torch.optim.Adam([a], lr=lr)

        # Training loop
        subsample_size = max(1, int(A.shape[0] * subsample_ratio))
        # Only instantiate tqdm when showing progress
        if show_progress:
            pbar = tqdm(range(epochs), desc="Epochs")
            iterator = pbar
        else:
            iterator = range(epochs)
        
        for epoch in iterator:
            optimizer.zero_grad()
            w      = A_tensor.matmul(a)                             # (N,)
            thresh = torch.quantile(w, q)                           # scalar
            probs  = torch.sigmoid(upper_sign * sharpness * (w - thresh))
            
            # --- regularisation: noise + subsample ---
            probs = (probs + torch.randn_like(probs) * probs_noise_std).clamp(0,1)
            
            # subsample:
            mask = torch.zeros(probs.size(0), dtype=torch.bool, device=device)
            idx  = torch.randperm(probs.size(0), device=device)[:subsample_size]  # is slow for large num of parameters
            mask[idx] = True
            probs_sub   = probs * mask                                # (N,)
            v_sub       = v_tensor * mask          # (N,)
            
            # per-date sums & means on the subsample
            probs_perdate = datesMat_sparse.matmul(probs_sub)             # (D,)
            perdate_mean  = (
                datesMat_sparse.matmul(probs_sub * v_sub) 
                / (probs_perdate + 1e-8)
            )  # (D,)
            nonzero_sum  = (probs_perdate > 0).sum().item()
            
            # Loss calculation
            loss = extreme_sign * perdate_mean.sum() / (nonzero_sum + 1e-8)
            loss.backward()
            optimizer.step()
            
            if show_progress:
                mean_all_v = ((probs * v_tensor).sum() / (probs.sum() + 1e-8)).item()
                perdate_mean_all = datesMat_sparse.matmul(probs * v_tensor) / (datesMat_sparse.matmul(probs) + 1e-8)
                nonzero_sum_all = (datesMat_sparse.matmul(probs) > 0).sum().item()
                mean_perdate_v = perdate_mean_all.sum().item() / nonzero_sum_all
                pbar.set_postfix(
                    mean_v=f"{mean_all_v:.4f}",
                    mean_perdate_v=f"{mean_perdate_v:.4f}"
                )

        # Return optimized `a` as numpy array
        return a.detach().cpu().numpy()