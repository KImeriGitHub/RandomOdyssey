import numpy as np
import optuna
import polars as pl
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import r_regression

from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperSLTP import HelperSLTP

import logging
logger = logging.getLogger(__name__)

class StratLGBMSlices(BaseStrategy):
    expected_load_params = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "subsample_ratio": 1.0,
    }

    base_params = {
    }

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        opt_params["do_transform"] = True #trial.suggest_categorical("do_transform", [True, False])
        
        opt_params["catsample_max_lag"] = trial.suggest_int("catsample_max_lag", 0, 20, step=5) # Note: try 100 at some point
        opt_params["max_features_colinsampling"] = trial.suggest_int("max_features_colinsampling", 25, 200, step=5)
        opt_params["threshold_colin_sampling"] = trial.suggest_float("threshold_colin_sampling", 0.01, 0.09, log=True)
        opt_params["inc_seasonal_in_colin"] = False #trial.suggest_categorical("inc_seasonal_in_colin", [True, False])
        opt_params["n_slices"] = trial.suggest_int("n_slices", 1, 2)
        opt_params["slice_quantile"] = trial.suggest_float("slice_quantile", 0.94, 0.965)
        opt_params["val_split"] = 0.05 #trial.suggest_float("val_split", 0.01, 0.2)
        
        opt_params["LGB_num_boost_round"]           = trial.suggest_int("LGB_num_boost_round", 100, 200, step=10)
        opt_params["LGB_lambda_l1"]                 = 0.005 #trial.suggest_float("LGB_lambda_l1", 0.00001, 0.1, log=True)
        opt_params["LGB_lambda_l2"]                 = 0.00004 #trial.suggest_float("LGB_lambda_l2", 0.00001, 0.1, log=True)
        opt_params["LGB_feature_fraction"]          = 1.0 #trial.suggest_float("LGB_feature_fraction", 0.8, 1.0)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 300, 1200, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 28, 50, step=1)
        opt_params["LGB_learning_rate"]             = trial.suggest_float("LGB_learning_rate", 0.01, 0.1, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 20, 150, log=True)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 0.000001, 0.0005, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 0.01, 0.9, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = 0.2 #trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["LGB_max_bin"]                   = 250 #trial.suggest_int("LGB_max_bin", 100, 800, step=25)
        opt_params["LGB_early_stopping_rounds"]     = opt_params["LGB_num_boost_round"]//10
        
        params = dict(self.base_params)
        params.update(opt_params)
        return params

    def run(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        ytr_tree_low,
        ytr_tree_high,
        ytr_tree_open,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
        opt_params: dict,
    ) -> tuple[np.ndarray, ...]:
        do_transform = opt_params.get("do_transform", False)
        catsample_max_lag = opt_params.get("catsample_max_lag", 1)
        row_ratio = opt_params.get("row_subsample_ratio", 1.0)
        m_features = opt_params.get("max_features_colinsampling", 20)
        thr_colin = opt_params.get("threshold_colin_sampling", 0.2)
        inc_seasonal_in_colin = opt_params.get("inc_seasonal_in_colin", False)
        n_slices = opt_params.get("n_slices", 5)
        qup = opt_params.get("slice_quantile", 0.3)
        val_split = opt_params.get("val_split", 0.05)
        
        mm: MachineModels = MachineModels(opt_params)
        
        if do_transform:
            scaler = StandardScaler().fit(Xtr_tree)
            Xtr_tree = scaler.transform(Xtr_tree)
            Xte_tree = scaler.transform(Xte_tree)
            
        ytr_opt = ytr_tree[:,-1]
            
        ## GET TREENAME MASKS
        tn = np.asarray(treenames, str)
        mask_tn = np.ones(tn.shape[0], dtype=bool)
        # Categorical subsampling of treenames
        mask_tn_lag, mask_tn_noncolin = self.catsample_treenames(
            treenames,
            max_lag=catsample_max_lag,
        )
        mask_tn &= mask_tn_lag | mask_tn_noncolin
        
        # Colinear sampling based on target correlation
        if inc_seasonal_in_colin:
            mask_tn_colin_input = mask_tn_lag
        else:
            mask_tn_colin_input = mask_tn_lag & ~mask_tn_noncolin
        mask_treenames_colinsampling = self.colinearity_treenames_mask(
            Xtr_tree[:, mask_tn_colin_input],
            ytr_opt,
            treenames=tn[mask_tn_colin_input],
            m_features=m_features,
            thr_crosscorr=thr_colin,
            row_ratio=row_ratio,
        )
        mask_tn[mask_tn_colin_input] = mask_treenames_colinsampling
        
        ## LGBM SLICES
        mask_train = np.zeros(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.zeros(Xte_tree.shape[0], dtype=bool)
        score_tr = np.zeros(Xtr_tree.shape[0], dtype=float)
        score_te = np.zeros(Xte_tree.shape[0], dtype=float)
        for i in range(n_slices):
            logger.info(f"Iteration {i+1}/{n_slices}")
            
            neg_mask_tr = ~mask_train
            neg_mask_te = ~mask_test
            Xtr_it, ytr_it = Xtr_tree[neg_mask_tr][:,mask_tn], ytr_opt[neg_mask_tr]
            Xte_it = Xte_tree[neg_mask_te][:,mask_tn]
            
            if neg_mask_tr.sum() <= 10 or neg_mask_te.sum() <= 10:
                logger.info("  Not enough samples left, breaking.")
                break

            m_tr_it, m_te_it, sc_tr_it, sc_te_it = self._lgbm_mask(Xtr_it, ytr_it, Xte_it, qup, val_split, mm)
            mask_train[neg_mask_tr] |= m_tr_it
            mask_test[neg_mask_te] |= m_te_it
            score_tr[neg_mask_tr] = sc_tr_it * (0.9**i)
            score_te[neg_mask_te] = sc_te_it * (0.9**i)
            
            #Logging
            sc_tr_it_q = sc_tr_it[m_tr_it] if m_tr_it.sum() > 0 else np.array([])
            sc_te_it_q = sc_te_it[m_te_it] if m_te_it.sum() > 0 else np.array([])
            sc_tr_full = score_tr[mask_train] if mask_train.sum() > 0 else np.array([])
            sc_te_full = score_te[mask_test] if mask_test.sum() > 0 else np.array([])
            logger.info(f"  Masked train kept: {mask_train.sum()/len(mask_train)} | test kept: {mask_test.sum()/len(mask_test)}")
            logger.info(f"  LGBM Scores train iter: mean {np.mean(sc_tr_it_q)} | test: mean {np.mean(sc_te_it_q)}")
            logger.info(f"  LGBM Scores train full: mean {np.mean(sc_tr_full)} | test: mean {np.mean(sc_te_full)}")

        sl_vec = [0.856918023350039, 0.907044803273807, 0.7880509119486553, 0.8653404222291542, 0.9629949865370335]
        tp_vec = [1.3869744119554812, 1.6876932380841254, 1.376056597544607, 1.5268853848238373, 1.5113575976609412]
        sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat = HelperSLTP.replicate(
            sl_vec,
            tp_vec,
            Xtr_tree,
            Xte_tree,
        )

        return mask_train, mask_test, sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat, score_tr, score_te

    def precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        ytr_tree_low,
        ytr_tree_high,
        ytr_tree_open,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
    ) -> tuple[np.ndarray, np.ndarray]:        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
        
        # rnadom row selection with subsample ratio
        subsample_ratio = self.precompute_params.get("subsample_ratio", 0.1)
        if subsample_ratio < 1.0:
            n_train_sub = max(2, int(Xtr_tree.shape[0] * subsample_ratio))
            n_test_sub = max(2, int(Xte_tree.shape[0] * subsample_ratio))
            #rng = np.random.default_rng(42)
            idx_train_sub = np.random.choice(Xtr_tree.shape[0], n_train_sub, replace=False)
            idx_test_sub = np.random.choice(Xte_tree.shape[0], n_test_sub, replace=False)
            mask_train[:] = False
            mask_train[idx_train_sub] = True
            #mask_test[:] = False
            #mask_test[idx_test_sub] = True

        sl_vec = [0.856918023350039, 0.907044803273807, 0.7880509119486553, 0.8653404222291542, 0.9629949865370335]
        tp_vec = [1.3869744119554812, 1.6876932380841254, 1.376056597544607, 1.5268853848238373, 1.5113575976609412]
        sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat = HelperSLTP.replicate(
            sl_vec,
            tp_vec,
            Xtr_tree,
            Xte_tree,
        )
        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_vec} | tp: {tp_vec}")

        return mask_train, mask_test, sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat
    
    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _lgbm_mask(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        Xte: np.ndarray, 
        q: float, 
        val_split: float = 0.05,
        mm: MachineModels = None,
    ) -> tuple[np.ndarray, ...]:
        nS = X.shape[0]
        nS_tr = int(nS * (1 - val_split))
        
        # Val split
        X_tr, y_tr = X[:nS_tr], y[:nS_tr]
        X_val, y_val = X[nS_tr:], y[nS_tr:]
        
        # Run model and predict
        model, _ = mm.run_LGB(X_tr, y_tr, X_val, y_val)
        yhat_tr = model.predict(X, num_iteration=model.best_iteration)
        yhat_te = model.predict(Xte, num_iteration=model.best_iteration)
        
        qhat_tr = np.quantile(yhat_tr, q)
        mask_top_tr = yhat_tr >= qhat_tr
        mask_top_te = yhat_te >= qhat_tr
        
        return mask_top_tr, mask_top_te, yhat_tr, yhat_te
    
    def catsample_treenames(
        self,
        treenames: np.ndarray,
        max_lag: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        tn = np.asarray(treenames, str)
        
        # max lag sampling
        mask_tn_lag = np.array(
            [(m:=re.search(r'_lag_m(\d+)', s)) is None or int(m.group(1)) <= max_lag
                for s in tn]
        )
        
        # not usable for colin
        mask_tn_non_colin = np.zeros(tn.shape[0], dtype=bool)
        mask_tn_lag |= np.char.find(tn, "Seasonal_year") >= 0
        mask_tn_lag |= np.char.find(tn, "Category_other") >= 0
        return mask_tn_lag, mask_tn_non_colin
    
    def _subsample_rows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        row_ratio: float,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optionally subsample rows of (X, y)."""
        if not (0 < row_ratio <= 1.0):
            raise ValueError("row_ratio must be in (0, 1].")

        if row_ratio == 1.0:
            return X, y

        n = X.shape[0]
        k = max(2, int(n * row_ratio))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, k, replace=False)
        return X[idx], y[idx]
    
    def colinearity_treenames_mask(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treenames: np.ndarray,
        m_features: int,
        thr_crosscorr: float = 0.2,
        row_ratio: float = 1.0,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Select up to m_features that are:
        - highly correlated with y
        - not too correlated (>|thr_crosscorr|) with already selected features.
        """
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows")

        # Optional row subsampling
        Xc, yc = self._subsample_rows(X, y, row_ratio, random_state)

        # Feature–feature correlation
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(Xc, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        corr[np.abs(corr) > 0.5] = 0.0

        # Feature–target correlations via sklearn
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_y = r_regression(Xc, yc)        # shape (p,)
        corr_y = np.nan_to_num(corr_y, nan=0.0)
        corr_y[np.abs(corr_y) > 0.5] = 0.0

        # Order by descending |corr(feature, y)|
        order = np.argsort(-np.abs(corr_y))
        keep = np.zeros(p, dtype=bool)

        # Greedy selection
        for j in order:
            if keep.sum() >= m_features:
                break
            if not keep.any() or np.all(np.abs(corr[j, keep]) <= thr_crosscorr):
                keep[j] = True
                logger.debug(f"Selected feature {treenames[j]} with |corr|={np.abs(corr_y[j])}")

        return keep