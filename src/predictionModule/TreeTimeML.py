import numpy as np
import polars as pl
import logging
import datetime

from sklearn.preprocessing import StandardScaler

from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.mathTools.DistributionTools import DistributionTools
from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.predictionModule.WeightSamples import WeightSamples

from src.hyperparameterTuning.HelperMetrics import HelperMetrics
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

logger = logging.getLogger(__name__)

class TreeTimeML:
    # Class-level default parameters
    treetime_params = {
        "TreeTime_top_n": 5,
    }
    loadup_params = {
        "idxAfterPrediction": 5,

        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }
    precompute_params = {
        "FilterSamples_cat_over10": False,
        "FilterSamples_cat_under5000": False,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.02": False,
        "FilterSamples_cat_volatility_qup0.975": False,
        "FilterSamples_cat_predictability_qup0.9": False,
        "FilterSamples_cat_predictability_qdown0.1": False,
        
        "volatility_w": 7,
        "volatility_dir": "qup",
        "volatility_q": 0.890885,
        
        "volprice_q": 0.586046,
        "volprice_w": 55,
        
        "predictability_w": 48,
        "predictability_dir": "qdown",
        "predictability_q": 0.314148,
    }
    
    base_params = {
        "LGB_num_boost_round"           : 1,
        "LGB_lambda_l1"                 : 0.000011,
        "LGB_lambda_l2"                 : 0.000041,
        "LGB_feature_fraction"          : 0.957214,
        "LGB_num_leaves"                : 2130,
        "LGB_max_depth"                 : 25,
        "LGB_learning_rate"             : 0.1,
        "LGB_min_data_in_leaf"          : 20, 
        "LGB_min_gain_to_split"         : 2.467275e-06,
        "LGB_path_smooth"               : 0.6,
        "LGB_min_sum_hessian_in_leaf"   : 0.275354,
        "LGB_max_bin"                   : 240,
        "LGB_early_stopping_rounds"     : 20,

        "do_transform"                  : True,
        "tree_n_max"                    : 1,
        "min_n_tar_daily"               : 3,
        "top_n_max"                     : 450,  
              
        "inc_FeatureTA"                 : False,
        "inc_GroupDynamics"             : False,
        "inc_Categorical"               : True,
        "inc_Financials"                : True,
        "inc_Mathematical"              : True,
        "inc_Seasonal"                  : True,
        "exc_lag"                       : True,
        "ytree_kind"                    : "abslast", 

        "t_win"                         : 23,
    }

    def __init__(
            self, 
            train_start_date: datetime.date,
            test_dates: list[datetime.date],
            treegroup: str,
            timegroup: str,
            params: dict = None,
            loadup: LoadupSamples = None
        ):
        
        self.params = {**self.loadup_params, **(params or {})}
        self.treegroup = treegroup
        self.timegroup = timegroup
        self.train_start_date = train_start_date
        self.test_dates = test_dates
        
        self.min_test_date = min(self.test_dates)
        self.max_test_date = max(self.test_dates)
        
        # Assign parameters to instance variables
        if loadup is None or not isinstance(loadup, LoadupSamples):
            ls = LoadupSamples(
                train_start_date=self.train_start_date,
                test_dates=self.test_dates,
                treegroup=self.treegroup,
                timegroup=self.timegroup,
                params=self.loadup_params,
            )
        else:
            ls = loadup
            if ls.treegroup != self.treegroup:
                raise ValueError("Provided LoadupSamples does not match the treegroup.")
            if ls.timegroup != self.timegroup:
                raise ValueError("Provided LoadupSamples does not match the timegroup.")
            if ls.train_start_date != self.train_start_date:
                raise ValueError("Provided LoadupSamples does not match the train start date.")
            if ls.test_dates != self.test_dates:
                raise ValueError("Provided LoadupSamples does not match the test dates.")
            if any(ls.params.get(k, None) is None for k, v in self.loadup_params.items()):
                raise ValueError("Provided LoadupSamples does not exist in the loadup parameters.")
            if any(ls.params.get(k) != v for k, v in self.loadup_params.items()):
                raise ValueError("Provided LoadupSamples does not match the loadup parameters.")
            self.test_dates = ls.test_dates
            
        self.idxDaysAfter = ls.idxAfter
        self.timesteps = ls.timesteps
        
        self.featureTreeNames: list[str] | None = ls.featureTreeNames
        self.featureTimeNames: list[str] | None = ls.featureTimeNames
        self.meta_pl_train: pl.DataFrame = ls.meta_pl_train
        self.meta_pl_test: pl.DataFrame  = ls.meta_pl_test
        
        self.train_Xtree: np.ndarray = ls.train_Xtree
        self.train_Xtime: np.ndarray = ls.train_Xtime
        self.train_ytree: np.ndarray = ls.train_ytree
        self.train_ytime: np.ndarray = ls.train_ytime
        
        self.test_Xtree: np.ndarray = ls.test_Xtree
        self.test_Xtime: np.ndarray = ls.test_Xtime
        self.test_ytree: np.ndarray = ls.test_ytree
        self.test_ytime: np.ndarray = ls.test_ytime
        
        self.train_ytree_low: np.ndarray  = ls.train_ytree_low
        self.train_ytree_open: np.ndarray = ls.train_ytree_open
        self.train_ytree_high: np.ndarray = ls.train_ytree_high
        self.test_ytree_low: np.ndarray   = ls.test_ytree_low
        self.test_ytree_open: np.ndarray  = ls.test_ytree_open
        self.test_ytree_high: np.ndarray  = ls.test_ytree_high
        
        self.mask_train = np.ones(self.train_Xtree.shape[0], dtype=bool)
        self.mask_test = np.ones(self.test_Xtree.shape[0], dtype=bool)
        
        self.sl_tr_vec = np.zeros(self.train_Xtree.shape[0], dtype=float)
        self.sl_te_vec = np.zeros(self.test_Xtree.shape[0], dtype=float)
        self.tp_tr_vec = np.zeros(self.train_Xtree.shape[0], dtype=float)
        self.tp_te_vec = np.zeros(self.test_Xtree.shape[0], dtype=float)
        
        logger.info("Initialized TreeTimeML with parameters:")
        for k,v in self.treetime_params.items():
            logger.info(f"  {k}: {v}")
        for k,v in self.loadup_params.items():
            logger.info(f"  {k}: {v}")
        for k,v in self.precompute_params.items():
            logger.info(f"  {k}: {v}")
        for k,v in self.base_params.items():
            logger.info(f"  {k}: {v}")
        
    def run_ext(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
    ) -> tuple[np.ndarray, ...]:
        opt_params = self.base_params
        
        t_win =             opt_params["t_win"]
        do_transform =      opt_params["do_transform"]
        tree_n_max =        opt_params["tree_n_max"]
        top_n_max =         opt_params["top_n_max"]
        min_n_tar_daily =   opt_params["min_n_tar_daily"]
        n_dates_test =      meta_test.get_column("date").n_unique()
        min_n_tar =         min_n_tar_daily * n_dates_test
        mm: MachineModels = MachineModels(opt_params)
            
        logger.info(f"  Before filtering: tr {Xtr_tree.shape}, te {Xte_tree.shape}")
            
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        if opt_params["ytree_kind"] == "last":
            ytr_tree_opt = ytr_tree[:, -1]
        elif opt_params["ytree_kind"] == "mean":
            ytr_tree_opt = np.mean(ytr_tree, axis=1)
        elif opt_params["ytree_kind"] == "abslast":
            ytr_tree_opt = np.abs(ytr_tree[:, -1])
        elif opt_params["ytree_kind"] == "max":
            ytr_tree_opt = np.max(ytr_tree, axis=1)

        tn = np.asarray(treenames, dtype=str)
        mask_treenames = np.zeros(len(treenames), dtype=bool)
        if opt_params.get("inc_FeatureTA"):
            mask_treenames |= np.char.find(tn, "FeatureTA_") >= 0
        if opt_params.get("inc_GroupDynamics"):
            mask_treenames |= np.char.find(tn, "FeatureGroup_") >= 0
        if opt_params.get("inc_Categorical"):
            mask_treenames |= np.char.find(tn, "Category_") >= 0
        if opt_params.get("inc_Financials"):
            mask_treenames |= np.char.find(tn, "FinData_") >= 0
        if opt_params.get("inc_Mathematical"):
            mask_treenames |= np.char.find(tn, "MathFeature_") >= 0
        if opt_params.get("inc_Seasonal"):
            mask_treenames |= np.char.find(tn, "Seasonal_") >= 0
        if opt_params.get("exc_lag"):
            mask_treenames &= np.char.find(tn, "_lag") < 0
            
        Xdtime_tr, _ = self._make_design(Xtr_time[mask_train], t_win)
        Xdtime_te, _ = self._make_design(Xte_time[mask_test], t_win)

        Xd_tr = np.hstack((Xtr_tree[mask_train][:, mask_treenames], Xdtime_tr))
        ytr_tree_opt = ytr_tree_opt[mask_train]
        Xd_te = np.hstack((Xte_tree[mask_test][:, mask_treenames], Xdtime_te))
        
        logger.info(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.info(f"   ytr_tree: n={ytr_tree_opt.size}, mean={ytr_tree_opt.mean():.6f}, std={ytr_tree_opt.std():.6f}")

        if do_transform:
            scaler = StandardScaler().fit(Xd_tr)
            Xd_tr = scaler.transform(Xd_tr)
            Xd_te = scaler.transform(Xd_te)

        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr,
                y_train=ytr_tree_opt,
                X_test=None,
                y_test=None,
            )
        except Exception as e:
            logger.disabled = False
            logger.warning(f"  LGB failed: {e}")
            return 1.0
        finally:
            logger.disabled = False

        # Log feature importance
        base = list(timenames[:5])
        T = t_win + 1
        t_feat_names = [
            (f"{base[f]}_t-{t_win - t}" if t_win - t else f"{base[f]}_t")
            for t in range(T) for f in range(5)
        ]
        colnames = tn[mask_treenames].tolist() + t_feat_names
        ModelAnalyzer.print_feature_importance_LGBM(lgbModel=model_lgb, featureColumnNames=colnames, n_feature=5)

        tree_n_max = min(model_lgb.num_trees(), tree_n_max)
        labels_top, scores_top = HelperFunctions.top_leaf_labels_per_tree(
            model_lgb, Xd_tr, ytr_tree_opt, tree_n_max=tree_n_max, top_n_max=max(1, top_n_max or 1)
        )
        n_trees = labels_top.shape[1]
        
        # Top labels by score
        leaf_tr = model_lgb.predict(Xd_tr, pred_leaf=True)
        leaf_tr = leaf_tr.reshape(-1, 1) if leaf_tr.ndim == 1 else leaf_tr  # shape (n_samples, n_trees)
        leaf_tr = leaf_tr[:, :n_trees]  # restrict to used trees
        leaf_te = model_lgb.predict(Xd_te, pred_leaf=True)
        leaf_te = leaf_te.reshape(-1, 1) if leaf_te.ndim == 1 else leaf_te  # shape (n_samples, n_trees)
        leaf_te = leaf_te[:, :n_trees]  # restrict to used trees

        # If everything is -1 across all ranks, bail out
        if labels_top.size == 0 or np.all(labels_top == -1):
            logger.warning("  LGB failed to generate predictions.")
            return 1.0
        
        # Rank every (rank, tree) pair by descending score
        r_idx, t_idx = np.unravel_index(np.argsort(scores_top.ravel())[::-1], scores_top.shape)
        
        sel_pairs = []  # (tree_idx, rank_idx)
        mask_sel = np.zeros(leaf_te.shape[0], dtype=bool)
        score_sel = np.zeros(leaf_te.shape[0], dtype=float)
        mask_tr = np.zeros(Xd_tr.shape[0], dtype=bool)     
        score_tr = np.zeros(Xd_tr.shape[0], dtype=float)   
        for i in range(len(r_idx)):
            r = r_idx[i]
            t = t_idx[i]
            lbl = labels_top[r, t]
            if lbl == -1:
                continue
            mask_sel |= (leaf_te[:, t] == lbl)
            mask_tr |= (leaf_tr[:, t] == lbl)
            
            score_sel[leaf_te[:, t] == lbl] = 1/(np.log(i + 1) + 1)
            score_tr[leaf_tr[:, t] == lbl] = 1/(np.log(i + 1) + 1)
            
            sel_pairs.append((int(lbl), int(t), int(r)))
            if mask_sel.sum() >= min_n_tar:
                break

        res_tr_mask = np.zeros(Xtr_tree.shape[0], dtype=bool)
        res_tr_mask[mask_train] = mask_tr
        res_te_mask = np.zeros(Xte_tree.shape[0], dtype=bool)
        res_te_mask[mask_test] = mask_sel
        
        score_tr_full = np.zeros(Xtr_tree.shape[0], dtype=float)
        score_tr_full[mask_train] = score_tr
        score_te_full = np.zeros(Xte_tree.shape[0], dtype=float)
        score_te_full[mask_test] = score_sel
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[res_tr_mask], 
            ytr_tree_low[res_tr_mask], 
            ytr_tree_high[res_tr_mask], 
            ytr_tree_open[res_tr_mask],
            n_grid=10
        )
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        return res_tr_mask, res_te_mask, sl_tr, sl_te, tp_tr, tp_te, score_tr_full, score_te_full
    
    def precompute(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        params = self.precompute_params

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
        
        def apply_step(
            mask_train: np.ndarray,
            mask_test: np.ndarray,
            params: dict
        ):
            fs = FilterSamples(
                Xtree_train=Xtr_tree[mask_train],
                ytree_train=ytr_tree[mask_train][:, -1],
                treenames=treenames,
                Xtree_test=Xte_tree[mask_test],
                ytree_test=None,
                meta_train=meta_train.filter(mask_train),
                meta_test=meta_test.filter(mask_test),
                params=params,
            )

            cat_train, cat_test = fs.categorical_masks()

            # Update masks in-place while preserving alignment
            mask_train[mask_train] &= cat_train
            if cat_test is not None:
                mask_test[mask_test] &= cat_test

            return fs, mask_train, mask_test
        
        volat_w = params["volatility_w"]
        volat_dir = params["volatility_dir"]
        volat_q = params["volatility_q"]
        
        volprice_q = params["volprice_q"]
        volprice_w = params["volprice_w"]
        
        predic_w = params["predictability_w"]
        predic_dir = params["predictability_dir"]
        predic_q = params["predictability_q"]
        
        def q_limit(mask_test):
            min_n_tar_daily = 30
            n_dates_test = meta_test.get_column("date").n_unique()
            return min(n_dates_test * min_n_tar_daily / mask_test.sum(), 1.0)
        
        q_l = q_limit(mask_test)
        q = min(volprice_q, 1-q_l)
        key = f"FilterSamples_cat_volumeprice_w{volprice_w}_q{q:.2f}"
        step_params = dict(params)
        step_params[key] = True
        logger.debug("Applying second category filter: %s", key)
        _, mask_train, mask_test = apply_step(mask_train, mask_test, step_params)
        logger.debug(f"  After second cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")

        
        q_l = q_limit(mask_test)
        q = min(volat_q, 1-q_l)
        key = f"FilterSamples_cat_volatility_w{volat_w}_{volat_dir}{q:.2f}"
        step_params = dict(params)
        step_params[key] = True
        logger.debug("Applying third category filter: %s", key)
        _, mask_train, mask_test = apply_step(mask_train, mask_test, step_params)
        logger.debug(f"  After third cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")
        
        q_l = q_limit(mask_test)
        q = max(predic_q, q_l)
        key = f"FilterSamples_cat_predictability_w{predic_w}_{predic_dir}{q:.2f}"
        step_params = dict(params)
        step_params[key] = True
        logger.debug("Applying first category filter: %s", key)
        _, mask_train, mask_test = apply_step(mask_train, mask_test, step_params)
        logger.debug(f"  After first cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")

                    
        unique_tickers = meta_test.filter(mask_test).get_column("ticker").unique().to_numpy()
        logger.debug(f"  Precompute -> test unique tickers kept: {unique_tickers.size} | total: {len(meta_test.get_column('ticker').unique())}")
        logger.debug(f"    Tickers: {unique_tickers}")

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid = 5
        )        
        sl_tr_vec = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te_vec = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr_vec = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te_vec = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.info(
            "  Precompute -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_val} | tp: {tp_val}")

        return mask_train, mask_test, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_design(self, X, t_win):
        Xw: np.ndarray = X[:, -(t_win+1):, 0:5].copy()
        Xw = (Xw-0.5)*2.0
        
        mask_bad = np.zeros(Xw.shape[0], dtype=bool)
        bound_bad = 1 - np.tanh(1 - 1e-4)
        mask_bad = np.any((Xw[:,:,0:4] <= (-1+bound_bad)) | (Xw[:,:,0:4] >= (1-bound_bad)), axis=(1,2))
        Xw[:,:,0:4] = np.clip(Xw[:,:,0:4], -1+bound_bad, 1-bound_bad)
        
        Xw[:,:,0:4] = np.arctanh(Xw[:,:,0:4]) + 1.0
        
        return Xw.reshape(Xw.shape[0], -1), mask_bad
    
    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    
    def _pipeline_res_analyze(self, stage: str, mode: str = "predict") -> None:
        res_tr_vec = HelperMetrics.collapse_sl_tp(
            self.train_ytree, 
            self.train_ytree_low, 
            self.train_ytree_high, 
            self.train_ytree_open, 
            self.sl_tr_vec, 
            self.tp_tr_vec, 
            spread_cost=0.000, 
            commission=0.000
        )
        if mode == "analyze":
            res_te_vec = HelperMetrics.collapse_sl_tp(
                    self.test_ytree, 
                    self.test_ytree_low, 
                    self.test_ytree_high, 
                    self.test_ytree_open, 
                    self.sl_te_vec, 
                    self.tp_te_vec, 
                    spread_cost=0.000, 
                    commission=0.000
                )
        logger.info(f"  After {stage}: ")
        logger.info(f"    Ratio training samples: {self.mask_train.sum() / len(self.mask_train)}")
        logger.info(f"    Ratio test samples: {self.mask_test.sum() / len(self.mask_test)}")
        
        pre_tr_score = HelperMetrics.evaluate_mask_nullonempty(self.mask_train, self.meta_pl_train['date'], res_tr_vec)
        logger.info(f"    {stage} train score: {pre_tr_score:.4f}")
        
        if mode == "analyze":
            pre_te_score = HelperMetrics.evaluate_mask_nullonempty(self.mask_test, self.meta_pl_test['date'], res_te_vec)
            logger.info(f"    {stage} test score: {pre_te_score:.4f}")
    
            sl_hits = (self.test_ytree_low[self.mask_test][:, -1] <= self.sl_te_vec[self.mask_test])
            tp_hits = (self.test_ytree_high[self.mask_test][:, -1] >= self.tp_te_vec[self.mask_test])
            tp_nosl_hits = tp_hits & ~sl_hits
            logger.info(f"    {stage} ratio sl hits test split: {np.sum(sl_hits)/len(self.sl_te_vec[self.mask_test]):.4f}, n_test_samples = {len(self.sl_te_vec[self.mask_test])}")
            logger.info(f"    {stage} ratio tp hits test split: {np.sum(tp_hits)/len(self.tp_te_vec[self.mask_test]):.4f}, n_test_samples = {len(self.tp_te_vec[self.mask_test])}")
            logger.info(f"    {stage} ratio tp no sl hits test split: {np.sum(tp_nosl_hits)/len(self.tp_te_vec[self.mask_test]):.4f}, n_test_samples = {len(self.tp_te_vec[self.mask_test])}")
        
    def pipeline(self, mode: str = "predict") -> dict:
        """
        Common pipeline steps shared by both analyze() and predict().
        Returns a dictionary of all relevant masked data, trained model, and predictions.
        """

        ########################
        ## PRE FILTER SAMPLES ##
        ########################
        logger.info("Running pre-filtering samples...")
        try:
            mask_tr, mask_te, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec = self.precompute(
                self.train_Xtree.copy(),
                self.train_Xtime.copy(),
                self.train_ytree.copy(),
                self.train_ytree_low.copy(),
                self.train_ytree_high.copy(),
                self.train_ytree_open.copy(),
                self.test_Xtree.copy(),
                self.test_Xtime.copy(),
                self.featureTreeNames,
                self.featureTimeNames,
                self.meta_pl_train,
                self.meta_pl_test,
            )
        except Exception as e:
            logger.warning(f"  Error occurred while pre-filtering samples: {e}")
            return {
                'y_test_scores': np.zeros(self.test_Xtree.shape[0], dtype=float),
                'mask_train': np.zeros(self.train_Xtree.shape[0], dtype=bool),
                'mask_test': np.zeros(self.test_Xtree.shape[0], dtype=bool),
            }
        self.sl_tr_vec = sl_tr_vec
        self.sl_te_vec = sl_te_vec
        self.tp_tr_vec = tp_tr_vec
        self.tp_te_vec = tp_te_vec
        self.mask_train = mask_tr
        self.mask_test = mask_te
        
        self._pipeline_res_analyze(
            stage="Pre-filter", 
            mode=mode, 
        )


        #########################
        ## MAIN FILTER SAMPLES ##
        #########################
        logger.info("Running main filtering samples...")
        try:
            mask_tr, mask_te, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec, vecscore_tr_vec, vecscore_te_vec = self.run_ext(
                self.train_Xtree[self.mask_train],
                self.train_Xtime[self.mask_train],
                self.train_ytree[self.mask_train],
                self.train_ytree_low[self.mask_train],
                self.train_ytree_high[self.mask_train],
                self.train_ytree_open[self.mask_train],
                self.test_Xtree[self.mask_test],
                self.test_Xtime[self.mask_test],
                self.featureTreeNames,
                self.featureTimeNames,
                self.meta_pl_train.filter(pl.Series(self.mask_train)),
                self.meta_pl_test.filter(pl.Series(self.mask_test)),
            )
        except Exception as e:
            logger.warning(f"  Error occurred while main-filtering samples: {e}")
            return {
                'y_test_scores': np.zeros(self.test_Xtree.shape[0], dtype=float),
                'mask_train': np.zeros(self.train_Xtree.shape[0], dtype=bool),
                'mask_test': np.zeros(self.test_Xtree.shape[0], dtype=bool),
            }
        self.sl_tr_vec[self.mask_train]  = sl_tr_vec
        self.sl_te_vec[self.mask_test]   = sl_te_vec
        self.tp_tr_vec[self.mask_train]  = tp_tr_vec
        self.tp_te_vec[self.mask_test]   = tp_te_vec
        
        self.mask_train[self.mask_train] = mask_tr
        self.mask_test[self.mask_test]   = mask_te
        
        self._pipeline_res_analyze(
            stage="Main-run", 
            mode=mode, 
        )

        score_te_full = np.zeros(self.test_Xtree.shape[0], dtype=float)
        score_te_full[self.mask_test] = vecscore_te_vec[mask_te]

        #############
        ## RETURNS ##
        #############
        return {
            'y_test_scores': score_te_full,
            'mask_train': self.mask_train,
            'mask_test': self.mask_test,
        }

    def __get_top_tickers(self, y_test_scores: np.ndarray) -> pl.DataFrame:
        m = self.params['TreeTime_top_n']
        
        res_pl = self.meta_pl_test.filter(pl.Series(self.mask_test)).with_columns(
            pl.Series("test_scores", y_test_scores[self.mask_test]),
        )
        
        res_pl = (
            res_pl
            .sort(["date", "test_scores"], descending=[False, True])
            .with_columns(
                pl.col("test_scores")
                .rank(method="random", descending=True)
                .over("date")
                .alias("score_rank")
            ).filter(
                pl.col("score_rank") <= m
            )
        )

        return res_pl
    
    def _get_res_df(self, y_test_scores: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame]:
        m = self.treetime_params['TreeTime_top_n']
        
        collapsed_te_vec = HelperMetrics.collapse_sl_tp(
            self.test_ytree[self.mask_test], 
            self.test_ytree_low[self.mask_test], 
            self.test_ytree_high[self.mask_test], 
            self.test_ytree_open[self.mask_test], 
            self.sl_te_vec[self.mask_test], 
            self.tp_te_vec[self.mask_test], 
            spread_cost=0.000, 
            commission=0.000
        )
        
        res_pl = self.meta_pl_test.filter(pl.Series(self.mask_test)).with_columns(
            pl.Series("collapsed_ratio", collapsed_te_vec),
            pl.Series("test_scores", y_test_scores[self.mask_test]),
        )
        
        res_pl = (
            res_pl
            .sort(["date", "test_scores"], descending=[False, True])
            .with_columns(
                pl.col("test_scores")
                .rank(method="random", descending=True)
                .over("date")
                .alias("test_scores_rank")
            ).filter(
                pl.col("test_scores_rank") <= m
            )
        )
        
        res_pl_perdate = (
            res_pl.group_by("date").agg([
                pl.count().alias("n_entries"),
                pl.col("collapsed_ratio").mean().alias("mean_collapsed_ratio"),
            ])
        )
        
        return res_pl, res_pl_perdate
    
    def analyze(self, logger_disabled: bool = False) -> tuple[float, dict]:
        logger_config = logger.disabled
        logger.disabled = logger_disabled
        
        # Run common pipeline in "analyze" mode
        data = self.pipeline(mode="analyze")
        
        # Additional analysis with test set
        y_test_scores: np.ndarray = data['y_test_scores']
        mask_test: np.ndarray = data['mask_test']
        
        if mask_test.sum() == 0:
            return (
                1.0, 
                {
                    "df_pred_res": pl.DataFrame(),
                    "df_pred_res_perdate": pl.DataFrame(),
                }
            )
        
        res_df, res_df_perdate = self._get_res_df(y_test_scores)
        
        logger.info("Analyzing test set predictions:")
        logger.info(f"  Number of test dates: {len(self.test_dates)}")
        logger.info(f"  Ratio of test dates with choices: {res_df_perdate.shape[0] / len(self.test_dates):.4f}")

        score_col = "test_scores"
        tar_col = "collapsed_ratio"
        ModelAnalyzer.log_test_result_perdate(res_df, self.test_dates, score_col = score_col, tar_col = tar_col)
        ModelAnalyzer.log_test_result_overall(res_df, score_col = score_col, last_col = tar_col)

        res_df_perdate = res_df.group_by("date").agg([
            pl.col(tar_col).mean().alias("mean_res"),
            pl.col(tar_col).first().alias("top_res"),
            pl.col(tar_col).count().alias("n_entries"),
            pl.col(score_col).max().alias("max_score"),  # this is also .first()
            pl.col(score_col).mean().alias("mean_score"),
        ])

        logger.disabled = logger_config
        return (
            res_df_perdate['mean_score'].mean(), 
            {
                "res_df": res_df.select(['date', 'ticker', 'Close', score_col, tar_col]),
                "res_df_perdate": res_df_perdate.select(['date', 'n_entries', 'max_score', 'mean_score', 'top_res', 'mean_res'])
            }
        )

    def predict(self, logger_disabled: bool = False) -> tuple[float, dict]:
        logger_config = logger.disabled
        logger.disabled = logger_disabled
        
        # Run common pipeline in "analyze" mode
        data = self.pipeline(mode="predict")

        # Additional analysis with test set
        y_test_scores: np.ndarray = data['y_test_scores']
        mask_test: np.ndarray = data['mask_test']
        
        if mask_test.sum() == 0:
            return (
                1.0, 
                {
                    "df_pred_res": pl.DataFrame(),
                    "df_pred_res_perdate": pl.DataFrame(),
                }
            )

        res_df = self.__get_top_tickers(y_test_scores)

        score_col = "test_scores"
        ModelAnalyzer.log_test_result_perdate(res_df, self.test_dates, score_col = score_col, last_col = None)
        ModelAnalyzer.log_test_result_overall(res_df, score_col = score_col, last_col = None)

        res_df_perdate = res_df.group_by("date").agg([
            pl.col(score_col).count().alias("n_entries"),
            pl.col(score_col).mean().alias("mean_score"),
            pl.col(score_col).max().alias("max_score"),
        ])

        logger.disabled = logger_config
        return (
            res_df_perdate['mean_score'].mean(), 
            {
                "res_df": res_df.select(['date', 'ticker', 'Close', score_col]),
                "res_df_perdate": res_df_perdate.select(['date', 'n_entries', 'max_score', 'mean_score']),
            }
        )