
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from mrmr import mrmr_classif
import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter, CoxPHFitter
from collections import defaultdict
import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def survival_km_stats(df,cutoff):

    df["risk_group"] = np.where(df["y_probs"] > cutoff, "High-risk", "Low-risk")
    high = df[df["risk_group"] == "High-risk"]
    low = df[df["risk_group"] == "Low-risk"]

    results = logrank_test(high["time"], low["time"],
                        event_observed_A=high["event"],
                        event_observed_B=low["event"])
    p = results.p_value

    if p < 1e-4:
        p_text = "p < 0.0001"
    elif p < 1e-3:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p:.3g}"

    cph = CoxPHFitter()
    cph.fit(df[["time", "event", "risk_group"]].replace({"risk_group":{"Low-risk":0, "High-risk":1}}),
            duration_col="time", event_col="event")

    hr = np.exp(cph.params_["risk_group"])
    ci_low = np.exp(cph.confidence_intervals_.loc["risk_group","95% lower-bound"])
    ci_high = np.exp(cph.confidence_intervals_.loc["risk_group","95% upper-bound"])
    hr_text = f"HR = {hr:.2f} ({ci_low:.2f} – {ci_high:.2f})"
    return high, low, p_text,hr_text


def fit_cox_loo_oof_penalized(
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    k: int = 20,
    inner_splits: int = 4,
    penalizer_grid=None,
    corr_threshold: float = 0.9,
    random_state: int = 42):
    """
    Outer: Leave-One-Out (LOO)
    Inner: CV on training fold to tune CoxPHFitter penalizer
    Returns out-of-fold risk (partial hazard) per patient.
    Notes:
    - Risk returned is predict_partial_hazard (higher => higher risk).
    """

    df = df.copy()
    l1_ratio= 0.0 

    # Features: all columns except metadata
    X_cols = df.columns[:-3]
    feature_cols = list(X_cols)

    loo = LeaveOneOut()
    oof_risk = pd.Series(index=df.index, dtype=float)

    models = []
    selected_feats_per_fold = []
    best_penalizer_per_fold = []
    train_medians = [] 

    if penalizer_grid is None:
        penalizer_grid = np.logspace(-3, 2, 21) 

    # outer loop
    for fold, (tr_idx, te_idx) in enumerate(loo.split(df), start=1):
        te_ids = df.index[te_idx]

        tr_meta = df.iloc[tr_idx][[time_col, event_col]].copy()
        te_meta = df.iloc[te_idx][[time_col, event_col]].copy()

        X_tr_raw = df.iloc[tr_idx, :-3].copy()
        X_te_raw = df.iloc[te_idx, :-3].copy()

        # event label for MRMR and inner split stratification
        y_tr_event = df.iloc[tr_idx][event_col].astype(int).values
        # -------------------------
        # Standardize (fit on training only)
        # -------------------------
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(
            scaler.fit_transform(X_tr_raw),
            columns=X_tr_raw.columns,
            index=X_tr_raw.index,
        )
        X_te_scaled = pd.DataFrame(
            scaler.transform(X_te_raw),
            columns=X_te_raw.columns,
            index=X_te_raw.index,
        )

        # -------------------------
        # Drop highly correlated features (training only)
        # -------------------------
        X_tr_uncorr, kept_features, to_drop = drop_highly_correlated_features(
            X_tr_scaled, threshold=corr_threshold
        )
        # Align test to kept features
        X_te_uncorr = X_te_scaled[kept_features].copy()
        # -------------------------
        # MRMR feature selection on training only
        # -------------------------
        selected = mrmr_classif(
            X=X_tr_uncorr,
            y=y_tr_event,
            K=k,
            denominator="mean",
            relevance="f",
            show_progress=False,
            return_scores=True,
        )
        selected_cols = list(selected[0])
        selected_feats_per_fold.append(selected)
        tr_feat = X_tr_uncorr[selected_cols]
        te_feat = X_te_uncorr[selected_cols]

        tr_fit = pd.concat([tr_meta, tr_feat], axis=1)
        te_fit = pd.concat([te_meta, te_feat], axis=1)

        # -------------------------
        # Inner CV to tune penalizer on tr_fit
        # -------------------------
        skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        y_event_all = tr_fit[event_col].astype(int).values
        best_pen = None
        best_score = -np.inf
        for pen in penalizer_grid:
            fold_scores = []
            for inner_tr, inner_va in skf.split(tr_fit, y_event_all):
                
                df_in_tr = tr_fit.iloc[inner_tr].copy()
                df_in_va = tr_fit.iloc[inner_va].copy()
                #print("events:", df_in_va[event_col].sum())

                # if validation fold has no events, concordance is unstable; skip
                if df_in_va[event_col].sum() == 0:
                    print("validation no event")
                    continue
                cph_in = CoxPHFitter(penalizer=float(pen), l1_ratio=float(l1_ratio))
                cph_in.fit(df_in_tr, duration_col=time_col, event_col=event_col)

                # predict risk on validation
                va_risk = cph_in.predict_partial_hazard(df_in_va).values.reshape(-1)

                # concordance_index expects higher score = longer survival, so use -risk
                ci = concordance_index(
                    df_in_va[time_col].values,
                    -va_risk,
                    df_in_va[event_col].values)
                fold_scores.append(ci)

            if len(fold_scores) == 0:
                continue

            mean_ci = float(np.mean(fold_scores))
            if mean_ci > best_score:
                best_score = mean_ci
                best_pen = float(pen)

        if best_pen is None:
            print("return mid penalizer")
            best_pen = float(penalizer_grid[len(penalizer_grid) // 2])

        best_penalizer_per_fold.append(best_pen)

        # -------------------------
        # Fit final penalized Cox on outer training and predict held-out
        # -------------------------
        cph = CoxPHFitter(penalizer=float(best_pen), l1_ratio=float(l1_ratio))
        cph.fit(tr_fit, duration_col=time_col, event_col=event_col)

        te_risk = cph.predict_partial_hazard(te_fit).values.reshape(-1)
        oof_risk.loc[te_ids] = te_risk

        # store training median risk for CV-safe cutoff if you want it
        tr_risk = cph.predict_partial_hazard(tr_fit).values.reshape(-1)
        train_medians.append(tr_risk)

        models.append(cph)

    median_of_train_medians = float(np.median(train_medians))
    mean_of_train_medians = float(np.mean(train_medians))

    return {
        "df_aligned": df,
        "oof_risk": oof_risk,  # higher => higher risk
        "models": models,
        "feature_cols": feature_cols,
        "selected_feats_per_fold": selected_feats_per_fold,
        "best_penalizer_per_fold": best_penalizer_per_fold,
        "time_col": time_col,
        "event_col": event_col,
        "cv": "LOO (nested penalized Cox)",
        "median": median_of_train_medians,
        "mean": mean_of_train_medians,
        "k": k,
        "inner_splits": inner_splits,
        "penalizer_grid": np.array(penalizer_grid, dtype=float),
        "l1_ratio": float(l1_ratio),
        "corr_threshold": float(corr_threshold),
    }


def loo_cv_bootstrap_nested_lasso(
    X: pd.DataFrame,
    Y,
    K: int = 20,
    inner_splits: int = 5,
    C_grid=None,
    n_bootstrap: int = 2000,
    random_state: int = 42):
    """
    LOOCV outer loop + nested (inner) 5-fold CV to select best C for L1-logistic regression.
    Feature selection (mRMR) + scaling are fitted on the outer training set only.

    Returns:
      dict with ROC curve (mean_fpr/mean_tpr), LOOCV AUC, bootstrap CI, predictions, and per-fold metadata.
    """
    rng = np.random.RandomState(random_state)

    if C_grid is None:
        # reasonable default grid (log-spaced)
        C_grid = np.logspace(-4, 3, 24)
    #print(C_grid)

    Y = np.asarray(Y).astype(int)
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    selected_features_per_fold = []
    best_C_per_fold = []
    inner_cv_scores_per_fold = []
    model = []

    mean_fpr = np.linspace(0, 1, 200)

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(X, Y), start=1):
        #print(f"fold:{fold_i}")
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # -----------------------
        # 1) Scale (fit on train)
        # -----------------------
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train_raw.values)
        X_test_np = scaler.transform(X_test_raw.values)

        X_train = pd.DataFrame(X_train_np, columns=X.columns, index=X_train_raw.index)
        X_test = pd.DataFrame(X_test_np, columns=X.columns, index=X_test_raw.index)

        # -----------------------------------------
        # 2) mRMR feature selection on outer training
        # -----------------------------------------
        X_train, kept_features, to_drop = drop_highly_correlated_features(X_train,threshold=0.9)
        selected = mrmr_classif(
            X=X_train,
            y=y_train,
            K=K,
            denominator="mean",
            relevance="f",
            show_progress=False,
            return_scores=True,  # returns list of feature names
        )
        selected_features_per_fold.append(selected)
        #print(selected[0])

        X_train_sel = X_train[selected[0]]
        X_test_sel = X_test[selected[0]]

        # -------------------------------------------------
        # 3) Inner 5-fold CV to choose best C (nested CV)
        # -------------------------------------------------
        inner = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=42
        )

        mean_auc_by_C = []
        for C in C_grid:
            aucs = []
            for tr_in, va_in in inner.split(X_train_sel, y_train):
                X_tr_in = X_train_sel.iloc[tr_in]
                y_tr_in = y_train[tr_in]
                X_va_in = X_train_sel.iloc[va_in]
                y_va_in = y_train[va_in]
                #print("C",C,"inner test y count",np.bincount(y_va_in))
                # L1 Logistic Regression ("lasso")
                clf = LogisticRegression(
                    penalty="l1",
                    C=float(C),
                    solver="liblinear",
                    #max_iter=1000,
                    class_weight="balanced", 
                    random_state=42,
                )
                clf.fit(X_tr_in, y_tr_in)

                # AUC on validation fold
                p_va = clf.predict_proba(X_va_in)[:, 1]
                
                # guard: if a fold has only one class in y_va_in (rare but possible for tiny data)
                if len(np.unique(y_va_in)) < 2:
                    print("error")
                    continue
                aucs.append(roc_auc_score(y_va_in, p_va))

            mean_auc = np.mean(aucs) if len(aucs) > 0 else np.nan
            mean_auc_by_C.append(mean_auc)

        mean_auc_by_C = np.asarray(mean_auc_by_C, dtype=float)

        # pick best C (ignore NaNs)
        if np.all(np.isnan(mean_auc_by_C)):
            # fallback: middle of grid if inner CV failed
            print("inner_CV fail")
            best_C = float(C_grid[len(C_grid) // 2])
        else:
            #print("best c idx",mean_auc_by_C,np.nanargmax(mean_auc_by_C))
            best_C = float(C_grid[np.nanargmax(mean_auc_by_C)])

        best_C_per_fold.append(best_C)
        inner_cv_scores_per_fold.append(dict(zip(map(float, C_grid), mean_auc_by_C.tolist())))

        # -----------------------------------------
        # 4) Refit best-C model on full outer training
        # -----------------------------------------
        #print(f"best_C_{best_C}")
        best_clf = LogisticRegression(
            penalty="l1",
            C=best_C,
            solver="liblinear", 
            #max_iter=1000,
            class_weight="balanced",
            random_state=42)
        best_clf.fit(X_train_sel, y_train)
        model.append(best_clf)

        # single test sample prediction
        pred_prob = best_clf.predict_proba(X_test_sel)[:, 1][0]
        #print("y_test",y_test[0],"pred_prob",pred_prob)

        y_true_all.append(int(y_test[0]))
        y_pred_all.append(float(pred_prob))

    y_true_all = np.asarray(y_true_all, dtype=int)
    y_pred_all = np.asarray(y_pred_all, dtype=float)

    # ---------------------
    # Final LOOCV ROC + AUC
    # ---------------------
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    auc_final = roc_auc_score(y_true_all, y_pred_all)

    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0

    # ----------------------------
    # Bootstrap 95% CI for AUC
    # ----------------------------
    auc_bootstrap = []
    n = len(y_true_all)

    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true_all[idx])) < 2:
            continue
        auc_bootstrap.append(roc_auc_score(y_true_all[idx], y_pred_all[idx]))

    auc_bootstrap = np.asarray(auc_bootstrap, dtype=float)
    ci_lower = float(np.percentile(auc_bootstrap, 2.5))
    ci_upper = float(np.percentile(auc_bootstrap, 97.5))

    print(f"Nested-LOOCV LASSO AUC = {auc_final:.4f}")
    print(f"95% bootstrap CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "auc_mean": float(auc_final),
        "auc_ci": (ci_lower, ci_upper),
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "bootstrap_samples": auc_bootstrap,
        "selected_features_per_fold": selected_features_per_fold,
        "best_C_per_fold": np.asarray(best_C_per_fold, dtype=float),
        "inner_cv_scores_per_fold": inner_cv_scores_per_fold,
        "model_per_fold": model
    }



def remove_dup(graph_feature):
    X = graph_feature.copy()
    dup_mask = X.T.duplicated()
    dup_cols = X.columns[dup_mask].tolist()
    graph_feature_unique = X.loc[:, ~dup_mask].copy()
    return graph_feature_unique

def drop_highly_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.9,
):
    """
    Remove highly correlated features (|r| > threshold),
    keeping the first feature in each correlated group.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (training data only!)
    threshold : float
        Correlation threshold (default = 0.9)

    Returns
    -------
    X_reduced : pd.DataFrame
        DataFrame with correlated features removed
    kept_features : list
        List of kept feature names (in order)
    dropped_features : list
        List of dropped feature names
    """

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    kept_features = [c for c in X.columns if c not in to_drop]

    return X[kept_features], kept_features, to_drop




def summarize_feature_selection(combine_res, n_folds=None):
    """
    combine_res:
      - combine_res["selected_features_per_fold"][i][0] : list of top-20 feature names (mRMR)
      - combine_res["selected_features_per_fold"][i][1] : mRMR scores for all available features in that fold
            * can be: dict {feature: score} OR pd.Series OR 2-tuple (names, scores)
      - combine_res["model_per_fold"][i] : fitted sklearn LogisticRegression (L1) model for that fold
            * coef_ corresponds to the *same order* as the top-20 list used to fit the model

    Returns: pandas DataFrame
    """
    sel_per_fold = combine_res["selected_features_per_fold"]
    models = combine_res["model_per_fold"]

    if n_folds is None:
        n_folds = len(models)

    # --- trackers ---
    mrmr_selected_count = defaultdict(int)          # times in top-20
    lasso_selected_count = defaultdict(int)         # times coef != 0
    coef_sum = defaultdict(float)                   # sum of coefs across folds (0 if absent)
    abscoef_sum = defaultdict(float)                # sum of abs(coefs) across folds (0 if absent)

    # store mRMR scores when available (feature may not exist in a fold due to corr-filtering)
    mrmr_scores_by_feat = defaultdict(list)

    # union of all features we ever see
    all_feats = set()

    for i in range(n_folds):
        top_feats = sel_per_fold[i][0]
        all_feats.update(top_feats)

        # ---- mRMR score container parsing ----
        score_obj = sel_per_fold[i][1]
        score_map = None

        if isinstance(score_obj, dict):
            score_map = score_obj
        elif isinstance(score_obj, pd.Series):
            score_map = score_obj.to_dict()
        elif isinstance(score_obj, (list, tuple)) and len(score_obj) == 2:
            # assume (feature_names, scores)
            names, scores = score_obj
            score_map = dict(zip(list(names), list(scores)))
        else:
            # if it's already a dataframe or something else, try common patterns
            if hasattr(score_obj, "to_dict"):
                try:
                    score_map = score_obj.to_dict()
                except Exception:
                    score_map = None

        if score_map is not None:
            for f, s in score_map.items():
                # handle nested dicts from pandas .to_dict() variants
                if isinstance(s, dict):
                    # e.g., {'score': value}
                    # try grab first scalar
                    try:
                        s = list(s.values())[0]
                    except Exception:
                        continue
                if np.isscalar(s) and np.isfinite(s):
                    mrmr_scores_by_feat[f].append(float(s))
                    all_feats.add(f)

        # ---- mRMR selection frequency ----
        for f in top_feats:
            mrmr_selected_count[f] += 1

        # ---- LASSO selection + importance ----
        model = models[i]
        coef = np.asarray(model.coef_).ravel()   # (n_features,) for binary

        # IMPORTANT: coef must align with top_feats order used to train this model
        if len(coef) != len(top_feats):
            raise ValueError(
                f"Fold {i}: coef length ({len(coef)}) != number of top features ({len(top_feats)}). "
                "This means model coefficients are not aligned with selected features."
            )

        for f, c in zip(top_feats, coef):
            # accumulate across folds (0 if not present in other folds)
            coef_sum[f] += float(c)
            abscoef_sum[f] += float(abs(c))
            if c != 0:
                lasso_selected_count[f] += 1

    # --- build dataframe ---
    rows = []
    for f in sorted(all_feats):
        mean_mrmr = np.mean(mrmr_scores_by_feat[f]) if len(mrmr_scores_by_feat[f]) > 0 else np.nan

        rows.append({
            "feature": f,
            "mean_mrmr_score": mean_mrmr,
            "freq_mrmr_top20": mrmr_selected_count[f] / n_folds,
            "freq_lasso_nonzero": lasso_selected_count[f] / n_folds,
            "lasso_coef_mean": coef_sum[f] / n_folds,                 # includes zeros when absent
            "lasso_importance_mean_abs": abscoef_sum[f] / n_folds,    # includes zeros when absent
        })

    df = pd.DataFrame(rows)

    # Useful sorts depending on what you want to show:
    # 1) Stable final features first:
    df = df.sort_values(
        by=["freq_lasso_nonzero", "freq_mrmr_top20", "lasso_importance_mean_abs"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    df = df.set_index("feature")

    return df

