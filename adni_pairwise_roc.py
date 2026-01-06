# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import warnings
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")


USE_CLASS_WEIGHTS = True          # Whether to enable class weights
CLASS_WEIGHTS_MODE = "auto"       # "auto": auto_class_weights='Balanced'; or "manual"
USE_RESAMPLING   = True           # Whether to apply resampling on training folds
RESAMPLER_NAME   = "SMOTE"        # "SMOTE"|"ROS"|"RUS"|"SMOTEENN"|"NONE"

# ============ Plotting and evaluation switches ============
PLOT_MULTICLASS_CM = True        
PLOT_PAIRWISE_CM   = False       
PLOT_PAIRWISE_ROC  = True         
SHOW_DIAGNOSTICS   = False       


data_path        = r'c:\Users\24426\Desktop\brain_age\surface\adni_voxel.csv'
information_path = r'c:\Users\24426\Desktop\brain_age\surface\adni.csv'
surface_path     = r'c:\Users\24426\Desktop\brain_age\surface\ADNI_thickness.csv'

def read_csv_file(data_path, information_path, method='voxel'):
    data = pd.read_csv(data_path)
    information = pd.read_csv(information_path)
    del information['Unnamed: 0']
    if method == 'voxel':
        left_brain = [col for col in data.columns if 'L' in col and col not in ['correctAge', 'trueAge', 'group', 'sex']]
        right_brain = [col for col in data.columns if 'R' in col and col not in ['correctAge', 'trueAge', 'group', 'sex']]
        left_brain = data[left_brain]
        right_brain = data[right_brain]
        asymmetry_index = (left_brain.values - right_brain.values)
        asymmetry_index = pd.DataFrame(asymmetry_index, columns=[f"AI_{col[:-1]}" for col in left_brain])

    elif method == 'surface':
        left_brain = [col for col in data.columns if col.startswith("l") and col.endswith("'")]
        right_brain = [col for col in data.columns if col.startswith("r") and col.endswith("'")]
        left_brain = data[left_brain]
        right_brain = data[right_brain]
        asymmetry_index = (left_brain.values - right_brain.values)
        asymmetry_index = pd.DataFrame(asymmetry_index, columns=[f"AI_{col[1:-1]}" for col in left_brain])
    else:
        raise ValueError("Invalid method specified for extracting brain information")

    asymmetry_index['id'] = data['id']
    merged_data = pd.merge(information, asymmetry_index, on='id', suffixes=('', '_asymmetry'))
    
    return merged_data


print("[Loading data...]")
data_voxel   = read_csv_file(data_path, information_path, method='voxel')
data_surface = read_csv_file(surface_path, information_path, method='surface')
merged_data  = pd.merge(data_surface, data_voxel, on='id', suffixes=('_surface', '_voxel'))
merged_data.columns = [col.replace('AI_', 'AM_') for col in merged_data.columns]


x = merged_data.filter(like='AM_').copy()
x.columns = ['AM_BSTS', 'AM_CACC', 'AM_CMF',
 'AM_CUN', 'AM_ENT', 'AM_FUS', 'AM_IPL',
 'AM_ITG', 'AM_ISTC', 'AM_LOC',
 'AM_LOFC', 'AM_LIN', 'AM_MOFC',
 'AM_MTG', 'AM_PHG', 'AM_PCEN',
 'AM_POP', 'AM_PORB', 'AM_PTRI',
 'AM_PERI', 'AM_POSTC', 'AM_PCC',
 'AM_PREC', 'AM_PRECUN', 'AM_RACC',
 'AM_RMF', 'AM_SFG', 'AM_SPL',
 'AM_STG', 'AM_SMG', 'AM_FP',
 'AM_TP', 'AM_TT', 'AM_INS', 'AM_AMYG',
 'AM_BG', 'AM_HIPP', 'AM_THA']

select_columns = ['AM_CACC', 'AM_ITG', 'AM_PRECUN', 'AM_AMYG', 'AM_LIN', 'AM_BG', 'AM_HIPP']
BAG_NAME = 'BAG'
x[BAG_NAME] = merged_data['bag_surface']
y = merged_data['Group_surface']
x = x.dropna()
y = y.loc[x.index]

# === Three feature configurations ===
# 1) BAG only
X_feat_BAG = x[[BAG_NAME]].copy()
# 2) Selected asymmetry measures only
X_feat_SEL = x[select_columns].copy()
# 3) BAG + selected asymmetry measures
X_feat_ALL = x[[BAG_NAME] + select_columns].copy()

feature_sets = {
    "BAG": X_feat_BAG,
    "SAM": X_feat_SEL,
    "BAG+SAM": X_feat_ALL,
}

feature_names_all = {k: v.columns.tolist() for k, v in feature_sets.items()}
classes_all = np.unique(y)
print(f"[Sample size: {len(x)}, Classes: {list(classes_all)}]")
print("[Feature configurations]")
for k, v in feature_sets.items():
    print(f"  - {k}: {list(v.columns)} ({v.shape[1]} features)")

# ============ 5-fold stratified cross-validation ============
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []            
pairwise_metrics = []       
fig_list_multiclass = []     
fig_list_pairwise = []       

class_order = ["CN", "SCD", "MCI", "AD"]
pairs = list(combinations(class_order, 2)) 

def plot_percent_cm_with_single_colorbar(cm_counts, labels, title):
    
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_norm = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, include_values=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    cbar = fig.colorbar(disp.im_, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    for (i, j), v in np.ndenumerate(cm_norm):
        ax.text(j, i, f"{v*100:.1f}%", ha='center', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()
    return fig

def build_resampler(name):
    name = (name or "NONE").upper()
    if name == "SMOTE":
        return SMOTE(k_neighbors=5, random_state=42)
    if name == "ROS":
        return RandomOverSampler(random_state=42)
    if name == "RUS":
        return RandomUnderSampler(random_state=42)
    if name == "SMOTEENN":
        return SMOTEENN(random_state=42)
    return None  # NONE

def build_cb_params(y_train):
    n_classes = y_train.nunique()
    cb_params = dict(
        iterations=200, depth=15, learning_rate=0.05, l2_leaf_reg=3,
        random_state=42, verbose=0, task_type='CPU'
    )
    if n_classes == 2:
        cb_params.update({"loss_function": "Logloss", "eval_metric": "BalancedAccuracy"})
    else:
        cb_params.update({"loss_function": "MultiClass", "eval_metric": "TotalF1"})

    if USE_CLASS_WEIGHTS:
        if CLASS_WEIGHTS_MODE.lower() == "auto":
            cb_params["auto_class_weights"] = "Balanced"  # or "SqrtBalanced"
        else:
            vc = y_train.value_counts()
            weights = [(len(y_train) / (len(vc) * max(vc.get(c, 0), 1))) for c in class_order]
            cb_params["class_weights"] = weights
    return cb_params

def fit_predict_proba(X_train, X_test, y_train):
    cb_params = build_cb_params(y_train)
    clf = CatBoostClassifier(**cb_params)
    resampler = build_resampler(RESAMPLER_NAME) if USE_RESAMPLING else None

    if resampler is not None:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("res", resampler),
            ("clf", clf),
        ])
        estimator.fit(X_train, y_train)
        y_proba = estimator.predict_proba(X_test)
        model_classes = list(estimator.named_steps["clf"].classes_)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        clf.fit(X_train_scaled, y_train)
        y_proba = clf.predict_proba(X_test_scaled)
        model_classes = list(clf.classes_)
    return y_proba, model_classes

def align_proba_to_order(y_proba, model_classes, class_order):
    out = np.zeros((y_proba.shape[0], len(class_order)))
    for j, c in enumerate(class_order):
        if c in model_classes:
            out[:, j] = y_proba[:, model_classes.index(c)]
    return out

oof_buckets = {name: [] for name in feature_sets.keys()}

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_feat_ALL.values, y.values), start=1):
    print(f"\n==== Fold {fold_idx} / 5 ====")
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_all, X_test_all = X_feat_ALL.iloc[train_idx], X_feat_ALL.iloc[test_idx]
    y_proba_main, model_classes_main = fit_predict_proba(X_train_all, X_test_all, y_train)
    y_pred_main = np.array(class_order)[np.argmax(align_proba_to_order(y_proba_main, model_classes_main, class_order), axis=1)]
    acc  = accuracy_score(y.iloc[test_idx], y_pred_main)
    prec = precision_score(y.iloc[test_idx], y_pred_main, average='macro', zero_division=0)
    rec  = recall_score(y.iloc[test_idx], y_pred_main, average='macro', zero_division=0)
    f1   = f1_score(y.iloc[test_idx], y_pred_main, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(
            y.iloc[test_idx],
            align_proba_to_order(y_proba_main, model_classes_main, class_order),
            multi_class="ovr", average="macro", labels=class_order
        )
    except Exception as e:
        auc = np.nan
        print(f"  [Warning] Multiclass ROC-AUC failed for this fold: {e}")

    print(
        f"  [Multiclass (BAG+SEL)] Acc: {acc:.4f} | Prec(macro): {prec:.4f} | "
        f"Rec(macro): {rec:.4f} | F1(macro): {f1:.4f} | AUC(macro-ovr): {auc:.4f}"
    )

    fold_metrics.append({
        "fold": fold_idx, "accuracy": acc, "precision_macro": prec,
        "recall_macro": rec, "f1_macro": f1, "auc_macro_ovr": auc
    })

    if PLOT_MULTICLASS_CM:
        cm_multi = confusion_matrix(y.iloc[test_idx], y_pred_main, labels=class_order)
        fig_m = plot_percent_cm_with_single_colorbar(
            cm_multi, class_order,
            f"Confusion Matrix (Row-Normalized %) - Fold {fold_idx} (BAG+SAM)"
        )
        fig_list_multiclass.append(fig_m)

    for name, X_feat in feature_sets.items():
        X_train = X_feat.iloc[train_idx]
        X_test  = X_feat.iloc[test_idx]
        y_proba, model_classes = fit_predict_proba(X_train, X_test, y_train)
        y_proba_ordered = align_proba_to_order(y_proba, model_classes, class_order)
        oof_buckets[name].append((test_idx, y_proba_ordered))

    y_proba_ordered_main = align_proba_to_order(y_proba_main, model_classes_main, class_order)
    for (a, b) in pairs:
        mask = y.iloc[test_idx].isin([a, b]).values
        n_pair = int(mask.sum())
        if n_pair == 0:
            continue
        y_true_pair = y.iloc[test_idx][mask].map({a: 0, b: 1}).values
        pa = y_proba_ordered_main[mask, class_order.index(a)]
        pb = y_proba_ordered_main[mask, class_order.index(b)]
        y_pred_pair = (pb > pa).astype(int)

        acc2  = accuracy_score(y_true_pair, y_pred_pair)
        prec2 = precision_score(y_true_pair, y_pred_pair, zero_division=0)
        rec2  = recall_score(y_true_pair, y_pred_pair, zero_division=0)
        f12   = f1_score(y_true_pair, y_pred_pair, zero_division=0)
        try:
            auc2 = roc_auc_score(y_true_pair, pb)
        except Exception:
            auc2 = np.nan

        print(
            f"  [Pair {a} vs {b} | pos={b} | BAG+SAM] n={n_pair:3d} | "
            f"Acc: {acc2:.4f} | Prec: {prec2:.4f} | Rec: {rec2:.4f} | "
            f"F1: {f12:.4f} | AUC: {auc2:.4f}"
        )

        pairwise_metrics.append({
            "fold": fold_idx, "pair": f"{a} vs {b}", "pos": b, "n": n_pair,
            "accuracy": acc2, "precision": prec2, "recall": rec2, "f1": f12, "auc": auc2
        })

def stitch_oof(oof_list, n_samples, n_classes):
    """oof_list: [(test_idx, proba_ordered), ...] -> full_proba[n_samples, n_classes]"""
    full = np.zeros((n_samples, n_classes))
    for test_idx, proba in oof_list:
        full[test_idx] = proba
    return full

n_samples = len(y)
oof_proba = {
    name: stitch_oof(bucket, n_samples, len(class_order))
    for name, bucket in oof_buckets.items()
}

def plot_pairwise_roc_compare(y_series, proba_dict, class_order, pairs, title_prefix="Pairwise ROC"):
    """
    y_series: pd.Series of true labels (entire dataset)
    proba_dict: {config_name: full_proba[n, n_classes] ordered by class_order}
    """
    for (a, b) in pairs:
        mask = y_series.isin([a, b]).values
        if mask.sum() == 0:
            continue
        y_true = y_series[mask].map({a: 0, b: 1}).values

        plt.figure(figsize=(7, 6), dpi=300)
        for name, proba in proba_dict.items():
            pb = proba[mask, class_order.index(b)]
            try:
                fpr, tpr, _ = roc_curve(y_true, pb)
                auc_val = roc_auc_score(y_true, pb)
            except Exception:
                fpr = np.array([0.0, 1.0])
                tpr = np.array([0.0, 1.0])
                auc_val = np.nan
            plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc_val:.3f})")

        plt.plot([0, 1], [0, 1], linestyle='--', lw=1.2, color='black', label='Random Guess')

        plt.title(f"{title_prefix}: {a} vs {b} (pos={b})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

if PLOT_PAIRWISE_ROC:
    plot_pairwise_roc_compare(y, oof_proba, class_order, pairs, title_prefix="Pairwise ROC")

# ============ Multiclass summary ============
metrics_df = pd.DataFrame(fold_metrics)
print("\n==== 5-Fold Multiclass Summary (BAG+SEL) ====")
print(metrics_df.to_string(index=False, justify='center', col_space=14))

print("\n==== 5-Fold Multiclass Mean (± Std) ====")
for col in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro_ovr"]:
    mean_v = metrics_df[col].mean()
    std_v  = metrics_df[col].std()
    print(f"{col:>16}: {mean_v:.4f} ± {std_v:.4f}")

if len(pairwise_metrics) > 0:
    pair_df = pd.DataFrame(pairwise_metrics)
    print("\n==== 5-Fold Pairwise Classification Summary (per fold | BAG+SEL) ====")
    show_cols = ["fold", "pair", "pos", "n", "accuracy", "precision", "recall", "f1", "auc"]
    print(pair_df[show_cols].to_string(index=False, justify='center', col_space=12))

    print("\n==== 5-Fold Pairwise Mean (± Std | BAG+SEL) ====")
    agg = pair_df.groupby(["pair", "pos"]).agg(
        n_total=("n", "sum"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        prec_mean=("precision", "mean"), prec_std=("precision", "std"),
        rec_mean=("recall", "mean"), rec_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    ).reset_index()

    for _, r in agg.iterrows():
        print(
            f"{r['pair']:>10} (pos={r['pos']}) | n={int(r['n_total']):4d} | "
            f"Acc {r['acc_mean']:.4f}±{(0 if pd.isna(r['acc_std']) else r['acc_std']):.4f} | "
            f"P {r['prec_mean']:.4f}±{(0 if pd.isna(r['prec_std']) else r['prec_std']):.4f} | "
            f"R {r['rec_mean']:.4f}±{(0 if pd.isna(r['rec_std']) else r['rec_std']):.4f} | "
            f"F1 {r['f1_mean']:.4f}±{(0 if pd.isna(r['f1_std']) else r['f1_std']):.4f} | "
            f"AUC {r['auc_mean']:.4f}±{(0 if pd.isna(r['auc_std']) else r['auc_std']):.4f}"
        )
else:
    print("\n[Info] No pairwise classification results available due to fold splits or data distribution.")
