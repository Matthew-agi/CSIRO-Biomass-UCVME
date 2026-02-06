# now import timm fresh
import timm, sys
print("python:", sys.version)
print("timm:", timm.__version__)
print("timm file:", timm.__file__)
print("dinov3 matches:", timm.list_models("*dinov3*")[:50])
import os, gc, math, cv2, numpy as np, pandas as pd


def _load_env_file(env_path):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_load_env_file(os.path.join(_SCRIPT_DIR, ".env"))

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = os.path.join(os.getcwd(), ".cache", "matplotlib")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from timm.utils import ModelEmaV2
from sklearn.model_selection import KFold, StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DATA_DIR = os.path.join(_REPO_DIR, "data")

class CFG:
    CREATE_SUBMISSION = True
    USE_TQDM        = False
    PRETRAINED_DIR  = None
    PRETRAINED      = True
    BASE_PATH       = os.environ.get(
        "CSIRO_BIOMASS_DIR",
        _LOCAL_DATA_DIR if os.path.isdir(_LOCAL_DATA_DIR) else "/data",
    )
    SEED            = 82947501
    FOLDS_TO_TRAIN   = [0,1,2,3,4]
    TRAIN_CSV       = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test')
    TEST_CSV = os.path.join(BASE_PATH, 'test.csv')
    SUBMISSION_DIR  = './'
    MODEL_DIR = os.environ.get("CSIRO_MODEL_DIR", os.path.join(os.getcwd(), "yolo2"))
    N_FOLDS         = 5

    MODEL_NAME      = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    BACKBONE_PATH   = None
    DINO_GRAD_CHECKPOINTING = True

    IMG_SIZE        = 512

    VAL_TTA_TIMES   = 1
    TTA_STEPS       = 1
    
    TRAIN_TFM       = "none" #"none", "aug"
    TRAIN_TFMS      = ["aug"]

    CUTMIX_MODE     = "none"  # "none", "random", "similar"
    CUTMIX_MODES    = ["similar"]
    CUTMIX_PROB     = 0.2
    CUTMIX_ALPHA    = 1.0
    CUTMIX_SIMILAR_TOP_PCT = 0.2
    CUTMIX_MATCH_SIMILAR = True
    CUTMIX_EMBED_BATCH_SIZE = 8

    USE_VISTAMILK  = False
    VISTAMILK_DIR  = os.environ.get("VISTAMILK_DIR", os.path.join(_REPO_DIR, "vistamilk"))
    VISTAMILK_USE_VAL = True
    VISTAMILK_INCLUDE_CAMERA = True
    VISTAMILK_INCLUDE_PHONE = True
    VISTAMILK_PLOT_M2 = 0.25
    VISTAMILK_HA_M2 = 10000.0
    VISTAMILK_KG_TO_G = 1000.0
    VISTAMILK_MASS_SCALE = 1.0
    VISTAMILK_SAMPLE_WEIGHT = 0.5
    VISTAMILK_TOTAL_TARGET = "gdm"  # "gdm" or "total"

    WEIGHTING_MODE = "none"  # "none", "sampler", "loss", "sampler+loss"
    WEIGHTING_DOMAIN_ALPHA = 0.5
    WEIGHTING_PHEN_ALPHA = 0.5
    WEIGHTING_DOMAIN_CLIP = (0.5, 2.0)
    WEIGHTING_PHEN_CLIP = (0.5, 2.0)
    WEIGHTING_COMBINED_CLIP = (0.25, 4.0)
    WEIGHTING_PHENO_STRAT = "kmeans"  # "kmeans" or "bins"
    WEIGHTING_PHENO_K = 4
    WEIGHTING_PHENO_BINS = 5
    WEIGHTING_SAMPLER_REPLACEMENT = True

    USE_GROUP_DRO = False
    GROUP_DRO_GROUP = "dom"  # "dro" or "dom"
    GROUP_DRO_ETA = 0.005
    GROUP_DRO_EMA = 0.90
    REPORT_GROUP_METRICS = True
    GROUP_METRIC_GROUP = "dro"  # "dro" or "dom"
    GROUP_METRIC_MIN_SAMPLES = 1
    SAVE_METRIC = "robust"  # "global", "avg", "robust"
    SAVE_SCORE_WORST = 0.7
    SAVE_SCORE_GLOBAL = 0.3

    PREDICT_TARGETS = True
    PREDICT_DATE = True
    PREDICT_STATE = False
    PREDICT_SPECIES = False
    PREDICT_NDVI = True
    PREDICT_HEIGHT = True
    DATE_MODE = "dayofyear_sincos"  # "ordinal", "dayofyear", "month", "dayofyear_sincos"
    DATE_SCALE = 1.0
    SPECIES_MULTI_LABEL = True
    SPECIES_SPLIT_DELIM = "_"
    SPECIES_LOWER = False
    LOSS_WEIGHT_TARGETS = 1.0
    LOSS_WEIGHT_DATE = 1.0
    LOSS_WEIGHT_STATE = 1.0
    LOSS_WEIGHT_SPECIES = 1.0
    LOSS_WEIGHT_NDVI = 1.0
    LOSS_WEIGHT_HEIGHT = 1.0

    USE_HETERO = True
    HETERO_MIN_LOGVAR = -10.0
    HETERO_MAX_LOGVAR = 4.0
    HETERO_EPS = 1e-6
    PRINT_TRAIN_LOSS_DIAG = True
    DIAG_NONFINITE = True

    USE_UCVME = True
    UCVME_MC_SAMPLES = 5
    UCVME_ULB_WEIGHT = 2.0
    UCVME_ENABLE_AFTER_PATIENCE = True
    UCVME_GRAD_ACC_MULT = 1
    UCVME_BACKBONE_DROP_RATE = 0.05
    UCVME_BACKBONE_DROP_PATH_RATE = 0.10
    UCVME_USE_VISTAMILK = True
    UCVME_INCLUDE_META = True
    UCVME_INCLUDE_CLOVER = True
    UCVME_SHARED_BACKBONE = False
    UCVME_VISTAMILK_ULB_MODE = "corner_pair"  # "split" or "corner_pair"
    UCVME_ULB_BASE_CROP = (1500, 1500)  # (width, height)
    UCVME_ULB_CORNER_CROP = (1000, 900)  # (width, height)
    UCVME_EXTRA_UNLABELED_DIRS = []
    UCVME_EXTRA_UNLABELED_MODES = None  # optional list aligned with EXTRA_UNLABELED_DIRS
    UCVME_EXTRA_UNLABELED_MODE = "split"

    UCVME_USE_VEGANN = True
    UCVME_VEGANN_DIR = os.environ.get("VEGANN_DIR", os.path.join(_REPO_DIR, "data", "VegAnn", "VegAnn_dataset"))
    UCVME_VEGANN_CSV = os.path.join(UCVME_VEGANN_DIR, "VegAnn_dataset.csv")
    UCVME_VEGANN_IMG_DIR = os.path.join(UCVME_VEGANN_DIR, "images")
    UCVME_VEGANN_SPECIES = ["Mix", "Wheat", "Grassland", "Alfalfa", "Barley"] #"Mix", "Wheat", "Grassland", "Alfalfa", "Barley"
    UCVME_VEGANN_ORIENTATIONS = ["Nadir"]
    UCVME_VEGANN_SYSTEMS = None  # optional list of allowed systems
    UCVME_VEGANN_SPLIT_COL = None  # e.g., "TVT-split1"
    UCVME_VEGANN_SPLITS = None  # e.g., ["Training", "Validation"]
    UCVME_VEGANN_ULB_MODE = "same"

    USE_KERNEL_REG = False
    KERNEL_REG_MODE = "mlp"  # "blend", "residual", or "mlp"
    KERNEL_REG_TOPK = 50
    KERNEL_REG_TAU = 0.07
    KERNEL_REG_EMBED_BATCH_SIZE = 8
    KERNEL_REG_PRED_BATCH_SIZE = 64
    KERNEL_REG_EXCLUDE_SELF = True
    KERNEL_REG_ALPHA_PER_TARGET = True
    KERNEL_REG_ALPHA_INIT = 0.5

    WRITE_FOLD_SPLITS = True
    FOLD_SPLITS_CSV = os.path.join(_REPO_DIR, "fold_splits.csv")
    USE_EXTERNAL_FOLD_SPLITS = True
    EXTERNAL_FOLD_SPLITS_CSV = os.path.join(_REPO_DIR, "new_5fold.csv")
    EXTERNAL_FOLD_SPLITS_FOLD_COL = "fold"
    EXTERNAL_FOLD_SPLITS_IMAGE_COL = "image_id"
    EXTERNAL_FOLD_SPLITS_SAMPLE_COL = "sample_id"
    EXTERNAL_FOLD_SPLITS_PATH_COL = "image_path"
    EXTERNAL_FOLD_SPLITS_STRICT = True
    VAL_SPLIT_MODE = "stratified_group"  # "stratified_group", "group", "stratified", "kfold"
    VAL_SPLIT_GROUP_COL = "Sampling_Date"
    VAL_SPLIT_STRATIFY_COL = "State"

    USE_BAD_MARKS_FILTER = True
    BAD_MARKS_CSV = os.environ.get("BAD_MARKS_CSV", os.path.join(_REPO_DIR, "cluster_bad_marks.csv"))
    BAD_MARKS_COLUMN = "bad"
    BAD_MARKS_SPLIT = "train"
    EVAL_BAD_MARKS = True
    BAD_MARKS_PRINT_LIMIT = 0
    BAD_MARKS_PRINT_VAL = False
    
    BATCH_SIZE      = 1
    GRAD_ACC        = 4
    NUM_WORKERS     = 4
    EPOCHS          = 100
    FREEZE_EPOCHS   = 100
    WARMUP_EPOCHS   = 3
    LR_REST         = 1e-3
    LR_BACKBONE     = 5e-4
    WD              = 1e-2
    EMA_DECAY       = 0.9
    CLIP_GRAD_NORM  = 1.0
    PATIENCE        = 5
    TARGET_COLS     = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS    = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    R2_WEIGHTS      = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    R2_EPS          = 1e-6
    LOSS_WEIGHTS    = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(CFG.MODEL_DIR, exist_ok=True)
print(f'Device : {CFG.DEVICE}')
print(f'Backbone: {CFG.MODEL_NAME} | Input: {CFG.IMG_SIZE}')
print(f'Freeze Epochs: {CFG.FREEZE_EPOCHS} | Warmup: {CFG.WARMUP_EPOCHS}')
print(f'EMA Decay: {CFG.EMA_DECAY} | Grad Acc: {CFG.GRAD_ACC}')
# Metrics
def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    eps = float(getattr(CFG, "R2_EPS", 1e-6))
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]; yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > eps else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted, r2_scores

def weighted_r2_score_global(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    eps = float(getattr(CFG, "R2_EPS", 1e-6))
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    w = np.tile(weights, y_true.shape[0])
    mean_w = np.sum(w * flat_true) / np.sum(w)
    ss_res = np.sum(w * (flat_true - flat_pred) ** 2)
    ss_tot = np.sum(w * (flat_true - mean_w) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > eps else 0.0
    avg_r2, per_r2 = weighted_r2_score(y_true, y_pred)
    return global_r2, avg_r2, per_r2

def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]
        
        print(f'\nTarget: {target}')
        print(f'{"Index":<6} | {"Image Path":<40} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}')
        print('-' * 90)
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')
def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]
        
        print(f'\nTarget: {target}')
        header = f'{"Index":<6} | {"Image Path":<40} | {"State":<6} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}'
        print(header)
        print('-' * len(header))
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            state = val_df.iloc[idx]['State'] if 'State' in val_df.columns else 'NA'
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {str(state):<6} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')
def compare_train_val(tr_df, val_df, targets, show_plots=True):
    """Quick comparison of target distributions and metadata between train and val splits."""
    print("\n--- Train / Val Comparison ---")

    for t in targets:
        tr = tr_df.get(t, pd.Series(dtype=float)).dropna()
        val = val_df.get(t, pd.Series(dtype=float)).dropna()
        print(f"\nTarget: {t}")
        print(f"  Train: n={len(tr)} mean={tr.mean():.3f} std={tr.std():.3f} min={tr.min():.3f} max={tr.max():.3f}")
        print(f"  Val  : n={len(val)} mean={val.mean():.3f} std={val.std():.3f} min={val.min():.3f} max={val.max():.3f}")
        if show_plots:
            try:
                plt.figure(figsize=(6, 3))
                sns.kdeplot(tr, label='train', fill=True)
                sns.kdeplot(val, label='val', fill=True)
                plt.legend()
                plt.title(f'Distribution: {t}')
                plt.show()
            except Exception as e:
                print('  Could not plot distributions for', t, '-', e)

    # Compare Sampling_Date and State if present
    if 'Sampling_Date' in tr_df.columns:
        try:
            tr_dates = pd.to_datetime(tr_df['Sampling_Date'], errors='coerce')
            val_dates = pd.to_datetime(val_df['Sampling_Date'], errors='coerce')
            print("\nSampling_Date range:")
            print(f"  Train: {tr_dates.min()} -> {tr_dates.max()} (missing {tr_dates.isna().sum()})")
            print(f"  Val  : {val_dates.min()} -> {val_dates.max()} (missing {val_dates.isna().sum()})")
        except Exception as e:
            print('  Could not parse Sampling_Date:', e)
    if 'State' in tr_df.columns:
        print("\nState distribution (train vs val):")
        tr_state = tr_df['State'].value_counts(normalize=True)
        val_state = val_df['State'].value_counts(normalize=True)
        state_df = pd.concat([tr_state, val_state], axis=1, keys=['train', 'val']).fillna(0)

        print(state_df)

    
def _split_species_tokens(value):
    if pd.isna(value):
        return []
    text = str(value)
    if CFG.SPECIES_LOWER:
        text = text.lower()
    parts = [p for p in text.split(CFG.SPECIES_SPLIT_DELIM) if p]
    return parts

def build_meta_info(df: pd.DataFrame) -> dict:
    meta = {
        "use_date": False,
        "use_state": False,
        "use_species": False,
        "use_ndvi": False,
        "use_height": False,
        "state_to_idx": {},
        "species_to_idx": {},
        "num_states": 0,
        "num_species": 0,
        "species_multi_label": bool(CFG.SPECIES_MULTI_LABEL),
    }

    def _warn(msg):
        print(f"Warning: {msg}")

    if CFG.PREDICT_DATE:
        if "Sampling_Date" in df.columns:
            meta["use_date"] = True
        else:
            _warn("Sampling_Date missing; disabling date prediction.")

    if CFG.PREDICT_STATE:
        if "State" in df.columns:
            states = sorted(df["State"].dropna().astype(str).unique().tolist())
            if states:
                meta["use_state"] = True
                meta["state_to_idx"] = {s: i for i, s in enumerate(states)}
                meta["num_states"] = len(states)
                print(f"State classes: {meta['num_states']}")
            else:
                _warn("No State values found; disabling state prediction.")
        else:
            _warn("State missing; disabling state prediction.")

    if CFG.PREDICT_SPECIES:
        if "Species" in df.columns:
            tokens = set()
            for val in df["Species"].dropna().astype(str).tolist():
                tokens.update(_split_species_tokens(val))
            species = sorted(tokens)
            if species:
                meta["use_species"] = True
                meta["species_to_idx"] = {s: i for i, s in enumerate(species)}
                meta["num_species"] = len(species)
                print(f"Species classes: {meta['num_species']}")
            else:
                _warn("No Species tokens found; disabling species prediction.")
        else:
            _warn("Species missing; disabling species prediction.")

    if CFG.PREDICT_NDVI:
        if "Pre_GSHH_NDVI" in df.columns:
            meta["use_ndvi"] = True
        else:
            _warn("Pre_GSHH_NDVI missing; disabling NDVI prediction.")

    if CFG.PREDICT_HEIGHT:
        if "Height_Ave_cm" in df.columns:
            meta["use_height"] = True
        else:
            _warn("Height_Ave_cm missing; disabling height prediction.")

    meta["enabled"] = any([
        meta["use_date"],
        meta["use_state"],
        meta["use_species"],
        meta["use_ndvi"],
        meta["use_height"],
    ])
    return meta

def _parse_dates(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce")
    if CFG.DATE_MODE == "ordinal":
        values = dt.map(lambda d: d.toordinal() if pd.notna(d) else np.nan)
    elif CFG.DATE_MODE == "dayofyear":
        values = dt.dt.dayofyear
    elif CFG.DATE_MODE == "month":
        values = dt.dt.month
    elif CFG.DATE_MODE == "dayofyear_sincos":
        doy = dt.dt.dayofyear.astype(float)
        angle = 2.0 * math.pi * (doy - 1.0) / 365.25
        sin = np.sin(angle)
        cos = np.cos(angle)
        values = np.stack([sin, cos], axis=1)
    else:
        raise ValueError(f"Unsupported DATE_MODE: {CFG.DATE_MODE}")
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    values = values.astype(np.float32) * float(CFG.DATE_SCALE)
    return values

def build_meta_arrays(df: pd.DataFrame, meta_info: dict) -> dict:
    meta = {}
    n = len(df)

    if meta_info.get("use_date"):
        if "Sampling_Date" in df.columns:
            date_vals = _parse_dates(df["Sampling_Date"])
        else:
            date_vals = np.full((n,), np.nan, dtype=np.float32)
        date_mask = (~np.isnan(date_vals)).astype(np.float32)
        date_vals = np.nan_to_num(date_vals, nan=0.0).astype(np.float32)
        if date_vals.ndim == 1:
            date_vals = date_vals.reshape(-1, 1)
        if date_mask.ndim == 1:
            date_mask = date_mask.reshape(-1, 1)
        meta["date"] = date_vals
        meta["date_mask"] = date_mask

    if meta_info.get("use_state"):
        if "State" in df.columns:
            states = df["State"].astype(str).fillna("")
            idx = np.array([meta_info["state_to_idx"].get(s, -1) for s in states], dtype=np.int64)
        else:
            idx = np.full((n,), -1, dtype=np.int64)
        state_mask = (idx >= 0).astype(np.float32)
        idx = np.where(idx < 0, 0, idx)
        meta["state"] = idx
        meta["state_mask"] = state_mask

    if meta_info.get("use_species"):
        num_species = int(meta_info.get("num_species", 0))
        if meta_info.get("species_multi_label", True):
            species_vec = np.zeros((n, num_species), dtype=np.float32)
            species_mask = np.zeros((n, 1), dtype=np.float32)
            if "Species" in df.columns:
                for i, val in enumerate(df["Species"].astype(str).fillna("").tolist()):
                    tokens = _split_species_tokens(val)
                    if not tokens:
                        continue
                    for tok in tokens:
                        idx = meta_info["species_to_idx"].get(tok)
                        if idx is not None:
                            species_vec[i, idx] = 1.0
                    species_mask[i, 0] = 1.0
            meta["species"] = species_vec
            meta["species_mask"] = species_mask
        else:
            if "Species" in df.columns:
                values = df["Species"].astype(str).fillna("")
                idx = np.array([meta_info["species_to_idx"].get(v, -1) for v in values], dtype=np.int64)
            else:
                idx = np.full((n,), -1, dtype=np.int64)
            species_mask = (idx >= 0).astype(np.float32)
            idx = np.where(idx < 0, 0, idx)
            meta["species"] = idx
            meta["species_mask"] = species_mask

    if meta_info.get("use_ndvi"):
        if "Pre_GSHH_NDVI" in df.columns:
            ndvi_vals = pd.to_numeric(df["Pre_GSHH_NDVI"], errors="coerce").to_numpy(dtype=np.float32)
        else:
            ndvi_vals = np.full((n,), np.nan, dtype=np.float32)
        ndvi_mask = (~np.isnan(ndvi_vals)).astype(np.float32)
        ndvi_vals = np.nan_to_num(ndvi_vals, nan=0.0).astype(np.float32)
        meta["ndvi"] = ndvi_vals.reshape(-1, 1)
        meta["ndvi_mask"] = ndvi_mask.reshape(-1, 1)

    if meta_info.get("use_height"):
        if "Height_Ave_cm" in df.columns:
            height_vals = pd.to_numeric(df["Height_Ave_cm"], errors="coerce").to_numpy(dtype=np.float32)
        else:
            height_vals = np.full((n,), np.nan, dtype=np.float32)
        height_mask = (~np.isnan(height_vals)).astype(np.float32)
        height_vals = np.nan_to_num(height_vals, nan=0.0).astype(np.float32)
        meta["height"] = height_vals.reshape(-1, 1)
        meta["height_mask"] = height_mask.reshape(-1, 1)

    return meta

def _extract_state_month(df: pd.DataFrame):
    if "State" in df.columns:
        state = df["State"].astype(str).fillna("NA")
    else:
        state = pd.Series(["NA"] * len(df), index=df.index)
    if "Sampling_Date" in df.columns:
        dt = pd.to_datetime(df["Sampling_Date"], errors="coerce")
        month = dt.dt.month.fillna(0).astype(int)
    else:
        month = pd.Series([0] * len(df), index=df.index)
    return state.to_numpy(), month.to_numpy()

def _extract_phenology_vals(df: pd.DataFrame):
    if "Pre_GSHH_NDVI" in df.columns:
        ndvi = pd.to_numeric(df["Pre_GSHH_NDVI"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        ndvi = np.full((len(df),), np.nan, dtype=np.float32)
    if "Height_Ave_cm" in df.columns:
        height = pd.to_numeric(df["Height_Ave_cm"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        height = np.full((len(df),), np.nan, dtype=np.float32)
    return ndvi, height

def _assign_bins(values, edges):
    if edges.size <= 1:
        return np.zeros_like(values, dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)

def _compute_phenology_strata(tr_df: pd.DataFrame, val_df: pd.DataFrame):
    ndvi_tr, height_tr = _extract_phenology_vals(tr_df)
    ndvi_val, height_val = _extract_phenology_vals(val_df)
    valid_tr = np.isfinite(ndvi_tr) & np.isfinite(height_tr)
    valid_val = np.isfinite(ndvi_val) & np.isfinite(height_val)

    phen_tr = np.full((len(tr_df),), -1, dtype=np.int64)
    phen_val = np.full((len(val_df),), -1, dtype=np.int64)

    if not valid_tr.any():
        print("Phenology stratification skipped: no valid NDVI/Height in training.")
        return phen_tr, phen_val

    mode = str(CFG.WEIGHTING_PHENO_STRAT).lower()
    if mode == "kmeans":
        k = int(CFG.WEIGHTING_PHENO_K)
        k = max(1, min(k, int(valid_tr.sum())))
        X_tr = np.stack([ndvi_tr[valid_tr], height_tr[valid_tr]], axis=1)
        kmeans = KMeans(n_clusters=k, random_state=CFG.SEED, n_init=10)
        kmeans.fit(X_tr)
        phen_tr[valid_tr] = kmeans.predict(X_tr)
        if valid_val.any():
            X_val = np.stack([ndvi_val[valid_val], height_val[valid_val]], axis=1)
            phen_val[valid_val] = kmeans.predict(X_val)
        return phen_tr, phen_val
    if mode == "bins":
        bins = int(CFG.WEIGHTING_PHENO_BINS)
        bins = max(1, bins)
        ndvi_edges = np.nanquantile(ndvi_tr[valid_tr], np.linspace(0, 1, bins + 1))
        height_edges = np.nanquantile(height_tr[valid_tr], np.linspace(0, 1, bins + 1))
        ndvi_edges = np.unique(ndvi_edges)
        height_edges = np.unique(height_edges)
        ndvi_bin_tr = _assign_bins(ndvi_tr, ndvi_edges)
        height_bin_tr = _assign_bins(height_tr, height_edges)
        ndvi_bin_val = _assign_bins(ndvi_val, ndvi_edges)
        height_bin_val = _assign_bins(height_val, height_edges)
        phen_tr = ndvi_bin_tr * max(1, height_edges.size - 1) + height_bin_tr
        phen_val = ndvi_bin_val * max(1, height_edges.size - 1) + height_bin_val
        phen_tr[~valid_tr] = -1
        phen_val[~valid_val] = -1
        return phen_tr.astype(np.int64), phen_val.astype(np.int64)
    raise ValueError(f"Unsupported WEIGHTING_PHENO_STRAT: {CFG.WEIGHTING_PHENO_STRAT}")

def _compute_inv_freq_weights(keys, alpha, clip_min, clip_max):
    if len(keys) == 0:
        return np.zeros((0,), dtype=np.float32)
    counts = pd.Series(keys).value_counts()
    mean_count = counts.mean() if len(counts) > 0 else 1.0
    denom = np.array([counts.get(k, mean_count) for k in keys], dtype=np.float32)
    denom = np.maximum(denom, 1.0)
    weights = (mean_count / denom) ** float(alpha)
    weights = np.clip(weights, float(clip_min), float(clip_max))
    return weights.astype(np.float32)

def build_group_info(tr_df: pd.DataFrame, val_df: pd.DataFrame):
    state_tr, month_tr = _extract_state_month(tr_df)
    state_val, month_val = _extract_state_month(val_df)

    dom_key_tr = np.array([f"{s}_{m}" for s, m in zip(state_tr, month_tr)], dtype=object)
    dom_key_val = np.array([f"{s}_{m}" for s, m in zip(state_val, month_val)], dtype=object)

    phen_tr, phen_val = _compute_phenology_strata(tr_df, val_df)
    phen_key_tr = np.array(["missing" if p < 0 else str(int(p)) for p in phen_tr], dtype=object)
    phen_key_val = np.array(["missing" if p < 0 else str(int(p)) for p in phen_val], dtype=object)

    dro_key_tr = np.array([f"{d}_{p}" for d, p in zip(dom_key_tr, phen_key_tr)], dtype=object)
    dro_key_val = np.array([f"{d}_{p}" for d, p in zip(dom_key_val, phen_key_val)], dtype=object)

    w_dom = _compute_inv_freq_weights(
        dom_key_tr,
        CFG.WEIGHTING_DOMAIN_ALPHA,
        CFG.WEIGHTING_DOMAIN_CLIP[0],
        CFG.WEIGHTING_DOMAIN_CLIP[1],
    )
    w_phen = _compute_inv_freq_weights(
        phen_key_tr,
        CFG.WEIGHTING_PHEN_ALPHA,
        CFG.WEIGHTING_PHEN_CLIP[0],
        CFG.WEIGHTING_PHEN_CLIP[1],
    )
    weights = w_dom * w_phen
    weights = np.clip(weights, CFG.WEIGHTING_COMBINED_CLIP[0], CFG.WEIGHTING_COMBINED_CLIP[1]).astype(np.float32)

    dom_map = {k: i for i, k in enumerate(sorted(set(dom_key_tr.tolist())))}
    dro_map = {k: i for i, k in enumerate(sorted(set(dro_key_tr.tolist())))}

    train_dom_ids = np.array([dom_map.get(k, -1) for k in dom_key_tr], dtype=np.int64)
    val_dom_ids = np.array([dom_map.get(k, -1) for k in dom_key_val], dtype=np.int64)
    train_dro_ids = np.array([dro_map.get(k, -1) for k in dro_key_tr], dtype=np.int64)
    val_dro_ids = np.array([dro_map.get(k, -1) for k in dro_key_val], dtype=np.int64)

    return {
        "weights": weights,
        "dom_keys": dom_key_tr,
        "dro_keys": dro_key_tr,
        "train_dom_ids": train_dom_ids,
        "val_dom_ids": val_dom_ids,
        "train_dro_ids": train_dro_ids,
        "val_dro_ids": val_dro_ids,
        "num_dom_groups": len(dom_map),
        "num_dro_groups": len(dro_map),
    }

def compute_group_r2_metrics(y_true, y_pred, group_ids, min_samples=1):
    group_ids = np.asarray(group_ids)
    valid = group_ids >= 0
    if not np.any(valid):
        return None, None, 0
    scores = []
    for gid in np.unique(group_ids[valid]):
        idx = group_ids == gid
        if idx.sum() < int(min_samples):
            continue
        global_r2, _, _ = weighted_r2_score_global(y_true[idx], y_pred[idx])
        scores.append(global_r2)
    if not scores:
        return None, None, 0
    return float(np.mean(scores)), float(np.min(scores)), len(scores)

def biomass_loss(outputs, labels, w=None, mask=None, sample_weight=None):
    total, gdm, green, clover, dead = outputs
    huber = nn.SmoothL1Loss(beta=5.0, reduction="none")

    def _as_vec(x):
        return x.view(-1)

    l_green  = huber(_as_vec(green),  labels[:, 0])
    l_dead   = huber(_as_vec(dead),   labels[:, 1])
    l_clover = huber(_as_vec(clover), labels[:, 2])
    l_gdm    = huber(_as_vec(gdm),    labels[:, 3])
    l_total  = huber(_as_vec(total),  labels[:, 4])

    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total], dim=1)

    if mask is None:
        mask_t = torch.ones_like(losses)
    else:
        mask_t = mask.to(device=losses.device, dtype=losses.dtype)
        if mask_t.ndim == 1:
            mask_t = mask_t.view(1, -1).expand_as(losses)

    if w is None:
        w_t = torch.ones((1, losses.shape[1]), device=losses.device, dtype=losses.dtype)
    else:
        w_t = torch.as_tensor(w, device=losses.device, dtype=losses.dtype).view(1, -1)

    if sample_weight is None:
        sw_t = torch.ones((losses.shape[0], 1), device=losses.device, dtype=losses.dtype)
    else:
        sw_t = sample_weight.to(device=losses.device, dtype=losses.dtype)
        if sw_t.ndim == 0:
            sw_t = sw_t.view(1, 1)
        elif sw_t.ndim == 1:
            sw_t = sw_t.view(-1, 1)

    weighted = losses * mask_t * w_t * sw_t
    denom = (mask_t * w_t * sw_t).sum()
    if denom <= 0:
        return losses.mean()
    return weighted.sum() / denom

def biomass_loss_per_sample(outputs, labels, w=None, mask=None, sample_weight=None):
    total, gdm, green, clover, dead = outputs
    huber = nn.SmoothL1Loss(beta=5.0, reduction="none")

    def _as_vec(x):
        return x.view(-1)

    l_green  = huber(_as_vec(green),  labels[:, 0])
    l_dead   = huber(_as_vec(dead),   labels[:, 1])
    l_clover = huber(_as_vec(clover), labels[:, 2])
    l_gdm    = huber(_as_vec(gdm),    labels[:, 3])
    l_total  = huber(_as_vec(total),  labels[:, 4])

    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total], dim=1)

    if mask is None:
        mask_t = torch.ones_like(losses)
    else:
        mask_t = mask.to(device=losses.device, dtype=losses.dtype)
        if mask_t.ndim == 1:
            mask_t = mask_t.view(1, -1).expand_as(losses)

    if w is None:
        w_t = torch.ones((1, losses.shape[1]), device=losses.device, dtype=losses.dtype)
    else:
        w_t = torch.as_tensor(w, device=losses.device, dtype=losses.dtype).view(1, -1)

    weighted = losses * mask_t * w_t
    denom = (mask_t * w_t).sum(dim=1)
    per = torch.where(denom > 0, weighted.sum(dim=1) / denom, torch.zeros_like(denom))

    if sample_weight is not None:
        sw_t = sample_weight.to(device=losses.device, dtype=losses.dtype)
        if sw_t.ndim == 0:
            sw_t = sw_t.view(1)
        elif sw_t.ndim > 1:
            sw_t = sw_t.view(-1)
        per = per * sw_t

    return per

def hetero_nll_loss(pred_pack, logvar_pack, labels, w=None, mask=None, sample_weight=None):
    if logvar_pack is None:
        return None
    diff = pred_pack - labels
    logvar = logvar_pack.to(device=pred_pack.device, dtype=pred_pack.dtype)
    loss = 0.5 * (torch.exp(-logvar) * diff * diff + logvar)

    if mask is None:
        mask_t = torch.ones_like(loss)
    else:
        mask_t = mask.to(device=loss.device, dtype=loss.dtype)
        if mask_t.ndim == 1:
            mask_t = mask_t.view(1, -1).expand_as(loss)
    if w is None:
        w_t = torch.ones((1, loss.shape[1]), device=loss.device, dtype=loss.dtype)
    else:
        w_t = torch.as_tensor(w, device=loss.device, dtype=loss.dtype).view(1, -1)

    if sample_weight is None:
        sw_t = torch.ones((loss.shape[0], 1), device=loss.device, dtype=loss.dtype)
    else:
        sw_t = sample_weight.to(device=loss.device, dtype=loss.dtype)
        if sw_t.ndim == 0:
            sw_t = sw_t.view(1, 1)
        elif sw_t.ndim == 1:
            sw_t = sw_t.view(-1, 1)

    weighted = loss * mask_t * w_t * sw_t
    denom = (mask_t * w_t * sw_t).sum()
    if denom <= 0:
        return loss.mean()
    return weighted.sum() / denom

def hetero_nll_loss_per_sample(pred_pack, logvar_pack, labels, w=None, mask=None, sample_weight=None):
    if logvar_pack is None:
        return None
    diff = pred_pack - labels
    logvar = logvar_pack.to(device=pred_pack.device, dtype=pred_pack.dtype)
    loss = 0.5 * (torch.exp(-logvar) * diff * diff + logvar)

    if mask is None:
        mask_t = torch.ones_like(loss)
    else:
        mask_t = mask.to(device=loss.device, dtype=loss.dtype)
        if mask_t.ndim == 1:
            mask_t = mask_t.view(1, -1).expand_as(loss)
    if w is None:
        w_t = torch.ones((1, loss.shape[1]), device=loss.device, dtype=loss.dtype)
    else:
        w_t = torch.as_tensor(w, device=loss.device, dtype=loss.dtype).view(1, -1)

    weighted = loss * mask_t * w_t
    denom = (mask_t * w_t).sum(dim=1)
    per = torch.where(denom > 0, weighted.sum(dim=1) / denom, torch.zeros_like(denom))

    if sample_weight is not None:
        sw_t = sample_weight.to(device=loss.device, dtype=loss.dtype)
        if sw_t.ndim == 0:
            sw_t = sw_t.view(1)
        elif sw_t.ndim > 1:
            sw_t = sw_t.view(-1)
        per = per * sw_t

    return per

class GroupDRO:
    def __init__(self, num_groups, eta=0.01, ema=0.0, device=None):
        self.num_groups = int(num_groups)
        self.eta = float(eta)
        self.ema = float(ema)
        if self.num_groups > 0:
            self.q = torch.ones(self.num_groups, device=device, dtype=torch.float32)
            self.q = self.q / self.q.sum()
        else:
            self.q = None

    def compute_loss(self, per_sample_loss, group_ids):
        if self.num_groups <= 0 or self.q is None:
            return per_sample_loss.mean()
        if group_ids is None:
            return per_sample_loss.mean()
        gids = group_ids.view(-1).to(device=per_sample_loss.device, dtype=torch.long)
        valid = gids >= 0
        if not torch.any(valid):
            return per_sample_loss.mean()
        gids = gids[valid]
        losses = per_sample_loss[valid]

        unique = torch.unique(gids)
        q_new = self.q.clone()
        group_losses = {}
        for gid in unique.tolist():
            mask = gids == gid
            gl = losses[mask].mean()
            group_losses[gid] = gl
            q_new[gid] = q_new[gid] * torch.exp(self.eta * gl.detach())

        q_new = q_new / q_new.sum()
        if self.ema > 0.0:
            q_new = self.ema * self.q + (1.0 - self.ema) * q_new
            q_new = q_new / q_new.sum()

        self.q = q_new.detach()

        dro_loss = 0.0
        for gid, gl in group_losses.items():
            dro_loss = dro_loss + q_new[gid].detach() * gl
        return dro_loss

# Transforms
def get_train_transforms(mode=None):
    preset = mode or getattr(CFG, "TRAIN_TFM", "aug")
    if preset == "none":
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], p=1.0)
    if preset == "aug":
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=(-10, 10), p=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], p=1.0)
    raise ValueError(f"Unknown train transform preset: {preset}")

def get_val_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0)

def get_embed_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0)

def get_tta_transforms(mode=0):
    # mode 0: original
    # mode 1: hflip
    # mode 2: vflip
    # mode 3: rotate90
    transforms_list = [
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    ]
    
    if mode == 1:
        transforms_list.append(A.HorizontalFlip(p=1.0))
    elif mode == 2:
        transforms_list.append(A.VerticalFlip(p=1.0))
    elif mode == 3:
        transforms_list.append(A.RandomRotate90(p=1.0)) # RandomRotate90 with p=1.0 rotates 90, 180, 270 randomly? 
        # Albumentations RandomRotate90 rotates by 90, 180, 270. 
        # Reference uses transforms.RandomRotation([90, 90]) which is exactly 90 degrees.
        # To match exactly 90 degrees in Albumentations, we might need Rotate(limit=(90,90), p=1.0)
        # But RandomRotate90 is standard TTA. Let's use Rotate(limit=(90,90)) to be precise if that's what reference does.
        # Reference: transforms.RandomRotation([90, 90]) -> rotates by exactly 90 degrees.
        transforms_list.append(A.Rotate(limit=(90, 90), p=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101))

    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list, p=1.0)

def _inpaint_orange_stamp(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img

def _ensure_min_size(img, min_w, min_h):
    h, w = img.shape[:2]
    if h >= min_h and w >= min_w:
        return img
    scale = max(float(min_h) / float(h), float(min_w) / float(w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def _center_crop(img, crop_w, crop_h):
    h, w = img.shape[:2]
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    return img[y0:y0 + crop_h, x0:x0 + crop_w]

def clean_image(img):
    # 1. Safe Crop (Remove artifacts at the bottom)
    h, w = img.shape[:2]
    # Cut bottom 10% where artifacts often appear
    img = img[0:int(h*0.90), :] 

    # 2. Inpaint Date Stamp (Remove orange text)
    return _inpaint_orange_stamp(img)

def _ulb_corner_pair(img):
    base_w, base_h = (int(x) for x in CFG.UCVME_ULB_BASE_CROP)
    crop_w, crop_h = (int(x) for x in CFG.UCVME_ULB_CORNER_CROP)
    if crop_w > base_w or crop_h > base_h:
        raise ValueError("UCVME_ULB_CORNER_CROP must fit inside UCVME_ULB_BASE_CROP.")
    img = _ensure_min_size(img, base_w, base_h)
    img = _center_crop(img, base_w, base_h)
    tl = img[0:crop_h, 0:crop_w]
    br = img[base_h - crop_h:base_h, base_w - crop_w:base_w]
    return tl, br

def _make_ulb_pair(img, mode):
    if mode == "corner_pair":
        img = _inpaint_orange_stamp(img)
        return _ulb_corner_pair(img)
    if mode == "split":
        img = clean_image(img)
        h, w, _ = img.shape
        mid = w // 2
        return img[:, :mid], img[:, mid:]
    if mode == "same":
        img = clean_image(img)
        return img, img
    raise ValueError(f"Unsupported unlabeled pair mode: {mode}")

def _rand_bbox(rng, width, height, lam):
    cut_rat = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)
    cx = int(rng.integers(0, width))
    cy = int(rng.integers(0, height))
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, width)
    bby2 = min(cy + cut_h // 2, height)
    area = max(1, (bbx2 - bbx1) * (bby2 - bby1))
    lam_adj = 1.0 - (area / float(width * height))
    return bbx1, bby1, bbx2, bby2, lam_adj

def _apply_cutmix(left, right, label, left2, right2, label2, rng, alpha):
    if alpha <= 0.0:
        return left, right, label
    lam = float(rng.beta(alpha, alpha))
    _, h, w = left.shape
    bbx1, bby1, bbx2, bby2, lam_adj = _rand_bbox(rng, w, h, lam)
    left = left.clone()
    right = right.clone()
    left[:, bby1:bby2, bbx1:bbx2] = left2[:, bby1:bby2, bbx1:bbx2]
    right[:, bby1:bby2, bbx1:bbx2] = right2[:, bby1:bby2, bbx1:bbx2]
    label = label * lam_adj + label2 * (1.0 - lam_adj)
    return left, right, label

class BiomassHalfEmbedDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
        self.paths = df['image_path'].values
        self.transform = get_embed_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        return idx, left, right

class BiomassDataset(Dataset):
    def __init__(
        self,
        df,
        transform,
        img_dir,
        cutmix_mode="none",
        cutmix_prob=0.0,
        cutmix_alpha=1.0,
        cutmix_similar_indices=None,
        seed=CFG.SEED,
        label_mask=None,
        return_mask=False,
        meta_info=None,
        return_meta=False,
        kernel_preds=None,
        kernel_mask=None,
        return_kernel=False,
        sample_weight=None,
        return_weight=False,
        group_ids=None,
        return_group=False,
    ):
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        self.paths = df['image_path'].values
        self.labels = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)
        self.cutmix_mode = cutmix_mode
        self.cutmix_prob = float(cutmix_prob)
        self.cutmix_alpha = float(cutmix_alpha)
        self.cutmix_similar_indices = (
            np.array(cutmix_similar_indices, dtype=np.int64)
            if cutmix_similar_indices is not None
            else None
        )
        self.cutmix_similar_set = (
            set(self.cutmix_similar_indices.tolist())
            if self.cutmix_similar_indices is not None
            else None
        )
        self.all_indices = np.arange(len(self.paths), dtype=np.int64)
        self.seed = int(seed)
        self._rng = None
        self.return_mask = bool(return_mask) or (label_mask is not None)
        if self.return_mask:
            if label_mask is None:
                self.label_mask = np.ones((len(self.paths), len(CFG.ALL_TARGET_COLS)), dtype=np.float32)
            else:
                self.label_mask = np.asarray(label_mask, dtype=np.float32)
        else:
            self.label_mask = None
        self.meta_info = meta_info or {}
        self.return_meta = bool(return_meta) and bool(self.meta_info.get("enabled"))
        if self.return_meta:
            self.meta_arrays = build_meta_arrays(df, self.meta_info)
        else:
            self.meta_arrays = None
        self.return_kernel = bool(return_kernel)
        if self.return_kernel:
            if kernel_preds is None:
                self.kernel_preds = np.zeros((len(self.paths), len(CFG.ALL_TARGET_COLS)), dtype=np.float32)
                if kernel_mask is None:
                    self.kernel_mask = np.zeros((len(self.paths), 1), dtype=np.float32)
                else:
                    self.kernel_mask = np.asarray(kernel_mask, dtype=np.float32)
            else:
                self.kernel_preds = np.asarray(kernel_preds, dtype=np.float32)
                if kernel_mask is None:
                    self.kernel_mask = np.ones((len(self.paths), 1), dtype=np.float32)
                else:
                    self.kernel_mask = np.asarray(kernel_mask, dtype=np.float32)
            if self.kernel_preds.shape[0] != len(self.paths):
                raise ValueError("kernel_preds length does not match dataset length.")
        else:
            self.kernel_preds = None
            self.kernel_mask = None
        self.return_weight = bool(return_weight)
        if self.return_weight:
            if sample_weight is None:
                self.sample_weight = np.ones((len(self.paths),), dtype=np.float32)
            else:
                self.sample_weight = np.asarray(sample_weight, dtype=np.float32)
                if self.sample_weight.shape[0] != len(self.paths):
                    raise ValueError("sample_weight length does not match dataset length.")
        else:
            self.sample_weight = None
        self.return_group = bool(return_group)
        if self.return_group:
            if group_ids is None:
                self.group_ids = np.full((len(self.paths),), -1, dtype=np.int64)
            else:
                self.group_ids = np.asarray(group_ids, dtype=np.int64)
                if self.group_ids.shape[0] != len(self.paths):
                    raise ValueError("group_ids length does not match dataset length.")
        else:
            self.group_ids = None

    def __len__(self):
        return len(self.df)

    def _pack_output(self, left, right, label, mask, meta, kpred, kmask, weight, group_id):
        out = [left, right, label]
        if self.return_mask:
            out.append(mask)
        if self.return_meta:
            out.append(meta)
        if self.return_kernel:
            out.extend([kpred, kmask])
        if self.return_weight:
            out.append(weight)
        if self.return_group:
            out.append(group_id)
        return tuple(out)

    def _get_rng(self):
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            self._rng = np.random.default_rng(self.seed + worker_id)
        return self._rng

    def _load_sample(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        label = torch.from_numpy(self.labels[idx])
        return left, right, label

    def __getitem__(self, idx):
        left, right, label = self._load_sample(idx)
        mask = None
        if self.return_mask:
            mask = torch.from_numpy(self.label_mask[idx])
        meta = None
        if self.return_meta:
            meta = {}
            if "date" in self.meta_arrays:
                meta["date"] = torch.from_numpy(self.meta_arrays["date"][idx])
                meta["date_mask"] = torch.from_numpy(self.meta_arrays["date_mask"][idx])
            if "state" in self.meta_arrays:
                meta["state"] = torch.tensor(self.meta_arrays["state"][idx], dtype=torch.long)
                meta["state_mask"] = torch.tensor(self.meta_arrays["state_mask"][idx], dtype=torch.float32)
            if "species" in self.meta_arrays:
                species_val = self.meta_arrays["species"][idx]
                if np.isscalar(species_val) or getattr(species_val, "shape", ()) == ():
                    meta["species"] = torch.tensor(species_val, dtype=torch.long)
                else:
                    meta["species"] = torch.from_numpy(species_val)
                mask_val = self.meta_arrays["species_mask"][idx]
                if np.isscalar(mask_val) or getattr(mask_val, "shape", ()) == ():
                    meta["species_mask"] = torch.tensor(mask_val, dtype=torch.float32)
                else:
                    meta["species_mask"] = torch.from_numpy(mask_val)
            if "ndvi" in self.meta_arrays:
                meta["ndvi"] = torch.from_numpy(self.meta_arrays["ndvi"][idx])
                meta["ndvi_mask"] = torch.from_numpy(self.meta_arrays["ndvi_mask"][idx])
            if "height" in self.meta_arrays:
                meta["height"] = torch.from_numpy(self.meta_arrays["height"][idx])
                meta["height_mask"] = torch.from_numpy(self.meta_arrays["height_mask"][idx])
        kpred = None
        kmask = None
        if self.return_kernel:
            kpred = torch.from_numpy(self.kernel_preds[idx])
            kmask = torch.from_numpy(self.kernel_mask[idx])
        weight = None
        if self.return_weight:
            weight = torch.tensor(self.sample_weight[idx], dtype=torch.float32)
        group_id = None
        if self.return_group:
            group_id = torch.tensor(self.group_ids[idx], dtype=torch.long)

        if self.cutmix_mode == "none" or self.cutmix_prob <= 0.0:
            return self._pack_output(left, right, label, mask, meta, kpred, kmask, weight, group_id)
        if self.cutmix_mode == "similar":
            if not self.cutmix_similar_set or idx not in self.cutmix_similar_set:
                return self._pack_output(left, right, label, mask, meta, kpred, kmask, weight, group_id)
            pool = self.cutmix_similar_indices
        else:
            pool = self.all_indices

        if pool is None or len(pool) < 2:
            return self._pack_output(left, right, label, mask, meta, kpred, kmask, weight, group_id)

        rng = self._get_rng()
        if rng.random() >= self.cutmix_prob:
            return self._pack_output(left, right, label, mask, meta, kpred, kmask, weight, group_id)

        mix_idx = int(rng.choice(pool))
        if mix_idx == idx and len(pool) > 1:
            mix_idx = int(rng.choice(pool))
        if mix_idx == idx:
            return self._pack_output(left, right, label, mask, meta, kpred, kmask, weight, group_id)

        left2, right2, label2 = self._load_sample(mix_idx)
        mask2 = None
        if self.return_mask:
            mask2 = torch.from_numpy(self.label_mask[mix_idx])
        left, right, label = _apply_cutmix(
            left, right, label, left2, right2, label2, rng, self.cutmix_alpha
        )
        if mask is not None and mask2 is not None:
            mask = mask * mask2
        if meta is not None:
            for key in list(meta.keys()):
                if key.endswith("_mask"):
                    meta[key] = torch.zeros_like(meta[key])
        if kmask is not None:
            kmask = torch.zeros_like(kmask)
            kpred = torch.zeros_like(kpred)
        return self._pack_output(left, right, label, mask, meta, kpred, kmask, weight, group_id)

class BiomassUnlabeledDataset(Dataset):
    def __init__(self, image_paths, transform, pair_mode="split"):
        self.paths = list(image_paths)
        self.transform = transform
        self.pair_mode = "split" if pair_mode is None else str(pair_mode)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        left, right = _make_ulb_pair(img, self.pair_mode)
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        return left, right

def _resolve_vistamilk_labelled_dir():
    root = CFG.VISTAMILK_DIR
    candidates = [
        os.path.join(root, "vistamilk", "labelled"),
        os.path.join(root, "labelled"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"VistaMilk labelled dir not found under {root}")

def _vistamilk_columns():
    return [
        "Image Name",
        "Herbage Mass (kg DM/ha)",
        "Grass Dried",
        "Clover Dried",
        "Weeds Dried",
    ]

def _load_vistamilk_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in _vistamilk_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    if CFG.VISTAMILK_TOTAL_TARGET not in ("gdm", "total"):
        raise ValueError(f"Unsupported VISTAMILK_TOTAL_TARGET: {CFG.VISTAMILK_TOTAL_TARGET}")

    total_kg_ha = pd.to_numeric(df["Herbage Mass (kg DM/ha)"], errors="coerce").fillna(0.0).astype(np.float32)
    grass = pd.to_numeric(df["Grass Dried"], errors="coerce").fillna(0.0).astype(np.float32)
    clover = pd.to_numeric(df["Clover Dried"], errors="coerce").fillna(0.0).astype(np.float32)
    weeds = pd.to_numeric(df["Weeds Dried"], errors="coerce").fillna(0.0).astype(np.float32)

    total = (
        total_kg_ha
        * (float(CFG.VISTAMILK_PLOT_M2) / float(CFG.VISTAMILK_HA_M2))
        * float(CFG.VISTAMILK_KG_TO_G)
        * float(CFG.VISTAMILK_MASS_SCALE)
    )
    green = (grass + weeds) / 100.0 * total
    clover_mass = clover / 100.0 * total
    gdm = total

    if CFG.VISTAMILK_TOTAL_TARGET == "total":
        gdm_val = np.zeros_like(gdm)
        total_val = total
        mask_row = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    else:
        gdm_val = gdm
        total_val = np.zeros_like(gdm)
        mask_row = np.array([1, 0, 1, 1, 0], dtype=np.float32)

    out = pd.DataFrame({
        "image_path": df["Image Name"].astype(str),
        "Dry_Green_g": green,
        "Dry_Dead_g": 0.0,
        "Dry_Clover_g": clover_mass,
        "GDM_g": gdm_val,
        "Dry_Total_g": total_val,
    })
    mask = np.tile(mask_row, (len(out), 1))
    return out, mask

def _load_vistamilk_group(csv_paths):
    dfs = []
    masks = []
    for p in csv_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"VistaMilk CSV not found: {p}")
        df, mask = _load_vistamilk_csv(p)
        dfs.append(df)
        masks.append(mask)
    if not dfs:
        return None, None
    df_all = pd.concat(dfs, ignore_index=True)
    mask_all = np.concatenate(masks, axis=0)
    return df_all, mask_all

def build_vistamilk_datasets(
    transform,
    meta_info=None,
    return_meta=False,
    return_kernel=False,
    return_weight=False,
    sample_weight=None,
    return_group=False,
    group_id_value=None,
):
    if not CFG.USE_VISTAMILK:
        return []
    labelled_dir = _resolve_vistamilk_labelled_dir()
    datasets = []

    if CFG.VISTAMILK_INCLUDE_CAMERA:
        camera_root = os.path.join(labelled_dir, "camera", "camera")
        camera_csvs = [os.path.join(camera_root, "train.csv")]
        if CFG.VISTAMILK_USE_VAL:
            camera_csvs.append(os.path.join(camera_root, "val.csv"))
        camera_df, camera_mask = _load_vistamilk_group(camera_csvs)
        if camera_df is not None:
            cam_weight = None
            if return_weight:
                w = 1.0 if sample_weight is None else float(sample_weight)
                cam_weight = np.full((len(camera_df),), w, dtype=np.float32)
            cam_group = None
            if return_group:
                gid = -1 if group_id_value is None else int(group_id_value)
                cam_group = np.full((len(camera_df),), gid, dtype=np.int64)
            ds = BiomassDataset(
                camera_df,
                transform,
                os.path.join(camera_root, "images"),
                cutmix_mode="none",
                return_mask=True,
                label_mask=camera_mask,
                meta_info=meta_info,
                return_meta=return_meta,
                return_kernel=return_kernel,
                sample_weight=cam_weight,
                return_weight=return_weight,
                group_ids=cam_group,
                return_group=return_group,
            )
            datasets.append(ds)

    if CFG.VISTAMILK_INCLUDE_PHONE:
        phone_root = os.path.join(labelled_dir, "phone", "phone")
        phone_csvs = [os.path.join(phone_root, "phone_gt_train.csv")]
        if CFG.VISTAMILK_USE_VAL:
            phone_csvs.append(os.path.join(phone_root, "phone_gt_val.csv"))
        phone_df, phone_mask = _load_vistamilk_group(phone_csvs)
        if phone_df is not None:
            phone_weight = None
            if return_weight:
                w = 1.0 if sample_weight is None else float(sample_weight)
                phone_weight = np.full((len(phone_df),), w, dtype=np.float32)
            phone_group = None
            if return_group:
                gid = -1 if group_id_value is None else int(group_id_value)
                phone_group = np.full((len(phone_df),), gid, dtype=np.int64)
            ds = BiomassDataset(
                phone_df,
                transform,
                os.path.join(phone_root, "images"),
                cutmix_mode="none",
                return_mask=True,
                label_mask=phone_mask,
                meta_info=meta_info,
                return_meta=return_meta,
                return_kernel=return_kernel,
                sample_weight=phone_weight,
                return_weight=return_weight,
                group_ids=phone_group,
                return_group=return_group,
            )
            datasets.append(ds)

    total = sum(len(ds) for ds in datasets)
    print(f"VistaMilk supplemental samples: {total}")
    return datasets

def _list_image_files(root_dir):
    if not os.path.isdir(root_dir):
        return []
    return [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

def _normalize_token(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()

def collect_vistamilk_unlabeled_paths():
    if not CFG.UCVME_USE_VISTAMILK:
        return []
    labelled_dir = _resolve_vistamilk_labelled_dir()
    paths = []
    if CFG.VISTAMILK_INCLUDE_CAMERA:
        cam_dir = os.path.join(labelled_dir, "camera", "camera", "images")
        paths.extend(_list_image_files(cam_dir))
    if CFG.VISTAMILK_INCLUDE_PHONE:
        phone_dir = os.path.join(labelled_dir, "phone", "phone", "images")
        paths.extend(_list_image_files(phone_dir))
    return sorted(set(paths))

def collect_vegann_unlabeled_paths():
    if not getattr(CFG, "UCVME_USE_VEGANN", False):
        return []
    csv_path = getattr(CFG, "UCVME_VEGANN_CSV", None)
    img_dir = getattr(CFG, "UCVME_VEGANN_IMG_DIR", None)
    if not csv_path or not os.path.exists(csv_path):
        print(f"VegAnn CSV not found: {csv_path}")
        return []
    if not img_dir or not os.path.isdir(img_dir):
        print(f"VegAnn image dir not found: {img_dir}")
        return []

    df = pd.read_csv(csv_path, sep=";")
    if "Name" not in df.columns:
        raise ValueError("VegAnn CSV missing 'Name' column.")

    species_allow = getattr(CFG, "UCVME_VEGANN_SPECIES", None)
    if species_allow and "Species" in df.columns:
        allow = {_normalize_token(s) for s in species_allow}
        df = df[df["Species"].map(_normalize_token).isin(allow)]

    orient_allow = getattr(CFG, "UCVME_VEGANN_ORIENTATIONS", None)
    if orient_allow and "Orientation" in df.columns:
        allow = {_normalize_token(s) for s in orient_allow}
        df = df[df["Orientation"].map(_normalize_token).isin(allow)]

    system_allow = getattr(CFG, "UCVME_VEGANN_SYSTEMS", None)
    if system_allow and "System" in df.columns:
        allow = {_normalize_token(s) for s in system_allow}
        df = df[df["System"].map(_normalize_token).isin(allow)]

    split_col = getattr(CFG, "UCVME_VEGANN_SPLIT_COL", None)
    split_allow = getattr(CFG, "UCVME_VEGANN_SPLITS", None)
    if split_col and split_allow and split_col in df.columns:
        allow = {_normalize_token(s) for s in split_allow}
        df = df[df[split_col].map(_normalize_token).isin(allow)]

    paths = []
    for name in df["Name"].astype(str).tolist():
        fname = os.path.basename(name)
        path = os.path.join(img_dir, fname)
        if os.path.isfile(path):
            paths.append(path)
    paths = sorted(set(paths))
    if paths:
        print(f"VegAnn unlabeled samples: {len(paths)}")
    return paths

def collect_ucvme_unlabeled_sources():
    sources = []
    if CFG.UCVME_USE_VISTAMILK:
        vm_paths = collect_vistamilk_unlabeled_paths()
        if vm_paths:
            sources.append((vm_paths, CFG.UCVME_VISTAMILK_ULB_MODE))
    if getattr(CFG, "UCVME_USE_VEGANN", False):
        veg_paths = collect_vegann_unlabeled_paths()
        if veg_paths:
            sources.append((veg_paths, CFG.UCVME_VEGANN_ULB_MODE))

    extra_dirs = list(getattr(CFG, "UCVME_EXTRA_UNLABELED_DIRS", []))
    if extra_dirs:
        extra_modes = getattr(CFG, "UCVME_EXTRA_UNLABELED_MODES", None)
        if extra_modes is not None and len(extra_modes) != len(extra_dirs):
            raise ValueError("UCVME_EXTRA_UNLABELED_MODES must match UCVME_EXTRA_UNLABELED_DIRS length.")
        for i, d in enumerate(extra_dirs):
            paths = _list_image_files(d)
            if not paths:
                continue
            mode = extra_modes[i] if extra_modes is not None else CFG.UCVME_EXTRA_UNLABELED_MODE
            sources.append((sorted(set(paths)), mode))

    return sources

def load_bad_marks_set():
    if not (CFG.USE_BAD_MARKS_FILTER or CFG.EVAL_BAD_MARKS):
        return set()
    path = CFG.BAD_MARKS_CSV
    if not path or not os.path.exists(path):
        print(f"Bad marks CSV not found at {path}; skipping.")
        return set()
    df = pd.read_csv(path)
    if CFG.BAD_MARKS_COLUMN not in df.columns:
        raise ValueError(f"Bad marks column missing: {CFG.BAD_MARKS_COLUMN}")
    bad_df = df[df[CFG.BAD_MARKS_COLUMN].astype(int) == 1]
    if CFG.BAD_MARKS_SPLIT and "split" in bad_df.columns:
        bad_df = bad_df[bad_df["split"] == CFG.BAD_MARKS_SPLIT]
    if bad_df.empty:
        return set()
    return set(bad_df["image_path"].astype(str).map(os.path.basename).tolist())

def _print_sample_diffs(df, preds, labels, title, max_rows=0):
    if max_rows is None:
        return
    if len(df) == 0:
        print(f"{title}: no samples.")
        return
    n = len(df)
    limit = n if max_rows == 0 else min(n, max_rows)
    print(f"\n{title} (showing {limit}/{n})")
    for i in range(limit):
        path = df.iloc[i]["image_path"]
        name = os.path.basename(str(path))
        diff = preds[i] - labels[i]
        diff_str = " ".join([f"{v:+.2f}" for v in diff.tolist()])
        print(f"{name} | diff [{diff_str}]")

@torch.inference_mode()
def evaluate_bad_marks(model, df, img_dir, title, print_limit=0, meta_info=None, return_meta=False):
    if df is None or len(df) == 0:
        print(f"{title}: no samples.")
        return
    ds = BiomassDataset(
        df,
        get_val_transforms(),
        img_dir,
        cutmix_mode="none",
        return_mask=True,
        meta_info=meta_info,
        return_meta=return_meta,
    )
    dl = DataLoader(
        ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )
    loss, global_r2, avg_r2, per_r2, preds, labels = valid_epoch(model, dl, CFG.DEVICE)
    print(f"\n{title} metrics | loss={loss:.5f} avgR2={avg_r2:.4f} globalR2={global_r2:.4f}")
    _print_sample_diffs(df, preds, labels, title, max_rows=print_limit)

@torch.inference_mode()
def compute_half_similarity(df, img_dir, device):
    print("Computing half-embeddings for cutmix similarity...")
    ds = BiomassHalfEmbedDataset(df, img_dir)
    dl = DataLoader(
        ds,
        batch_size=CFG.CUTMIX_EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    if CFG.BACKBONE_PATH and os.path.exists(CFG.BACKBONE_PATH):
        embed_model = timm.create_model(
            CFG.MODEL_NAME, pretrained=False, num_classes=0, global_pool='avg'
        )
        sd = torch.load(CFG.BACKBONE_PATH, map_location='cpu')
        if isinstance(sd, dict) and ('model' in sd or 'state_dict' in sd):
            key = 'model' if 'model' in sd else 'state_dict'
            sd = sd[key]
        embed_model.load_state_dict(sd, strict=False)
    else:
        embed_model = timm.create_model(
            CFG.MODEL_NAME, pretrained=True, num_classes=0, global_pool='avg'
        )
    embed_model.to(device)
    embed_model.eval()

    sims = np.empty(len(ds), dtype=np.float32)
    it = tqdm(dl, desc='embed', leave=False) if CFG.USE_TQDM else dl
    for idxs, left, right in it:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        emb_l = embed_model(left)
        emb_r = embed_model(right)
        emb_l = F.normalize(emb_l, dim=1)
        emb_r = F.normalize(emb_r, dim=1)
        sim = (emb_l * emb_r).sum(dim=1).detach().cpu().numpy()
        sims[idxs.numpy()] = sim

    del embed_model
    torch.cuda.empty_cache()
    gc.collect()
    return sims

def _build_embed_model(device):
    if CFG.BACKBONE_PATH and os.path.exists(CFG.BACKBONE_PATH):
        embed_model = timm.create_model(
            CFG.MODEL_NAME, pretrained=False, num_classes=0, global_pool='avg'
        )
        sd = torch.load(CFG.BACKBONE_PATH, map_location='cpu')
        if isinstance(sd, dict) and ('model' in sd or 'state_dict' in sd):
            key = 'model' if 'model' in sd else 'state_dict'
            sd = sd[key]
        embed_model.load_state_dict(sd, strict=False)
    else:
        embed_model = timm.create_model(
            CFG.MODEL_NAME, pretrained=True, num_classes=0, global_pool='avg'
        )
    embed_model.to(device)
    embed_model.eval()
    return embed_model

@torch.inference_mode()
def compute_kernel_embeddings(df, img_dir, device):
    print("Computing kernel embeddings...")
    ds = BiomassHalfEmbedDataset(df, img_dir)
    dl = DataLoader(
        ds,
        batch_size=CFG.KERNEL_REG_EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    embed_model = _build_embed_model(device)

    emb = np.empty((len(ds), embed_model.num_features), dtype=np.float32)
    it = tqdm(dl, desc='kernel-embed', leave=False) if CFG.USE_TQDM else dl
    for idxs, left, right in it:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        emb_l = embed_model(left)
        emb_r = embed_model(right)
        if emb_l.ndim == 3:
            emb_l = emb_l.mean(dim=1)
        if emb_r.ndim == 3:
            emb_r = emb_r.mean(dim=1)
        emb_lr = (emb_l + emb_r) * 0.5
        emb_lr = F.normalize(emb_lr, dim=1)
        emb[idxs.numpy()] = emb_lr.detach().float().cpu().numpy()

    del embed_model
    torch.cuda.empty_cache()
    gc.collect()
    return emb

@torch.inference_mode()
def kernel_regression_predict(
    train_emb,
    train_labels,
    query_emb,
    device,
    topk,
    tau,
    batch_size,
    exclude_self=False,
    train_indices=None,
    query_indices=None,
):
    train_emb_t = torch.as_tensor(train_emb, dtype=torch.float32, device=device)
    train_labels_t = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    if tau <= 0:
        raise ValueError("KERNEL_REG_TAU must be > 0.")

    train_id_to_pos = None
    if exclude_self and train_indices is not None:
        train_id_to_pos = {int(idx): i for i, idx in enumerate(train_indices)}

    preds = np.zeros((len(query_emb), train_labels.shape[1]), dtype=np.float32)
    n_train = train_emb_t.shape[0]
    use_topk = topk is not None and topk > 0 and topk < n_train

    for start in range(0, len(query_emb), batch_size):
        end = min(start + batch_size, len(query_emb))
        q = torch.as_tensor(query_emb[start:end], dtype=torch.float32, device=device)
        sims = q @ train_emb_t.T
        if train_id_to_pos is not None and query_indices is not None:
            for i, qidx in enumerate(query_indices[start:end]):
                pos = train_id_to_pos.get(int(qidx))
                if pos is not None:
                    sims[i, pos] = -1e9

        if use_topk:
            vals, idx = torch.topk(sims, k=topk, dim=1)
            weights = torch.softmax(vals / tau, dim=1)
            labels_k = train_labels_t[idx]
            pred = torch.einsum("bk,bkd->bd", weights, labels_k)
        else:
            weights = torch.softmax(sims / tau, dim=1)
            pred = weights @ train_labels_t

        preds[start:end] = pred.detach().cpu().numpy()

    return preds

def _clamp_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        logvar,
        min=float(CFG.HETERO_MIN_LOGVAR),
        max=float(CFG.HETERO_MAX_LOGVAR),
    )

def _build_logvar_pack(logvar_comp):
    if logvar_comp is None:
        return None
    logvar_comp = _clamp_logvar(logvar_comp)
    lv_green = logvar_comp[:, 0:1]
    lv_dead = logvar_comp[:, 1:2]
    lv_clover = logvar_comp[:, 2:3]

    var_green = torch.exp(lv_green)
    var_dead = torch.exp(lv_dead)
    var_clover = torch.exp(lv_clover)

    if logvar_comp.shape[1] >= 5:
        lv_gdm = logvar_comp[:, 3:4]
        lv_total = logvar_comp[:, 4:5]
        var_gdm = torch.exp(lv_gdm)
        var_total = torch.exp(lv_total)
    else:
        var_gdm = var_green + var_clover
        var_total = var_gdm + var_dead

    var_pack = torch.cat([var_green, var_dead, var_clover, var_gdm, var_total], dim=1)
    return torch.log(var_pack + float(CFG.HETERO_EPS))

_NONFINITE_SEEN = set()

def _check_nonfinite(name, tensor, context=""):
    if tensor is None or not getattr(CFG, "DIAG_NONFINITE", False):
        return False
    t = tensor.detach()
    if torch.isfinite(t).all():
        return False
    key = f"{context}:{name}"
    if key in _NONFINITE_SEEN:
        return True
    _NONFINITE_SEEN.add(key)
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    finite = t[torch.isfinite(t)]
    min_val = finite.min().item() if finite.numel() else float("nan")
    max_val = finite.max().item() if finite.numel() else float("nan")
    print(
        f"[nonfinite] {context} {name}: nan={nan_count} inf={inf_count} "
        f"finite_min={min_val:.3g} finite_max={max_val:.3g}"
    )
    return True

# Layers
class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) from the reference notebook.
    Efficiently mixes tokens with linear complexity.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g
        # Spatial mixing via 1D Conv (requires transpose)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        # Projection
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x

# Model
class BiomassModel(nn.Module):
    def __init__(self, model_name, pretrained=True, backbone_path=None, meta_info=None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        self.meta_info = meta_info or {}
        
        # 1. Load Backbone with global_pool='' to keep patch tokens
        #    (B, 197, 1024) instead of (B, 1024)
        backbone_kwargs = {}
        if getattr(CFG, "USE_UCVME", False):
            backbone_kwargs["drop_rate"] = float(CFG.UCVME_BACKBONE_DROP_RATE)
            backbone_kwargs["drop_path_rate"] = float(CFG.UCVME_BACKBONE_DROP_PATH_RATE)
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
            global_pool='',
            **backbone_kwargs,
        )
        
        # 2. Enable Gradient Checkpointing (Crucial for ViT-Large memory!)
        if hasattr(self.backbone, 'set_grad_checkpointing') and CFG.DINO_GRAD_CHECKPOINTING:
            self.backbone.set_grad_checkpointing(True)
            print("✓ Gradient Checkpointing enabled (saves ~50% VRAM)")
            
        nf = self.backbone.num_features
        
        # 3. Mamba Fusion Neck
        #    Mixes the concatenated tokens [Left, Right]
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
        )
        
        # 4. Pooling & Heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Heads (using the same logic as before, but on fused features)
        self.head_green_raw  = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_clover_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_dead_raw   = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_gdm_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_total_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf//2, 1), nn.Softplus()
        )

        self.use_hetero = bool(CFG.USE_HETERO)
        if self.use_hetero:
            self.head_green_logvar = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_clover_logvar = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_dead_logvar = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_gdm_logvar = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_total_logvar = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )

        self.use_kernel_residual = bool(CFG.USE_KERNEL_REG and CFG.KERNEL_REG_MODE == "residual")
        if self.use_kernel_residual:
            self.head_green_res = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_clover_res = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_dead_res = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_gdm_res = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
            self.head_total_res = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )

        self.use_date = bool(self.meta_info.get("use_date"))
        self.use_state = bool(self.meta_info.get("use_state"))
        self.use_species = bool(self.meta_info.get("use_species"))
        self.use_ndvi = bool(self.meta_info.get("use_ndvi"))
        self.use_height = bool(self.meta_info.get("use_height"))

        if self.use_date:
            date_out = 2 if CFG.DATE_MODE == "dayofyear_sincos" else 1
            self.head_date = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, date_out)
            )
        if self.use_state:
            num_states = int(self.meta_info.get("num_states", 0))
            if num_states <= 0:
                raise ValueError("State prediction enabled but no state classes available.")
            self.head_state = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, num_states)
            )
        if self.use_species:
            num_species = int(self.meta_info.get("num_species", 0))
            if num_species <= 0:
                raise ValueError("Species prediction enabled but no species classes available.")
            self.head_species = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, num_species)
            )
        if self.use_ndvi:
            self.head_ndvi = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )
        if self.use_height:
            self.head_height = nn.Sequential(
                nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf//2, 1)
            )

        self.kernel_alpha = None
        if CFG.USE_KERNEL_REG and CFG.KERNEL_REG_MODE == "blend":
            alpha_shape = len(CFG.ALL_TARGET_COLS) if CFG.KERNEL_REG_ALPHA_PER_TARGET else 1
            init = float(CFG.KERNEL_REG_ALPHA_INIT)
            self.kernel_alpha = nn.Parameter(torch.full((alpha_shape,), init))

        self.use_kernel_mlp = bool(CFG.USE_KERNEL_REG and CFG.KERNEL_REG_MODE == "mlp")
        if self.use_kernel_mlp:
            in_dim = nf + len(CFG.ALL_TARGET_COLS)
            self.head_green_kernel = nn.Sequential(
                nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(nf//2, 1), nn.Softplus()
            )
            self.head_clover_kernel = nn.Sequential(
                nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(nf//2, 1), nn.Softplus()
            )
            self.head_dead_kernel = nn.Sequential(
                nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(nf//2, 1), nn.Softplus()
            )
            self.head_gdm_kernel = nn.Sequential(
                nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(nf//2, 1), nn.Softplus()
            )
            self.head_total_kernel = nn.Sequential(
                nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(nf//2, 1), nn.Softplus()
            )
            if self.use_hetero:
                self.head_green_logvar_kernel = nn.Sequential(
                    nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(nf//2, 1)
                )
                self.head_clover_logvar_kernel = nn.Sequential(
                    nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(nf//2, 1)
                )
                self.head_dead_logvar_kernel = nn.Sequential(
                    nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(nf//2, 1)
                )
                self.head_gdm_logvar_kernel = nn.Sequential(
                    nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(nf//2, 1)
                )
                self.head_total_logvar_kernel = nn.Sequential(
                    nn.Linear(in_dim, nf//2), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(nf//2, 1)
                )
        
        if pretrained:
            self.load_pretrained()
    
    def load_pretrained(self):
        try:
            # Load weights normally
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(f"Loading backbone weights from local file: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                # Handle common checkpoint wrappers (e.g. if saved with 'model' key)
                if 'model' in sd: sd = sd['model']
                elif 'state_dict' in sd: sd = sd['state_dict']
            else:
                # Original behavior: Download from internet
                print("Downloading backbone weights...")
                sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='').state_dict()
            
            # Interpolate pos_embed if needed (for 256x256 vs 224x224)
            if 'pos_embed' in sd and hasattr(self.backbone, 'pos_embed'):
                pe_ck = sd['pos_embed']
                pe_m  = self.backbone.pos_embed
                if pe_ck.shape != pe_m.shape:
                    print(f"Interpolating pos_embed: {pe_ck.shape} -> {pe_m.shape}")
                    # (Simple interpolation logic here or rely on timm's load if strict=False handles it well enough)
                    # For robust interpolation, use the snippet provided in previous turn
            
            self.backbone.load_state_dict(sd, strict=False)
            print('Pretrained weights loaded.')
        except Exception as e:
            print(f'Warning: pretrained load failed: {e}')
    
    def forward(self, left, right):
        # 1. Extract Tokens (B, N, D)
        #    Note: ViT usually returns [CLS, Patch1, Patch2...]
        #    We remove CLS token for spatial mixing, or keep it. Let's keep it.
        x_l = self.backbone(left)
        x_r = self.backbone(right)
        
        # 2. Concatenate Left and Right tokens along sequence dimension
        #    (B, N, D) + (B, N, D) -> (B, 2N, D)
        x_cat = torch.cat([x_l, x_r], dim=1)
        
        # 3. Apply Mamba Fusion
        #    This allows tokens from Left image to interact with tokens from Right image
        x_fused = self.fusion(x_cat)
        
        # 4. Global Pooling
        #    (B, 2N, D) -> (B, D, 2N) -> (B, D, 1) -> (B, D)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        
        # 5. Prediction Heads
        if self.use_kernel_residual:
            green = self.head_green_res(x_pool)
            clover = self.head_clover_res(x_pool)
            dead = self.head_dead_res(x_pool)
            gdm = self.head_gdm_res(x_pool)
            total = self.head_total_res(x_pool)
        else:
            green = self.head_green_raw(x_pool)
            clover = self.head_clover_raw(x_pool)
            dead = self.head_dead_raw(x_pool)
            gdm = self.head_gdm_raw(x_pool)
            total = self.head_total_raw(x_pool)

        logvar_comp = None
        if self.use_hetero:
            lv_green = self.head_green_logvar(x_pool)
            lv_clover = self.head_clover_logvar(x_pool)
            lv_dead = self.head_dead_logvar(x_pool)
            lv_gdm = self.head_gdm_logvar(x_pool)
            lv_total = self.head_total_logvar(x_pool)
            logvar_comp = torch.cat([lv_green, lv_dead, lv_clover, lv_gdm, lv_total], dim=1)

        meta_out = {}
        if self.use_date:
            meta_out["date"] = self.head_date(x_pool)
        if self.use_state:
            meta_out["state"] = self.head_state(x_pool)
        if self.use_species:
            meta_out["species"] = self.head_species(x_pool)
        if self.use_ndvi:
            meta_out["ndvi"] = self.head_ndvi(x_pool)
        if self.use_height:
            meta_out["height"] = self.head_height(x_pool)

        need_feat = bool(CFG.USE_KERNEL_REG and CFG.KERNEL_REG_MODE == "mlp")
        if meta_out or need_feat or logvar_comp is not None:
            out = {"targets": (total, gdm, green, clover, dead)}
            if meta_out:
                out["meta"] = meta_out
            if need_feat:
                out["feat"] = x_pool
            if logvar_comp is not None:
                out["target_logvar"] = logvar_comp
            return out
        return total, gdm, green, clover, dead


class UCVMESharedModel(nn.Module):
    def __init__(self, model_name, pretrained=True, backbone_path=None, meta_info=None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        self.meta_info = meta_info or {}

        backbone_kwargs = {}
        if getattr(CFG, "USE_UCVME", False):
            backbone_kwargs["drop_rate"] = float(CFG.UCVME_BACKBONE_DROP_RATE)
            backbone_kwargs["drop_path_rate"] = float(CFG.UCVME_BACKBONE_DROP_PATH_RATE)
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
            global_pool='',
            **backbone_kwargs,
        )

        if hasattr(self.backbone, 'set_grad_checkpointing') and CFG.DINO_GRAD_CHECKPOINTING:
            self.backbone.set_grad_checkpointing(True)
            print("✓ Gradient Checkpointing enabled (saves ~50% VRAM)")

        nf = self.backbone.num_features
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.use_hetero = bool(CFG.USE_HETERO)
        self.use_date = bool(self.meta_info.get("use_date"))
        self.use_state = bool(self.meta_info.get("use_state"))
        self.use_species = bool(self.meta_info.get("use_species"))
        self.use_ndvi = bool(self.meta_info.get("use_ndvi"))
        self.use_height = bool(self.meta_info.get("use_height"))

        self.branch_a = self._make_branch(nf)
        self.branch_b = self._make_branch(nf)

        if pretrained:
            self.load_pretrained()

    def _make_branch(self, nf):
        branch = nn.ModuleDict()
        branch["head_green_raw"] = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus(),
        )
        branch["head_clover_raw"] = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus(),
        )
        branch["head_dead_raw"] = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus(),
        )
        branch["head_gdm_raw"] = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus(),
        )
        branch["head_total_raw"] = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf // 2, 1), nn.Softplus(),
        )

        if self.use_hetero:
            branch["head_green_logvar"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, 1),
            )
            branch["head_clover_logvar"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, 1),
            )
            branch["head_dead_logvar"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, 1),
            )
            branch["head_gdm_logvar"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, 1),
            )
            branch["head_total_logvar"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, 1),
            )

        if self.use_date:
            date_out = 2 if CFG.DATE_MODE == "dayofyear_sincos" else 1
            branch["head_date"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, date_out),
            )
        if self.use_state:
            num_states = int(self.meta_info.get("num_states", 0))
            if num_states <= 0:
                raise ValueError("State prediction enabled but no state classes available.")
            branch["head_state"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, num_states),
            )
        if self.use_species:
            num_species = int(self.meta_info.get("num_species", 0))
            if num_species <= 0:
                raise ValueError("Species prediction enabled but no species classes available.")
            branch["head_species"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, num_species),
            )
        if self.use_ndvi:
            branch["head_ndvi"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, 1),
            )
        if self.use_height:
            branch["head_height"] = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(nf // 2, 1),
            )
        return branch

    def load_pretrained(self):
        try:
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(f"Loading backbone weights from local file: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                if 'model' in sd: sd = sd['model']
                elif 'state_dict' in sd: sd = sd['state_dict']
            else:
                print("Downloading backbone weights...")
                sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='').state_dict()
            self.backbone.load_state_dict(sd, strict=False)
            print('Pretrained weights loaded.')
        except Exception as e:
            print(f'Warning: pretrained load failed: {e}')

    def _forward_branch(self, branch, x_pool):
        green = branch["head_green_raw"](x_pool)
        clover = branch["head_clover_raw"](x_pool)
        dead = branch["head_dead_raw"](x_pool)
        gdm = branch["head_gdm_raw"](x_pool)
        total = branch["head_total_raw"](x_pool)

        logvar_comp = None
        if self.use_hetero:
            lv_green = branch["head_green_logvar"](x_pool)
            lv_clover = branch["head_clover_logvar"](x_pool)
            lv_dead = branch["head_dead_logvar"](x_pool)
            lv_gdm = branch["head_gdm_logvar"](x_pool)
            lv_total = branch["head_total_logvar"](x_pool)
            logvar_comp = torch.cat([lv_green, lv_dead, lv_clover, lv_gdm, lv_total], dim=1)

        meta_out = {}
        if self.use_date:
            meta_out["date"] = branch["head_date"](x_pool)
        if self.use_state:
            meta_out["state"] = branch["head_state"](x_pool)
        if self.use_species:
            meta_out["species"] = branch["head_species"](x_pool)
        if self.use_ndvi:
            meta_out["ndvi"] = branch["head_ndvi"](x_pool)
        if self.use_height:
            meta_out["height"] = branch["head_height"](x_pool)

        out = {"targets": (total, gdm, green, clover, dead)}
        if meta_out:
            out["meta"] = meta_out
        if logvar_comp is not None:
            out["target_logvar"] = logvar_comp
        return out

    def forward(self, left, right):
        x_l = self.backbone(left)
        x_r = self.backbone(right)
        x_cat = torch.cat([x_l, x_r], dim=1)
        x_fused = self.fusion(x_cat)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        out_a = self._forward_branch(self.branch_a, x_pool)
        out_b = self._forward_branch(self.branch_b, x_pool)
        return out_a, out_b

# Utility Functions
def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


def build_optimizer(model: BiomassModel):
    # 1. Get backbone parameter IDs for exclusion
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    
    # 2. Separate params into backbone vs. everything else (heads, fusion, etc.)
    backbone_params = []
    rest_params = []
    
    for p in model.parameters():
        if p.requires_grad:
            if id(p) in backbone_ids:
                backbone_params.append(p)
            else:
                rest_params.append(p)
    
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WD},
        {'params': rest_params,     'lr': CFG.LR_REST,     'weight_decay': CFG.WD},
])

def build_scheduler(optimizer):
    def lr_lambda(epoch):
        e = max(0, epoch - 1)
        if e < CFG.WARMUP_EPOCHS:
            return float(e + 1) / float(max(1, CFG.WARMUP_EPOCHS))
        progress = (e - CFG.WARMUP_EPOCHS) / float(max(1, CFG.EPOCHS - CFG.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

# Training and Validation Loops 

from contextlib import nullcontext

USE_BF16 = True
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

scaler = torch.amp.GradScaler(
    'cuda',
    enabled=(torch.cuda.is_available() and AMP_DTYPE == torch.float16)
)

def autocast_ctx():
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE)

def _as_col(x: torch.Tensor, bs: int) -> torch.Tensor:
    """Force predictions to (B,1) regardless of model head returning (B,) or (B,1)."""
    if x.ndim == 1:
        return x.view(bs, 1)
    if x.ndim == 2:
        return x
    return x.view(bs, 1)

def _ensure_2d_lab(lab: torch.Tensor, bs: int) -> torch.Tensor:
    """Force labels to (B,T)."""
    if lab.ndim == 1:
        return lab.view(bs, -1)
    return lab

def _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs: int) -> torch.Tensor:
    """
    Pack predictions in the exact order expected by your metric:
    CFG.ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    """
    pg = _as_col(p_green,  bs)
    pd = _as_col(p_dead,   bs)
    pc = _as_col(p_clover, bs)
    pgdm = _as_col(p_gdm,  bs)
    pt = _as_col(p_total,  bs)
    return torch.cat([pg, pd, pc, pgdm, pt], dim=1)  # (B,5)

def _unpack_pred_pack(pred_pack: torch.Tensor):
    pg = pred_pack[:, 0:1]
    pd = pred_pack[:, 1:2]
    pc = pred_pack[:, 2:3]
    pgdm = pred_pack[:, 3:4]
    pt = pred_pack[:, 4:5]
    return pt, pgdm, pg, pc, pd

def _apply_kernel_reg(pred_pack, kernel_pred, kernel_mask, model, feat=None):
    if not CFG.USE_KERNEL_REG or kernel_pred is None:
        return pred_pack
    kpred = kernel_pred.to(device=pred_pack.device, dtype=pred_pack.dtype)
    kmask = kernel_mask
    if kmask is None:
        kmask = torch.ones((pred_pack.size(0), 1), device=pred_pack.device, dtype=pred_pack.dtype)
    else:
        kmask = kmask.to(device=pred_pack.device, dtype=pred_pack.dtype)
        if kmask.ndim == 1:
            kmask = kmask.view(-1, 1)
        elif kmask.ndim > 2:
            kmask = kmask.view(kmask.size(0), -1)
        if kmask.shape[1] != 1:
            kmask = kmask[:, :1]

    if CFG.KERNEL_REG_MODE == "blend":
        if hasattr(model, "kernel_alpha") and model.kernel_alpha is not None:
            alpha = torch.sigmoid(model.kernel_alpha)
            if alpha.ndim == 0:
                alpha = alpha.view(1, 1)
            elif alpha.ndim == 1:
                alpha = alpha.view(1, -1)
        else:
            alpha = torch.tensor(float(CFG.KERNEL_REG_ALPHA_INIT), device=pred_pack.device).view(1, 1)
        if alpha.shape[1] == 1:
            alpha = alpha.expand(1, pred_pack.shape[1])
        blended = alpha * pred_pack + (1.0 - alpha) * kpred
        return blended * kmask + pred_pack * (1.0 - kmask)
    if CFG.KERNEL_REG_MODE == "residual":
        blended = kpred + pred_pack
        return blended * kmask + pred_pack * (1.0 - kmask)
    if CFG.KERNEL_REG_MODE == "mlp":
        if not hasattr(model, "head_green_kernel") or not model.use_kernel_mlp:
            raise RuntimeError("Kernel heads not initialized for KERNEL_REG_MODE='mlp'.")
        if feat is None:
            raise RuntimeError("Missing feature embeddings for KERNEL_REG_MODE='mlp'.")
        mlp_in = torch.cat([feat, kpred], dim=1)
        green = model.head_green_kernel(mlp_in)
        clover = model.head_clover_kernel(mlp_in)
        dead = model.head_dead_kernel(mlp_in)
        gdm = model.head_gdm_kernel(mlp_in)
        total = model.head_total_kernel(mlp_in)
        fused = torch.cat([green, dead, clover, gdm, total], dim=1)
        return fused * kmask + pred_pack * (1.0 - kmask)
    raise ValueError(f"Unsupported KERNEL_REG_MODE: {CFG.KERNEL_REG_MODE}")

def _split_outputs(outputs):
    if isinstance(outputs, dict):
        return (
            outputs.get("targets"),
            outputs.get("meta"),
            outputs.get("feat"),
            outputs.get("target_logvar"),
        )
    if isinstance(outputs, (list, tuple)) and len(outputs) == 3 and isinstance(outputs[1], dict):
        return outputs[0], outputs[1], outputs[2], None
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2 and isinstance(outputs[1], dict):
        return outputs[0], outputs[1], None, None
    return outputs, None, None, None

def _ucvme_pack_from_output(output, bs, model=None, kpred=None, kmask=None):
    target_out, meta_out, feat, logvar_comp = _split_outputs(output)
    total, gdm, green, clover, dead = target_out
    pred_pack = _pred_pack(total, gdm, green, clover, dead, bs)
    if kpred is not None:
        pred_pack = _apply_kernel_reg(pred_pack, kpred, kmask, model, feat)
    logvar_pack = _build_logvar_pack(logvar_comp) if CFG.USE_HETERO else None
    return pred_pack, logvar_pack, logvar_comp, meta_out

def _ucvme_forward_pack(model, left, right, kpred=None, kmask=None):
    outputs = model(left, right)
    return _ucvme_pack_from_output(outputs, left.size(0), model=model, kpred=kpred, kmask=kmask)

def _ucvme_uncertainty_consistency(logvar_a, logvar_b, mask=None):
    if logvar_a is None or logvar_b is None:
        return None
    diff = logvar_a - logvar_b
    if mask is None:
        return (diff * diff).mean()
    mask_t = mask.to(device=diff.device, dtype=diff.dtype)
    if mask_t.ndim == 1:
        mask_t = mask_t.view(1, -1).expand_as(diff)
    weighted = diff * diff * mask_t
    denom = mask_t.sum()
    if denom <= 0:
        return weighted.sum() * 0.0
    return weighted.sum() / denom

def _ucvme_unc_to_target(logvar_pred, logvar_target, mask=None):
    if logvar_pred is None or logvar_target is None:
        return None
    logvar_target = _clamp_logvar(logvar_target)
    diff = logvar_pred - logvar_target
    if mask is None:
        return (diff * diff).mean()
    mask_t = mask.to(device=diff.device, dtype=diff.dtype)
    if mask_t.ndim == 1:
        mask_t = mask_t.view(1, -1).expand_as(diff)
    weighted = diff * diff * mask_t
    denom = mask_t.sum()
    if denom <= 0:
        return weighted.sum() * 0.0
    return weighted.sum() / denom

def _ucvme_reg_fixed_logvar(pred_pack, target, logvar_target, mask=None):
    logvar_target = _clamp_logvar(logvar_target)
    diff = pred_pack - target
    loss = 0.5 * (torch.exp(-logvar_target) * diff * diff + logvar_target)
    if mask is None:
        return loss.mean()
    mask_t = mask.to(device=loss.device, dtype=loss.dtype)
    if mask_t.ndim == 1:
        mask_t = mask_t.view(1, -1).expand_as(loss)
    weighted = loss * mask_t
    denom = mask_t.sum()
    if denom <= 0:
        return weighted.sum() * 0.0
    return weighted.sum() / denom

@torch.no_grad()
def _ucvme_pseudo_labels(model_a, model_b, left, right, mc_samples):
    preds = []
    logvars = []
    for _ in range(mc_samples):
        pred_a, logvar_a, _, _ = _ucvme_forward_pack(model_a, left, right)
        pred_b, logvar_b, _, _ = _ucvme_forward_pack(model_b, left, right)
        if logvar_a is None or logvar_b is None:
            raise RuntimeError("UCVME requires heteroscedastic logvar outputs.")
        preds.extend([pred_a, pred_b])
        logvars.extend([logvar_a, logvar_b])
    y_e = torch.stack(preds, dim=0).mean(dim=0)
    z_e = torch.stack(logvars, dim=0).mean(dim=0)
    z_e = _clamp_logvar(z_e)
    return y_e.detach(), z_e.detach()

@torch.no_grad()
def _ucvme_pseudo_labels_shared(model, left, right, mc_samples):
    preds = []
    logvars = []
    for _ in range(mc_samples):
        out_a, out_b = model(left, right)
        pred_a, logvar_a, _, _ = _ucvme_pack_from_output(out_a, left.size(0))
        pred_b, logvar_b, _, _ = _ucvme_pack_from_output(out_b, left.size(0))
        if logvar_a is None or logvar_b is None:
            raise RuntimeError("UCVME requires heteroscedastic logvar outputs.")
        preds.extend([pred_a, pred_b])
        logvars.extend([logvar_a, logvar_b])
    y_e = torch.stack(preds, dim=0).mean(dim=0)
    z_e = torch.stack(logvars, dim=0).mean(dim=0)
    z_e = _clamp_logvar(z_e)
    return y_e.detach(), z_e.detach()

def _move_meta(meta, device):
    if meta is None:
        return None
    out = {}
    for k, v in meta.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def _unpack_batch(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) < 3:
        raise ValueError("Unexpected batch format.")
    l, r, lab = batch[0], batch[1], batch[2]
    idx = 3
    mask = None
    meta = None
    kpred = None
    kmask = None
    weight = None
    group_id = None

    if idx < len(batch) and not isinstance(batch[idx], dict):
        mask = batch[idx]
        idx += 1
    if idx < len(batch) and isinstance(batch[idx], dict):
        meta = batch[idx]
        idx += 1
    if idx + 1 < len(batch):
        kpred = batch[idx]
        kmask = batch[idx + 1]
        idx += 2
    if idx < len(batch):
        item = batch[idx]
        idx += 1
        if torch.is_tensor(item) and item.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            group_id = item
        else:
            weight = item
            if idx < len(batch):
                group_id = batch[idx]

    return l, r, lab, mask, meta, kpred, kmask, weight, group_id

def _masked_smooth_l1(pred, target, mask, beta=5.0):
    pred = pred.view(-1)
    target = target.view(-1)
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    if mask is None:
        return loss.mean()
    mask = mask.view(-1).float()
    denom = mask.sum()
    if denom <= 0:
        return loss.sum() * 0.0
    return (loss * mask).sum() / denom

def _masked_ce(logits, target, mask):
    loss = F.cross_entropy(logits, target, reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.view(-1).float()
    denom = mask.sum()
    if denom <= 0:
        return loss.sum() * 0.0
    return (loss * mask).sum() / denom

def _masked_bce(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.view(-1, 1).float()
    denom = mask.sum() * logits.shape[1]
    if denom <= 0:
        return loss.sum() * 0.0
    return (loss * mask).sum() / denom

def compute_meta_loss(meta_out, meta_labels):
    if not meta_out or meta_labels is None:
        return None
    total = None

    if "date" in meta_out and "date" in meta_labels:
        l_date = _masked_smooth_l1(meta_out["date"], meta_labels["date"], meta_labels.get("date_mask"))
        total = l_date * float(CFG.LOSS_WEIGHT_DATE) if total is None else total + l_date * float(CFG.LOSS_WEIGHT_DATE)

    if "state" in meta_out and "state" in meta_labels:
        l_state = _masked_ce(meta_out["state"], meta_labels["state"], meta_labels.get("state_mask"))
        total = l_state * float(CFG.LOSS_WEIGHT_STATE) if total is None else total + l_state * float(CFG.LOSS_WEIGHT_STATE)

    if "species" in meta_out and "species" in meta_labels:
        if CFG.SPECIES_MULTI_LABEL:
            l_species = _masked_bce(meta_out["species"], meta_labels["species"], meta_labels.get("species_mask"))
        else:
            l_species = _masked_ce(meta_out["species"], meta_labels["species"], meta_labels.get("species_mask"))
        total = l_species * float(CFG.LOSS_WEIGHT_SPECIES) if total is None else total + l_species * float(CFG.LOSS_WEIGHT_SPECIES)

    if "ndvi" in meta_out and "ndvi" in meta_labels:
        l_ndvi = _masked_smooth_l1(meta_out["ndvi"], meta_labels["ndvi"], meta_labels.get("ndvi_mask"))
        total = l_ndvi * float(CFG.LOSS_WEIGHT_NDVI) if total is None else total + l_ndvi * float(CFG.LOSS_WEIGHT_NDVI)

    if "height" in meta_out and "height" in meta_labels:
        l_height = _masked_smooth_l1(meta_out["height"], meta_labels["height"], meta_labels.get("height_mask"))
        total = l_height * float(CFG.LOSS_WEIGHT_HEIGHT) if total is None else total + l_height * float(CFG.LOSS_WEIGHT_HEIGHT)

    return total

@torch.inference_mode()
def valid_epoch(eval_model, loader, device):
    eval_model.eval()

    n = len(loader.dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    preds_cpu  = torch.empty((n, n_targets), dtype=torch.float32)
    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)

    total_loss = 0.0
    offset = 0

    for batch in loader:
        l, r, lab, mask, meta, kpred, kmask, weight, group_id = _unpack_batch(batch)
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        meta = _move_meta(meta, device)
        if kpred is not None:
            kpred = kpred.to(device, non_blocking=True)
        if kmask is not None:
            kmask = kmask.to(device, non_blocking=True)
        if weight is not None:
            weight = weight.to(device, non_blocking=True)

        with autocast_ctx():
            outputs = eval_model(l, r)
            target_out, meta_out, feat, logvar_comp = _split_outputs(outputs)
            p_total, p_gdm, p_green, p_clover, p_dead = target_out
            pred_pack = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs)
            pred_pack = _apply_kernel_reg(pred_pack, kpred, kmask, eval_model, feat)
            p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f = _unpack_pred_pack(pred_pack)
            if (
                CFG.USE_HETERO
                and CFG.USE_KERNEL_REG
                and CFG.KERNEL_REG_MODE == "mlp"
                and kpred is not None
                and hasattr(eval_model, "head_green_logvar_kernel")
                and feat is not None
            ):
                mlp_in = torch.cat([feat, kpred], dim=1)
                lv_green = eval_model.head_green_logvar_kernel(mlp_in)
                lv_clover = eval_model.head_clover_logvar_kernel(mlp_in)
                lv_dead = eval_model.head_dead_logvar_kernel(mlp_in)
                lv_gdm = eval_model.head_gdm_logvar_kernel(mlp_in)
                lv_total = eval_model.head_total_logvar_kernel(mlp_in)
                logvar_comp = torch.cat([lv_green, lv_dead, lv_clover, lv_gdm, lv_total], dim=1)
            logvar_pack = _build_logvar_pack(logvar_comp) if CFG.USE_HETERO else None
            if CFG.USE_HETERO:
                _check_nonfinite("logvar_comp", logvar_comp, "valid")
                _check_nonfinite("logvar_pack", logvar_pack, "valid")
            loss = None
            if CFG.PREDICT_TARGETS:
                if CFG.USE_HETERO and logvar_pack is not None:
                    loss_target = hetero_nll_loss(
                        pred_pack,
                        logvar_pack,
                        lab,
                        w=CFG.LOSS_WEIGHTS,
                        mask=mask,
                        sample_weight=weight,
                    )
                else:
                    loss_target = biomass_loss((p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f),
                                               lab, w=CFG.LOSS_WEIGHTS, mask=mask, sample_weight=weight)
                loss = loss_target * float(CFG.LOSS_WEIGHT_TARGETS)
            loss_meta = compute_meta_loss(meta_out, meta)
            if loss_meta is not None:
                loss = loss_meta if loss is None else loss + loss_meta
            if loss is None:
                loss = p_total.sum() * 0.0
            _check_nonfinite("loss", loss, "valid")

        total_loss += loss.detach().float().item() * bs
        batch_pred = pred_pack.float().cpu()
        batch_lab  = _ensure_2d_lab(lab, bs).float().cpu()

        # Safety: ensure dims match (avoids silent 4/5 bugs)
        if batch_pred.shape[1] != n_targets or batch_lab.shape[1] != n_targets:
            raise RuntimeError(
                f"Target dim mismatch: pred={batch_pred.shape}, lab={batch_lab.shape}, "
                f"CFG.ALL_TARGET_COLS={n_targets}"
            )

        preds_cpu[offset:offset+bs]  = batch_pred
        labels_cpu[offset:offset+bs] = batch_lab
        offset += bs

    pred_all    = preds_cpu.numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, pred_all)

    return total_loss / n, global_r2, avg_r2, per_r2, pred_all, true_labels


@torch.inference_mode()
def valid_epoch_tta(eval_model, loaders, device):
    """
    Fixed: always accumulate (N,5). Works even if a head outputs (B,1).
    Assumes each loader iterates in identical order (shuffle=False).
    """
    eval_model.eval()
    assert len(loaders) > 0

    n = len(loaders[0].dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)
    preds_sum  = torch.zeros((n, n_targets), dtype=torch.float32)

    total_loss = 0.0

    for tta_i, loader in enumerate(loaders):
        offset = 0
        tta_loss = 0.0

        for batch in loader:
            l, r, lab, mask, meta, kpred, kmask, weight, group_id = _unpack_batch(batch)
            bs = l.size(0)
            l   = l.to(device, non_blocking=True)
            r   = r.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            meta = _move_meta(meta, device)
            if kpred is not None:
                kpred = kpred.to(device, non_blocking=True)
            if kmask is not None:
                kmask = kmask.to(device, non_blocking=True)
            if weight is not None:
                weight = weight.to(device, non_blocking=True)

            with autocast_ctx():
                outputs = eval_model(l, r)
                target_out, meta_out, feat, logvar_comp = _split_outputs(outputs)
                p_total, p_gdm, p_green, p_clover, p_dead = target_out
                pred_pack = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs)
                pred_pack = _apply_kernel_reg(pred_pack, kpred, kmask, eval_model, feat)
                p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f = _unpack_pred_pack(pred_pack)
                if (
                    CFG.USE_HETERO
                    and CFG.USE_KERNEL_REG
                    and CFG.KERNEL_REG_MODE == "mlp"
                    and kpred is not None
                    and hasattr(eval_model, "head_green_logvar_kernel")
                    and feat is not None
                ):
                    mlp_in = torch.cat([feat, kpred], dim=1)
                    lv_green = eval_model.head_green_logvar_kernel(mlp_in)
                    lv_clover = eval_model.head_clover_logvar_kernel(mlp_in)
                    lv_dead = eval_model.head_dead_logvar_kernel(mlp_in)
                    lv_gdm = eval_model.head_gdm_logvar_kernel(mlp_in)
                    lv_total = eval_model.head_total_logvar_kernel(mlp_in)
                    logvar_comp = torch.cat([lv_green, lv_dead, lv_clover, lv_gdm, lv_total], dim=1)
                logvar_pack = _build_logvar_pack(logvar_comp) if CFG.USE_HETERO else None
                if CFG.USE_HETERO:
                    _check_nonfinite("logvar_comp", logvar_comp, "valid_tta")
                    _check_nonfinite("logvar_pack", logvar_pack, "valid_tta")
                loss = None
                if CFG.PREDICT_TARGETS:
                    if CFG.USE_HETERO and logvar_pack is not None:
                        loss_target = hetero_nll_loss(
                            pred_pack,
                            logvar_pack,
                            lab,
                            w=CFG.LOSS_WEIGHTS,
                            mask=mask,
                            sample_weight=weight,
                        )
                    else:
                        loss_target = biomass_loss((p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f),
                                                   lab, w=CFG.LOSS_WEIGHTS, mask=mask, sample_weight=weight)
                    loss = loss_target * float(CFG.LOSS_WEIGHT_TARGETS)
                loss_meta = compute_meta_loss(meta_out, meta)
                if loss_meta is not None:
                    loss = loss_meta if loss is None else loss + loss_meta
                if loss is None:
                    loss = p_total.sum() * 0.0
                _check_nonfinite("loss", loss, "valid_tta")

            tta_loss += loss.detach().float().item() * bs

            batch_pred = pred_pack.float().cpu()
            preds_sum[offset:offset+bs] += batch_pred

            if tta_i == 0:
                batch_lab = _ensure_2d_lab(lab, bs).float().cpu()
                if batch_lab.shape[1] != n_targets:
                    raise RuntimeError(f"Label dim mismatch: {batch_lab.shape} vs {n_targets}")
                labels_cpu[offset:offset+bs] = batch_lab

            offset += bs

        total_loss += tta_loss / n

    avg_preds   = (preds_sum / len(loaders)).numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, avg_preds)

    return total_loss / len(loaders), global_r2, avg_r2, per_r2, avg_preds, true_labels


@torch.inference_mode()
def valid_epoch_tta_ucvme(model_a, model_b, loaders, device):
    model_a.eval()
    model_b.eval()
    assert len(loaders) > 0

    n = len(loaders[0].dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)
    preds_sum = torch.zeros((n, n_targets), dtype=torch.float32)
    total_loss = 0.0

    for tta_i, loader in enumerate(loaders):
        offset = 0
        tta_loss = 0.0
        for batch in loader:
            l, r, lab, mask, meta, kpred, kmask, weight, group_id = _unpack_batch(batch)
            bs = l.size(0)
            l = l.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            if kpred is not None:
                kpred = kpred.to(device, non_blocking=True)
            if kmask is not None:
                kmask = kmask.to(device, non_blocking=True)
            if weight is not None:
                weight = weight.to(device, non_blocking=True)

            with autocast_ctx():
                pred_a, logvar_a, _, _ = _ucvme_forward_pack(model_a, l, r, kpred, kmask)
                pred_b, logvar_b, _, _ = _ucvme_forward_pack(model_b, l, r, kpred, kmask)
                pred_pack = 0.5 * (pred_a + pred_b)
                logvar_pack = None
                if logvar_a is not None and logvar_b is not None:
                    logvar_pack = 0.5 * (logvar_a + logvar_b)

                p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f = _unpack_pred_pack(pred_pack)
                if CFG.USE_HETERO and logvar_pack is not None:
                    loss_target = hetero_nll_loss(
                        pred_pack,
                        logvar_pack,
                        lab,
                        w=CFG.LOSS_WEIGHTS,
                        mask=mask,
                        sample_weight=weight,
                    )
                else:
                    loss_target = biomass_loss(
                        (p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f),
                        lab,
                        w=CFG.LOSS_WEIGHTS,
                        mask=mask,
                        sample_weight=weight,
                    )
                tta_loss += loss_target.detach().float().item() * bs

            batch_pred = pred_pack.float().cpu()
            preds_sum[offset:offset + bs] += batch_pred

            if tta_i == 0:
                batch_lab = _ensure_2d_lab(lab, bs).float().cpu()
                if batch_lab.shape[1] != n_targets:
                    raise RuntimeError(f"Label dim mismatch: {batch_lab.shape} vs {n_targets}")
                labels_cpu[offset:offset + bs] = batch_lab

            offset += bs

        total_loss += tta_loss / n

    avg_preds = (preds_sum / len(loaders)).numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, avg_preds)
    return total_loss / len(loaders), global_r2, avg_r2, per_r2, avg_preds, true_labels


@torch.inference_mode()
def valid_epoch_tta_ucvme_shared(model, loaders, device):
    model.eval()
    assert len(loaders) > 0

    n = len(loaders[0].dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)
    preds_sum = torch.zeros((n, n_targets), dtype=torch.float32)
    total_loss = 0.0

    for tta_i, loader in enumerate(loaders):
        offset = 0
        tta_loss = 0.0
        for batch in loader:
            l, r, lab, mask, meta, kpred, kmask, weight, group_id = _unpack_batch(batch)
            bs = l.size(0)
            l = l.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            if weight is not None:
                weight = weight.to(device, non_blocking=True)

            with autocast_ctx():
                out_a, out_b = model(l, r)
                pred_a, logvar_a, _, _ = _ucvme_pack_from_output(out_a, bs)
                pred_b, logvar_b, _, _ = _ucvme_pack_from_output(out_b, bs)
                pred_pack = 0.5 * (pred_a + pred_b)
                logvar_pack = None
                if logvar_a is not None and logvar_b is not None:
                    logvar_pack = 0.5 * (logvar_a + logvar_b)

                p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f = _unpack_pred_pack(pred_pack)
                if CFG.USE_HETERO and logvar_pack is not None:
                    loss_target = hetero_nll_loss(
                        pred_pack,
                        logvar_pack,
                        lab,
                        w=CFG.LOSS_WEIGHTS,
                        mask=mask,
                        sample_weight=weight,
                    )
                else:
                    loss_target = biomass_loss(
                        (p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f),
                        lab,
                        w=CFG.LOSS_WEIGHTS,
                        mask=mask,
                        sample_weight=weight,
                    )
                tta_loss += loss_target.detach().float().item() * bs

            batch_pred = pred_pack.float().cpu()
            preds_sum[offset:offset + bs] += batch_pred

            if tta_i == 0:
                batch_lab = _ensure_2d_lab(lab, bs).float().cpu()
                if batch_lab.shape[1] != n_targets:
                    raise RuntimeError(f"Label dim mismatch: {batch_lab.shape} vs {n_targets}")
                labels_cpu[offset:offset + bs] = batch_lab

            offset += bs

        total_loss += tta_loss / n

    avg_preds = (preds_sum / len(loaders)).numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, avg_preds)
    return total_loss / len(loaders), global_r2, avg_r2, per_r2, avg_preds, true_labels


def train_epoch_ucvme(
    model_a,
    model_b,
    loader_lb,
    loader_ulb,
    opt_a,
    opt_b,
    scheduler_a,
    scheduler_b,
    device,
    ema_a=None,
    ema_b=None,
    ulb_active=False,
):
    if not CFG.USE_HETERO:
        raise RuntimeError("UCVME requires USE_HETERO=True.")

    model_a.train()
    model_b.train()
    running = 0.0

    grad_acc = int(CFG.GRAD_ACC) * (int(CFG.UCVME_GRAD_ACC_MULT) if ulb_active else 1)
    opt_a.zero_grad(set_to_none=True)
    opt_b.zero_grad(set_to_none=True)
    ulb_iter = iter(loader_ulb) if (ulb_active and loader_ulb is not None) else None

    itera = tqdm(loader_lb, desc="train-ucvme", leave=False) if CFG.USE_TQDM else loader_lb
    for i, batch in enumerate(itera):
        l, r, lab, mask, meta, kpred, kmask, weight, group_id = _unpack_batch(batch)
        bs = l.size(0)
        l = l.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        meta = _move_meta(meta, device)
        if kpred is not None:
            kpred = kpred.to(device, non_blocking=True)
        if kmask is not None:
            kmask = kmask.to(device, non_blocking=True)
        if weight is not None:
            weight = weight.to(device, non_blocking=True)

        with autocast_ctx():
            pred_a, logvar_a, _, meta_a = _ucvme_forward_pack(
                model_a, l, r, kpred, kmask
            )
            pred_b, logvar_b, _, meta_b = _ucvme_forward_pack(
                model_b, l, r, kpred, kmask
            )

            loss_reg_a = hetero_nll_loss(
                pred_a,
                logvar_a,
                lab,
                w=CFG.LOSS_WEIGHTS,
                mask=mask,
                sample_weight=weight,
            )
            loss_reg_b = hetero_nll_loss(
                pred_b,
                logvar_b,
                lab,
                w=CFG.LOSS_WEIGHTS,
                mask=mask,
                sample_weight=weight,
            )
            loss_reg_lb = 0.5 * (loss_reg_a + loss_reg_b) * float(CFG.LOSS_WEIGHT_TARGETS)
            loss_unc_lb = _ucvme_uncertainty_consistency(logvar_a, logvar_b, mask)
            loss = loss_reg_lb
            if loss_unc_lb is not None:
                loss = loss + loss_unc_lb

            if CFG.UCVME_INCLUDE_META:
                loss_meta_a = compute_meta_loss(meta_a, meta)
                loss_meta_b = compute_meta_loss(meta_b, meta)
                if loss_meta_a is not None and loss_meta_b is not None:
                    loss = loss + 0.5 * (loss_meta_a + loss_meta_b)
                elif loss_meta_a is not None:
                    loss = loss + loss_meta_a
                elif loss_meta_b is not None:
                    loss = loss + loss_meta_b

            if ulb_iter is not None:
                try:
                    ulb_left, ulb_right = next(ulb_iter)
                except StopIteration:
                    ulb_iter = iter(loader_ulb)
                    ulb_left, ulb_right = next(ulb_iter)
                ulb_left = ulb_left.to(device, non_blocking=True)
                ulb_right = ulb_right.to(device, non_blocking=True)

                pseudo_y, pseudo_z = _ucvme_pseudo_labels(
                    model_a,
                    model_b,
                    ulb_left,
                    ulb_right,
                    int(CFG.UCVME_MC_SAMPLES),
                )

                pred_u_a, logvar_u_a, _, _ = _ucvme_forward_pack(model_a, ulb_left, ulb_right)
                pred_u_b, logvar_u_b, _, _ = _ucvme_forward_pack(model_b, ulb_left, ulb_right)
                loss_reg_ulb = 0.5 * (
                    _ucvme_reg_fixed_logvar(pred_u_a, pseudo_y, pseudo_z)
                    + _ucvme_reg_fixed_logvar(pred_u_b, pseudo_y, pseudo_z)
                )
                loss_unc_ulb = 0.5 * (
                    _ucvme_unc_to_target(logvar_u_a, pseudo_z)
                    + _ucvme_unc_to_target(logvar_u_b, pseudo_z)
                )
                loss = loss + float(CFG.UCVME_ULB_WEIGHT) * (loss_reg_ulb + loss_unc_ulb)

            loss = loss / float(grad_acc)

        if not torch.isfinite(loss):
            print("Non-finite loss detected; skipping batch")
            opt_a.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += loss.detach().float().item() * bs * float(grad_acc)

        do_step = ((i + 1) % grad_acc == 0) or ((i + 1) == len(loader_lb))
        if do_step:
            if scaler.is_enabled():
                scaler.unscale_(opt_a)
                scaler.unscale_(opt_b)
                torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
                scaler.step(opt_a)
                scaler.step(opt_b)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
                opt_a.step()
                opt_b.step()

            if ema_a is not None:
                ema_a.update(model_a.module if hasattr(model_a, "module") else model_a)
            if ema_b is not None:
                ema_b.update(model_b.module if hasattr(model_b, "module") else model_b)

            opt_a.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)

    scheduler_a.step()
    scheduler_b.step()
    return running / len(loader_lb.dataset)


def train_epoch_ucvme_shared(
    model,
    loader_lb,
    loader_ulb,
    optimizer,
    scheduler,
    device,
    ema=None,
    ulb_active=False,
):
    if not CFG.USE_HETERO:
        raise RuntimeError("UCVME requires USE_HETERO=True.")

    model.train()
    running = 0.0

    grad_acc = int(CFG.GRAD_ACC) * (int(CFG.UCVME_GRAD_ACC_MULT) if ulb_active else 1)
    optimizer.zero_grad(set_to_none=True)
    ulb_iter = iter(loader_ulb) if (ulb_active and loader_ulb is not None) else None

    itera = tqdm(loader_lb, desc="train-ucvme-shared", leave=False) if CFG.USE_TQDM else loader_lb
    for i, batch in enumerate(itera):
        l, r, lab, mask, meta, kpred, kmask, weight, group_id = _unpack_batch(batch)
        bs = l.size(0)
        l = l.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        meta = _move_meta(meta, device)
        if weight is not None:
            weight = weight.to(device, non_blocking=True)

        with autocast_ctx():
            out_a, out_b = model(l, r)
            pred_a, logvar_a, _, meta_a = _ucvme_pack_from_output(out_a, bs)
            pred_b, logvar_b, _, meta_b = _ucvme_pack_from_output(out_b, bs)

            loss_reg_a = hetero_nll_loss(
                pred_a,
                logvar_a,
                lab,
                w=CFG.LOSS_WEIGHTS,
                mask=mask,
                sample_weight=weight,
            )
            loss_reg_b = hetero_nll_loss(
                pred_b,
                logvar_b,
                lab,
                w=CFG.LOSS_WEIGHTS,
                mask=mask,
                sample_weight=weight,
            )
            loss_reg_lb = 0.5 * (loss_reg_a + loss_reg_b) * float(CFG.LOSS_WEIGHT_TARGETS)
            loss_unc_lb = _ucvme_uncertainty_consistency(logvar_a, logvar_b, mask)
            loss = loss_reg_lb
            if loss_unc_lb is not None:
                loss = loss + loss_unc_lb

            if CFG.UCVME_INCLUDE_META:
                loss_meta_a = compute_meta_loss(meta_a, meta)
                loss_meta_b = compute_meta_loss(meta_b, meta)
                if loss_meta_a is not None and loss_meta_b is not None:
                    loss = loss + 0.5 * (loss_meta_a + loss_meta_b)
                elif loss_meta_a is not None:
                    loss = loss + loss_meta_a
                elif loss_meta_b is not None:
                    loss = loss + loss_meta_b

            if ulb_iter is not None:
                try:
                    ulb_left, ulb_right = next(ulb_iter)
                except StopIteration:
                    ulb_iter = iter(loader_ulb)
                    ulb_left, ulb_right = next(ulb_iter)
                ulb_left = ulb_left.to(device, non_blocking=True)
                ulb_right = ulb_right.to(device, non_blocking=True)

                pseudo_y, pseudo_z = _ucvme_pseudo_labels_shared(
                    model,
                    ulb_left,
                    ulb_right,
                    int(CFG.UCVME_MC_SAMPLES),
                )

                out_u_a, out_u_b = model(ulb_left, ulb_right)
                pred_u_a, logvar_u_a, _, _ = _ucvme_pack_from_output(out_u_a, ulb_left.size(0))
                pred_u_b, logvar_u_b, _, _ = _ucvme_pack_from_output(out_u_b, ulb_left.size(0))
                loss_reg_ulb = 0.5 * (
                    _ucvme_reg_fixed_logvar(pred_u_a, pseudo_y, pseudo_z)
                    + _ucvme_reg_fixed_logvar(pred_u_b, pseudo_y, pseudo_z)
                )
                loss_unc_ulb = 0.5 * (
                    _ucvme_unc_to_target(logvar_u_a, pseudo_z)
                    + _ucvme_unc_to_target(logvar_u_b, pseudo_z)
                )
                loss = loss + float(CFG.UCVME_ULB_WEIGHT) * (loss_reg_ulb + loss_unc_ulb)

            loss = loss / float(grad_acc)

        if not torch.isfinite(loss):
            print("Non-finite loss detected; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += loss.detach().float().item() * bs * float(grad_acc)

        do_step = ((i + 1) % grad_acc == 0) or ((i + 1) == len(loader_lb))
        if do_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

            optimizer.zero_grad(set_to_none=True)

    scheduler.step()
    return running / len(loader_lb.dataset)


def train_epoch(model, loader, opt, scheduler, device, ema: ModelEmaV2 | None = None, group_dro: GroupDRO | None = None):
    model.train()
    running = 0.0
    running_unweighted = 0.0 if CFG.PRINT_TRAIN_LOSS_DIAG else None

    opt.zero_grad(set_to_none=True)
    itera = tqdm(loader, desc='train', leave=False) if CFG.USE_TQDM else loader

    for i, batch in enumerate(itera):
        l, r, lab, mask, meta, kpred, kmask, weight, group_id = _unpack_batch(batch)
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        meta = _move_meta(meta, device)
        if kpred is not None:
            kpred = kpred.to(device, non_blocking=True)
        if kmask is not None:
            kmask = kmask.to(device, non_blocking=True)
        if weight is not None:
            weight = weight.to(device, non_blocking=True)
        if group_id is not None:
            group_id = group_id.to(device, non_blocking=True)

        with autocast_ctx():
            outputs = model(l, r)
            target_out, meta_out, feat, logvar_comp = _split_outputs(outputs)
            p_total, p_gdm, p_green, p_clover, p_dead = target_out
            pred_pack = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs)
            pred_pack = _apply_kernel_reg(pred_pack, kpred, kmask, model, feat)
            p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f = _unpack_pred_pack(pred_pack)
            if (
                CFG.USE_HETERO
                and CFG.USE_KERNEL_REG
                and CFG.KERNEL_REG_MODE == "mlp"
                and kpred is not None
                and hasattr(model, "head_green_logvar_kernel")
                and feat is not None
            ):
                mlp_in = torch.cat([feat, kpred], dim=1)
                lv_green = model.head_green_logvar_kernel(mlp_in)
                lv_clover = model.head_clover_logvar_kernel(mlp_in)
                lv_dead = model.head_dead_logvar_kernel(mlp_in)
                lv_gdm = model.head_gdm_logvar_kernel(mlp_in)
                lv_total = model.head_total_logvar_kernel(mlp_in)
                logvar_comp = torch.cat([lv_green, lv_dead, lv_clover, lv_gdm, lv_total], dim=1)
            logvar_pack = _build_logvar_pack(logvar_comp) if CFG.USE_HETERO else None
            if CFG.USE_HETERO:
                _check_nonfinite("logvar_comp", logvar_comp, "train")
                _check_nonfinite("logvar_pack", logvar_pack, "train")
            loss = None
            diag_loss = None
            if CFG.PREDICT_TARGETS:
                if CFG.USE_HETERO and logvar_pack is not None:
                    if group_dro is not None and group_id is not None:
                        per_sample = hetero_nll_loss_per_sample(
                            pred_pack,
                            logvar_pack,
                            lab,
                            w=CFG.LOSS_WEIGHTS,
                            mask=mask,
                            sample_weight=weight,
                        )
                        loss_target = group_dro.compute_loss(per_sample, group_id)
                    else:
                        loss_target = hetero_nll_loss(
                            pred_pack,
                            logvar_pack,
                            lab,
                            w=CFG.LOSS_WEIGHTS,
                            mask=mask,
                            sample_weight=weight,
                        )
                else:
                    if group_dro is not None and group_id is not None:
                        per_sample = biomass_loss_per_sample(
                            (p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f),
                            lab,
                            w=CFG.LOSS_WEIGHTS,
                            mask=mask,
                            sample_weight=weight,
                        )
                        loss_target = group_dro.compute_loss(per_sample, group_id)
                    else:
                        loss_target = biomass_loss((p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f),
                                                   lab, w=CFG.LOSS_WEIGHTS, mask=mask, sample_weight=weight)
                loss = loss_target * float(CFG.LOSS_WEIGHT_TARGETS)
                if CFG.PRINT_TRAIN_LOSS_DIAG:
                    if CFG.USE_HETERO and logvar_pack is not None:
                        diag_loss_target = hetero_nll_loss(
                            pred_pack,
                            logvar_pack,
                            lab,
                            w=CFG.LOSS_WEIGHTS,
                            mask=mask,
                            sample_weight=None,
                        )
                    else:
                        diag_loss_target = biomass_loss(
                            (p_total_f, p_gdm_f, p_green_f, p_clover_f, p_dead_f),
                            lab,
                            w=CFG.LOSS_WEIGHTS,
                            mask=mask,
                            sample_weight=None,
                        )
                    diag_loss = diag_loss_target * float(CFG.LOSS_WEIGHT_TARGETS)
            loss_meta = compute_meta_loss(meta_out, meta)
            if loss_meta is not None:
                loss = loss_meta if loss is None else loss + loss_meta
                if CFG.PRINT_TRAIN_LOSS_DIAG:
                    diag_loss = loss_meta if diag_loss is None else diag_loss + loss_meta
            if loss is None:
                loss = p_total.sum() * 0.0
            if CFG.PRINT_TRAIN_LOSS_DIAG:
                if diag_loss is None:
                    diag_loss = loss.detach() * 0.0
            loss = loss / CFG.GRAD_ACC
            _check_nonfinite("loss", loss, "train")


        if not torch.isfinite(loss):
            print("Non-finite loss detected; skipping batch")
            opt.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += loss.detach().float().item() * bs * CFG.GRAD_ACC
        if running_unweighted is not None and diag_loss is not None:
            running_unweighted += diag_loss.detach().float().item() * bs

        do_step = ((i + 1) % CFG.GRAD_ACC == 0) or ((i + 1) == len(loader))
        if do_step:
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

            opt.zero_grad(set_to_none=True)

    scheduler.step()
    avg_loss = running / len(loader.dataset)
    if running_unweighted is None:
        return avg_loss
    return avg_loss, (running_unweighted / len(loader.dataset))

def _resolve_split_series(df: pd.DataFrame, col_name: str):
    if not col_name:
        return None
    name = str(col_name)
    if name in df.columns:
        series = df[name]
    elif name.lower() == "dom":
        state, month = _extract_state_month(df)
        series = pd.Series([f"{s}_{m}" for s, m in zip(state, month)], index=df.index)
    else:
        return None
    series = series.copy()
    if pd.api.types.is_numeric_dtype(series):
        if series.isna().any():
            min_val = series.min()
            if not np.isfinite(min_val):
                min_val = 0.0
            series = series.fillna(min_val - 1)
    else:
        series = series.fillna("NA").astype(str)
    return series.to_numpy()

def _build_external_fold_maps(split_df: pd.DataFrame):
    fold_col = str(getattr(CFG, "EXTERNAL_FOLD_SPLITS_FOLD_COL", "fold"))
    image_col = str(getattr(CFG, "EXTERNAL_FOLD_SPLITS_IMAGE_COL", "image_id"))
    sample_col = str(getattr(CFG, "EXTERNAL_FOLD_SPLITS_SAMPLE_COL", "sample_id"))
    path_col = str(getattr(CFG, "EXTERNAL_FOLD_SPLITS_PATH_COL", "image_path"))

    if fold_col not in split_df.columns:
        raise ValueError(f"External fold file missing '{fold_col}' column.")

    fold_vals = pd.to_numeric(split_df[fold_col], errors="coerce")
    if fold_vals.isna().any():
        raise ValueError("External fold file has non-numeric fold values.")
    split_df = split_df.copy()
    split_df["_fold"] = fold_vals.astype(int)

    if image_col in split_df.columns:
        split_df["_image_id"] = split_df[image_col].astype(str)
    elif sample_col in split_df.columns:
        split_df["_image_id"] = split_df[sample_col].astype(str).str.split("__", n=1).str[0]
    elif path_col in split_df.columns:
        split_df["_image_id"] = split_df[path_col].astype(str).map(
            lambda p: os.path.splitext(os.path.basename(str(p)))[0]
        )
    else:
        raise ValueError(
            f"External fold file needs '{image_col}', '{sample_col}', or '{path_col}' column."
        )

    dup = split_df.groupby("_image_id")["_fold"].nunique()
    if (dup > 1).any():
        bad = dup[dup > 1].index[:5].tolist()
        raise ValueError(f"External fold file has conflicting folds for image_id(s): {bad}")

    id_map = (
        split_df.drop_duplicates("_image_id")
        .set_index("_image_id")["_fold"]
        .to_dict()
    )

    path_map = None
    if path_col in split_df.columns:
        path_map = (
            split_df.drop_duplicates(path_col)
            .set_index(path_col)["_fold"]
            .to_dict()
        )

    return id_map, path_map

def load_external_fold_splits(df: pd.DataFrame):
    path = getattr(CFG, "EXTERNAL_FOLD_SPLITS_CSV", None)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"External fold split CSV not found: {path}")

    split_df = pd.read_csv(path)
    id_map, path_map = _build_external_fold_maps(split_df)

    folds = []
    missing = 0
    for img_path in df["image_path"].astype(str).tolist():
        fold = None
        if path_map is not None:
            fold = path_map.get(img_path)
        if fold is None:
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            fold = id_map.get(image_id)
        if fold is None:
            missing += 1
            fold = -1
        folds.append(int(fold))

    folds = np.asarray(folds, dtype=np.int64)
    if missing > 0:
        msg = f"External fold split missing {missing}/{len(folds)} images."
        if getattr(CFG, "EXTERNAL_FOLD_SPLITS_STRICT", True):
            raise ValueError(msg)
        print(f"Warning: {msg} Falling back to auto splits.")
        return None

    if folds.min() < 0:
        raise ValueError("External fold split produced invalid fold indices.")

    splits = []
    for fold in range(int(CFG.N_FOLDS)):
        val_idx = np.where(folds == fold)[0]
        tr_idx = np.where(folds != fold)[0]
        splits.append((tr_idx, val_idx))
    return splits

def build_fold_splits(df: pd.DataFrame):
    if getattr(CFG, "USE_EXTERNAL_FOLD_SPLITS", False):
        splits = load_external_fold_splits(df)
        if splits is not None:
            return splits, "external", None, None

    split_mode = str(getattr(CFG, "VAL_SPLIT_MODE", "stratified_group")).lower()
    if split_mode in ("hard", "hard_group", "group_hard"):
        split_mode = "group"
    group_col = getattr(CFG, "VAL_SPLIT_GROUP_COL", "Sampling_Date")
    strat_col = getattr(CFG, "VAL_SPLIT_STRATIFY_COL", "State")
    groups = _resolve_split_series(df, group_col)
    y_stratify = _resolve_split_series(df, strat_col)

    if groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups < int(CFG.N_FOLDS):
            print(
                f"Split groups too few ({n_groups}) for {CFG.N_FOLDS} folds; "
                "disabling group-based split."
            )
            groups = None

    used_group = None
    used_strat = None
    if split_mode == "stratified_group":
        if groups is None or y_stratify is None:
            split_mode = "group" if groups is not None else ("stratified" if y_stratify is not None else "kfold")
        else:
            splitter = StratifiedGroupKFold(
                n_splits=CFG.N_FOLDS,
                shuffle=True,
                random_state=CFG.SEED,
            )
            used_group = group_col
            used_strat = strat_col
            return list(splitter.split(df, y_stratify, groups=groups)), split_mode, used_group, used_strat

    if split_mode == "group":
        if groups is None:
            split_mode = "stratified" if y_stratify is not None else "kfold"
        else:
            splitter = GroupKFold(n_splits=CFG.N_FOLDS)
            used_group = group_col
            return list(splitter.split(df, groups=groups)), split_mode, used_group, used_strat

    if split_mode == "stratified":
        if y_stratify is None:
            split_mode = "kfold"
        else:
            splitter = StratifiedKFold(
                n_splits=CFG.N_FOLDS,
                shuffle=True,
                random_state=CFG.SEED,
            )
            used_strat = strat_col
            return list(splitter.split(df, y_stratify)), split_mode, used_group, used_strat

    splitter = KFold(
        n_splits=CFG.N_FOLDS,
        shuffle=True,
        random_state=CFG.SEED,
    )
    return list(splitter.split(df)), "kfold", used_group, used_strat

# === MAIN TRAINING LOOP === #

# Helper for accurate GPU timings
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


print("Loading data...")
df_long = pd.read_csv(CFG.TRAIN_CSV)

df_wide = (
    df_long.pivot(index="image_path", columns="target_name", values="target")
    .reset_index()
)
assert df_wide["image_path"].is_unique, "Leakage risk: duplicate image_path rows"

# Merge metadata (date/state/species/ndvi/height) from long CSV
meta_cols = ["Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
available_meta_cols = [c for c in meta_cols if c in df_long.columns]
if available_meta_cols:
    print(f"Merging metadata for stratification: {available_meta_cols}")
    meta_df = df_long[["image_path"] + available_meta_cols].drop_duplicates()
    df_wide = df_wide.merge(meta_df, on="image_path", how="left")

# Keep necessary columns
keep_meta_cols = [c for c in meta_cols if c in df_wide.columns]
df_wide = df_wide[["image_path"] + keep_meta_cols + CFG.ALL_TARGET_COLS]
print(f"{len(df_wide)} training images")

META_INFO = build_meta_info(df_wide)
RETURN_META = bool(META_INFO.get("enabled"))
if RETURN_META:
    enabled = [k.replace("use_", "") for k in ["use_date", "use_state", "use_species", "use_ndvi", "use_height"] if META_INFO.get(k)]
    print(f"Meta prediction enabled: {enabled}")

KERNEL_TRAIN_EMB = None
KERNEL_TRAIN_LABELS = None
if CFG.USE_KERNEL_REG and CFG.PREDICT_TARGETS:
    KERNEL_TRAIN_LABELS = df_wide[CFG.ALL_TARGET_COLS].values.astype(np.float32)
    KERNEL_TRAIN_EMB = compute_kernel_embeddings(df_wide, CFG.TRAIN_IMAGE_DIR, CFG.DEVICE)

bad_mark_set = load_bad_marks_set()
if bad_mark_set:
    df_wide["is_bad"] = df_wide["image_path"].astype(str).map(
        lambda p: os.path.basename(p) in bad_mark_set
    )
    print(f"Bad marks flagged: {int(df_wide['is_bad'].sum())}")
else:
    df_wide["is_bad"] = False

train_tfm_presets = getattr(CFG, "TRAIN_TFMS", None)
if train_tfm_presets is None:
    train_tfm_presets = [getattr(CFG, "TRAIN_TFM", "aug")]
elif isinstance(train_tfm_presets, str):
    train_tfm_presets = [train_tfm_presets]
if not train_tfm_presets:
    train_tfm_presets = [getattr(CFG, "TRAIN_TFM", "aug")]

cutmix_modes = getattr(CFG, "CUTMIX_MODES", None)
if cutmix_modes is None:
    cutmix_modes = [getattr(CFG, "CUTMIX_MODE", "none")]
elif isinstance(cutmix_modes, str):
    cutmix_modes = [cutmix_modes]
if not cutmix_modes:
    cutmix_modes = [getattr(CFG, "CUTMIX_MODE", "none")]

if any(mode == "similar" for mode in cutmix_modes):
    df_wide["half_sim"] = compute_half_similarity(df_wide, CFG.TRAIN_IMAGE_DIR, CFG.DEVICE)
    print("Half similarity computed for cutmix.")

fold_splits, split_mode, split_group_col, split_strat_col = build_fold_splits(df_wide)
print(
    "Fold split:",
    f"mode={split_mode}",
    f"group={split_group_col or 'none'}",
    f"stratify={split_strat_col or 'none'}",
)

if getattr(CFG, "WRITE_FOLD_SPLITS", False):
    split_rows = []
    split_cols = ["image_path"]
    for col in ("Sampling_Date", "State", "Species", split_group_col, split_strat_col):
        if col and col in df_wide.columns and col not in split_cols:
            split_cols.append(col)
    base_df = df_wide[split_cols].reset_index(drop=True)
    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
        tr = base_df.iloc[tr_idx].copy()
        tr["fold"] = fold_idx
        tr["split"] = "train"
        val = base_df.iloc[val_idx].copy()
        val["fold"] = fold_idx
        val["split"] = "val"
        split_rows.append(tr)
        split_rows.append(val)
    if split_rows:
        fold_df = pd.concat(split_rows, ignore_index=True)
        fold_df.to_csv(CFG.FOLD_SPLITS_CSV, index=False)
        print(f"Fold splits saved to {CFG.FOLD_SPLITS_CSV}")

# One place for loader kwargs (fast)
DL_KW = dict(
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(CFG.NUM_WORKERS > 0),
    prefetch_factor=4 if CFG.NUM_WORKERS > 0 else None,
)

for tfm_name in train_tfm_presets:
    for cm_mode in cutmix_modes:
        run_model_dir = CFG.MODEL_DIR
        if len(train_tfm_presets) > 1:
            run_model_dir = os.path.join(run_model_dir, f"tfm_{tfm_name}")
        if len(cutmix_modes) > 1 or cm_mode != "none":
            run_model_dir = os.path.join(run_model_dir, f"cutmix_{cm_mode}")
        os.makedirs(run_model_dir, exist_ok=True)
        print(
            f"\n=== Training preset: {tfm_name} | Cutmix: {cm_mode} | "
            f"Model dir: {run_model_dir} ==="
        )

        oof_true, oof_pred, fold_summary = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(fold_splits):
            if fold not in CFG.FOLDS_TO_TRAIN:
                print(f"Skipping fold {fold} as per configuration.")
                continue

            print("\n" + "=" * 70)
            print(f"FOLD {fold + 1}/{CFG.N_FOLDS} | {len(tr_idx)} train / {len(val_idx)} val")
            print("=" * 70)

            # NOTE: avoid empty_cache/gc inside epoch loop; only between folds if you really want
            _sync()
            torch.cuda.empty_cache()
            gc.collect()

            tr_df_all = df_wide.iloc[tr_idx].copy()
            tr_df_all["_orig_idx"] = tr_df_all.index
            tr_df_all = tr_df_all.reset_index(drop=True)
            val_df = df_wide.iloc[val_idx].copy()
            val_df["_orig_idx"] = val_df.index
            val_df = val_df.reset_index(drop=True)
            bad_df = tr_df_all[tr_df_all["is_bad"]].reset_index(drop=True)
            if CFG.USE_BAD_MARKS_FILTER:
                tr_df = tr_df_all[~tr_df_all["is_bad"]].reset_index(drop=True)
                if len(tr_df_all) != len(tr_df):
                    print(f"Removed {len(tr_df_all) - len(tr_df)} bad-mark samples from training.")
            else:
                tr_df = tr_df_all

            weighting_mode = str(CFG.WEIGHTING_MODE).lower()
            use_manual_weights = weighting_mode in ("sampler", "loss", "sampler+loss")
            use_sampler = weighting_mode in ("sampler", "sampler+loss")
            use_loss_weight = weighting_mode in ("loss", "sampler+loss")
            use_group_dro = bool(CFG.USE_GROUP_DRO)
            if CFG.USE_UCVME and use_group_dro:
                print("GroupDRO disabled in UCVME mode.")
                use_group_dro = False
            if CFG.USE_UCVME and CFG.UCVME_SHARED_BACKBONE and CFG.USE_KERNEL_REG:
                raise RuntimeError("UCVME shared-backbone mode does not support kernel regression.")

            group_info = None
            train_weights = None
            train_group_ids = None
            val_group_ids = None
            group_dro = None
            vistamilk_group_id = None
            if use_manual_weights or use_group_dro or CFG.REPORT_GROUP_METRICS:
                group_info = build_group_info(tr_df, val_df)
                if use_manual_weights:
                    train_weights = group_info["weights"]
                    print(
                        f"Sample weights: min={train_weights.min():.3f} "
                        f"mean={train_weights.mean():.3f} max={train_weights.max():.3f}"
                    )
                if use_group_dro:
                    if str(CFG.GROUP_DRO_GROUP).lower() == "dom":
                        train_group_ids = group_info["train_dom_ids"]
                        num_groups = group_info["num_dom_groups"]
                    else:
                        train_group_ids = group_info["train_dro_ids"]
                        num_groups = group_info["num_dro_groups"]
                    if CFG.USE_VISTAMILK:
                        vistamilk_group_id = num_groups
                        num_groups += 1
                    group_dro = GroupDRO(
                        num_groups=num_groups,
                        eta=CFG.GROUP_DRO_ETA,
                        ema=CFG.GROUP_DRO_EMA,
                        device=CFG.DEVICE,
                    )
                    print(f"GroupDRO enabled: groups={num_groups} eta={CFG.GROUP_DRO_ETA}")
                if CFG.REPORT_GROUP_METRICS:
                    if str(CFG.GROUP_METRIC_GROUP).lower() == "dom":
                        val_group_ids = group_info["val_dom_ids"]
                    else:
                        val_group_ids = group_info["val_dro_ids"]

            kernel_train_pred = None
            kernel_val_pred = None
            kernel_train_mask = None
            kernel_val_mask = None
            if CFG.USE_KERNEL_REG and CFG.PREDICT_TARGETS and KERNEL_TRAIN_EMB is not None:
                tr_orig_idx = tr_df["_orig_idx"].to_numpy()
                val_orig_idx = val_df["_orig_idx"].to_numpy()
                train_emb = KERNEL_TRAIN_EMB[tr_orig_idx]
                train_labels = KERNEL_TRAIN_LABELS[tr_orig_idx]
                kernel_train_pred = kernel_regression_predict(
                    train_emb=train_emb,
                    train_labels=train_labels,
                    query_emb=train_emb,
                    device=CFG.DEVICE,
                    topk=CFG.KERNEL_REG_TOPK,
                    tau=CFG.KERNEL_REG_TAU,
                    batch_size=CFG.KERNEL_REG_PRED_BATCH_SIZE,
                    exclude_self=CFG.KERNEL_REG_EXCLUDE_SELF,
                    train_indices=tr_orig_idx,
                    query_indices=tr_orig_idx,
                )
                kernel_val_pred = kernel_regression_predict(
                    train_emb=train_emb,
                    train_labels=train_labels,
                    query_emb=KERNEL_TRAIN_EMB[val_orig_idx],
                    device=CFG.DEVICE,
                    topk=CFG.KERNEL_REG_TOPK,
                    tau=CFG.KERNEL_REG_TAU,
                    batch_size=CFG.KERNEL_REG_PRED_BATCH_SIZE,
                    exclude_self=False,
                    train_indices=tr_orig_idx,
                    query_indices=val_orig_idx,
                )
                kernel_train_mask = np.ones((len(tr_df), 1), dtype=np.float32)
                kernel_val_mask = np.ones((len(val_df), 1), dtype=np.float32)
                print(
                    f"Kernel regression ready | train={len(tr_df)} val={len(val_df)} "
                    f"topk={CFG.KERNEL_REG_TOPK} tau={CFG.KERNEL_REG_TAU}"
                )

            cutmix_prob = 0.0
            cutmix_similar_indices = None
            if cm_mode != "none":
                cutmix_prob = float(CFG.CUTMIX_PROB)
                if cm_mode == "similar":
                    if "half_sim" not in tr_df.columns:
                        raise RuntimeError("half_sim missing for similar cutmix.")
                    sims = tr_df["half_sim"].to_numpy()
                    if sims.size > 0:
                        thresh = np.quantile(sims, 1.0 - CFG.CUTMIX_SIMILAR_TOP_PCT)
                        cutmix_similar_indices = np.flatnonzero(sims >= thresh)
                    else:
                        cutmix_similar_indices = np.array([], dtype=np.int64)
                    if CFG.CUTMIX_MATCH_SIMILAR:
                        top_pct = max(CFG.CUTMIX_SIMILAR_TOP_PCT, 1e-6)
                        cutmix_prob = min(1.0, cutmix_prob / top_pct)
                    print(
                        f"Cutmix similar pool: {len(cutmix_similar_indices)}/{len(tr_df)} "
                        f"| prob={cutmix_prob:.3f}"
                    )
                else:
                    print(f"Cutmix random prob={cutmix_prob:.3f}")

            return_weight = use_loss_weight or CFG.USE_VISTAMILK
            csiro_weight = None
            if return_weight:
                if use_loss_weight and train_weights is not None:
                    csiro_weight = train_weights
                else:
                    csiro_weight = np.ones((len(tr_df),), dtype=np.float32)
            csiro_group = train_group_ids if use_group_dro else None

            tr_set = BiomassDataset(
                tr_df,
                get_train_transforms(tfm_name),
                CFG.TRAIN_IMAGE_DIR,
                cutmix_mode=cm_mode,
                cutmix_prob=cutmix_prob,
                cutmix_alpha=CFG.CUTMIX_ALPHA,
                cutmix_similar_indices=cutmix_similar_indices,
                seed=CFG.SEED + fold,
                return_mask=True,
                meta_info=META_INFO,
                return_meta=RETURN_META,
                kernel_preds=kernel_train_pred,
                kernel_mask=kernel_train_mask,
                return_kernel=CFG.USE_KERNEL_REG,
                sample_weight=csiro_weight,
                return_weight=return_weight,
                group_ids=csiro_group,
                return_group=use_group_dro,
            )
            use_vistamilk_labelled = CFG.USE_VISTAMILK and not (
                CFG.USE_UCVME and CFG.UCVME_USE_VISTAMILK
            )
            tr_sets = [tr_set]
            if use_vistamilk_labelled:
                use_weight = float(CFG.VISTAMILK_SAMPLE_WEIGHT)
                tr_sets.extend(build_vistamilk_datasets(
                    get_train_transforms(tfm_name),
                    META_INFO,
                    RETURN_META,
                    return_kernel=CFG.USE_KERNEL_REG,
                    return_weight=return_weight,
                    sample_weight=use_weight,
                    return_group=use_group_dro,
                    group_id_value=vistamilk_group_id,
                ))
            tr_set = tr_sets[0]
            if len(tr_sets) > 1:
                tr_set = torch.utils.data.ConcatDataset(tr_sets)

            sampler = None
            if use_sampler:
                weight_parts = []
                if train_weights is None:
                    train_weights = np.ones((len(tr_df),), dtype=np.float32)
                weight_parts.append(train_weights)
                if use_vistamilk_labelled:
                    for ds in tr_sets[1:]:
                        weight_parts.append(
                            np.full((len(ds),), float(CFG.VISTAMILK_SAMPLE_WEIGHT), dtype=np.float32)
                        )
                sampler_weights = np.concatenate(weight_parts, axis=0)
                if len(sampler_weights) != len(tr_set):
                    raise ValueError("Sampler weights length mismatch with training dataset.")
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sampler_weights, dtype=torch.double),
                    num_samples=len(sampler_weights),
                    replacement=CFG.WEIGHTING_SAMPLER_REPLACEMENT,
                )

            # Create TTA loaders (keep TTAs as requested)
            val_loaders = []
            for mode in range(CFG.VAL_TTA_TIMES):  # 0: orig, 1: hflip, 2: vflip, 3: rot90
                val_set_tta = BiomassDataset(
                    val_df,
                    get_tta_transforms(mode),
                    CFG.TRAIN_IMAGE_DIR,
                    return_mask=True,
                    meta_info=META_INFO,
                    return_meta=RETURN_META,
                    kernel_preds=kernel_val_pred,
                    kernel_mask=kernel_val_mask,
                    return_kernel=CFG.USE_KERNEL_REG,
                )
                val_loader_tta = DataLoader(
                    val_set_tta,
                    batch_size=CFG.BATCH_SIZE,
                    shuffle=False,
                    drop_last=False,
                    **{k: v for k, v in DL_KW.items() if v is not None},
                )
                val_loaders.append(val_loader_tta)

            tr_loader = DataLoader(
                tr_set,
                batch_size=CFG.BATCH_SIZE,
                shuffle=(sampler is None),
                sampler=sampler,
                drop_last=True,
                **{k: v for k, v in DL_KW.items() if v is not None},
            )

            ulb_loader = None
            if CFG.USE_UCVME:
                ulb_sources = collect_ucvme_unlabeled_sources()
                if ulb_sources:
                    ulb_sets = []
                    for paths, mode in ulb_sources:
                        ulb_sets.append(
                            BiomassUnlabeledDataset(
                                paths,
                                get_train_transforms(tfm_name),
                                pair_mode=mode,
                            )
                        )
                    ulb_set = ulb_sets[0]
                    if len(ulb_sets) > 1:
                        ulb_set = torch.utils.data.ConcatDataset(ulb_sets)
                    ulb_loader = DataLoader(
                        ulb_set,
                        batch_size=CFG.BATCH_SIZE,
                        shuffle=True,
                        drop_last=True,
                        **{k: v for k, v in DL_KW.items() if v is not None},
                    )
                    total_ulb = sum(len(ds) for ds in ulb_sets)
                    print(f"UCVME unlabeled samples: {total_ulb}")

            print("Building model...")
            backbone_path = getattr(CFG, "BACKBONE_PATH", None)
            if CFG.USE_UCVME:
                if CFG.UCVME_SHARED_BACKBONE:
                    model_shared = UCVMESharedModel(
                        CFG.MODEL_NAME,
                        pretrained=CFG.PRETRAINED,
                        backbone_path=backbone_path,
                        meta_info=META_INFO,
                    ).to(CFG.DEVICE)
                else:
                    model_a = BiomassModel(
                        CFG.MODEL_NAME,
                        pretrained=CFG.PRETRAINED,
                        backbone_path=backbone_path,
                        meta_info=META_INFO,
                    ).to(CFG.DEVICE)
                    model_b = BiomassModel(
                        CFG.MODEL_NAME,
                        pretrained=CFG.PRETRAINED,
                        backbone_path=backbone_path,
                        meta_info=META_INFO,
                    ).to(CFG.DEVICE)

                def _load_pretrained(model, path, label):
                    if not os.path.exists(path):
                        return False
                    try:
                        state = torch.load(path, map_location="cpu")
                        if isinstance(state, dict) and ("model_state_dict" in state or "state_dict" in state):
                            key = "model_state_dict" if "model_state_dict" in state else "state_dict"
                            sd = state[key]
                        else:
                            sd = state
                        model.load_state_dict(sd, strict=False)
                        model.to(CFG.DEVICE)
                        print(f"  ✓ Loaded pretrained weights ({label}) from {path}")
                        return True
                    except Exception as e:
                        print(f"  ✗ Failed to load pretrained ({label}) from {path}: {e}")
                        return False

                if getattr(CFG, "PRETRAINED_DIR", None) and os.path.isdir(CFG.PRETRAINED_DIR):
                    path_base = os.path.join(CFG.PRETRAINED_DIR, f"best_model_fold{fold}.pth")
                    path_a = os.path.join(CFG.PRETRAINED_DIR, f"best_model_fold{fold}_a.pth")
                    path_b = os.path.join(CFG.PRETRAINED_DIR, f"best_model_fold{fold}_b.pth")
                    path_shared = os.path.join(CFG.PRETRAINED_DIR, f"best_model_fold{fold}_shared.pth")
                    if CFG.UCVME_SHARED_BACKBONE:
                        loaded_shared = _load_pretrained(model_shared, path_shared, "shared") or _load_pretrained(model_shared, path_base, "shared")
                        if not loaded_shared:
                            print(f"  (No pretrained file for shared model at {path_shared} or {path_base})")
                    else:
                        loaded_a = _load_pretrained(model_a, path_a, "a") or _load_pretrained(model_a, path_base, "a")
                        loaded_b = _load_pretrained(model_b, path_b, "b") or _load_pretrained(model_b, path_base, "b")
                        if not loaded_a:
                            print(f"  (No pretrained file for model a at {path_a} or {path_base})")
                        if not loaded_b:
                            print(f"  (No pretrained file for model b at {path_b} or {path_base})")
                else:
                    print("  (No PRETRAINED_DIR configured or directory missing)")

                if CFG.UCVME_SHARED_BACKBONE:
                    set_backbone_requires_grad(model_shared, False)
                    optimizer_shared = build_optimizer(model_shared)
                    scheduler_shared = build_scheduler(optimizer_shared)
                    ema_shared = ModelEmaV2(model_shared, decay=CFG.EMA_DECAY)
                else:
                    set_backbone_requires_grad(model_a, False)
                    set_backbone_requires_grad(model_b, False)

                    optimizer_a = build_optimizer(model_a)
                    scheduler_a = build_scheduler(optimizer_a)
                    optimizer_b = build_optimizer(model_b)
                    scheduler_b = build_scheduler(optimizer_b)

                    ema_a = ModelEmaV2(model_a, decay=CFG.EMA_DECAY)
                    ema_b = ModelEmaV2(model_b, decay=CFG.EMA_DECAY)
            else:
                model = BiomassModel(
                    CFG.MODEL_NAME,
                    pretrained=CFG.PRETRAINED,
                    backbone_path=backbone_path,
                    meta_info=META_INFO,
                ).to(CFG.DEVICE)

                # Load pretrained fold weights if available (for resuming or fine-tuning)
                if getattr(CFG, "PRETRAINED_DIR", None) and os.path.isdir(CFG.PRETRAINED_DIR):
                    pretrained_path = os.path.join(CFG.PRETRAINED_DIR, f"best_model_fold{fold}.pth")
                    if os.path.exists(pretrained_path):
                        try:
                            state = torch.load(pretrained_path, map_location="cpu")
                            if isinstance(state, dict) and ("model_state_dict" in state or "state_dict" in state):
                                key = "model_state_dict" if "model_state_dict" in state else "state_dict"
                                sd = state[key]
                            else:
                                sd = state
                            model.load_state_dict(sd, strict=False)
                            model.to(CFG.DEVICE)
                            print(f"  ✓ Loaded pretrained weights for fold {fold} from {pretrained_path}")
                        except Exception as e:
                            print(f"  ✗ Failed to load pretrained fold {fold}: {e}")
                    else:
                        print(f"  (No pretrained file for fold {fold} at {pretrained_path})")
                else:
                    print("  (No PRETRAINED_DIR configured or directory missing)")

                # Freeze/unfreeze backbone
                set_backbone_requires_grad(model, False)

                optimizer = build_optimizer(model)
                scheduler = build_scheduler(optimizer)

                # EMA on the real model
                ema = ModelEmaV2(model, decay=CFG.EMA_DECAY)

            best_global_r2 = -np.inf
            best_avg_r2 = -np.inf
            best_score = -np.inf
            patience = 0
            best_fold_preds = None
            best_fold_true = None
            ucvme_ulb_active = False
            save_path = os.path.join(run_model_dir, f"best_model_fold{fold}.pth")
            save_path_a = os.path.join(run_model_dir, f"best_model_fold{fold}_a.pth")
            save_path_b = os.path.join(run_model_dir, f"best_model_fold{fold}_b.pth")
            save_path_shared = os.path.join(run_model_dir, f"best_model_fold{fold}_shared.pth")

            save_metric = str(getattr(CFG, "SAVE_METRIC", "global")).lower()
            save_worst_w = float(getattr(CFG, "SAVE_SCORE_WORST", 0.7))
            save_global_w = float(getattr(CFG, "SAVE_SCORE_GLOBAL", 0.3))
            warned_no_group_metric = False

            for epoch in range(1, CFG.EPOCHS + 1):
                if epoch == CFG.FREEZE_EPOCHS + 1:
                    patience = 0
                    if CFG.USE_UCVME:
                        if CFG.UCVME_SHARED_BACKBONE:
                            set_backbone_requires_grad(model_shared, True)
                        else:
                            set_backbone_requires_grad(model_a, True)
                            set_backbone_requires_grad(model_b, True)
                    else:
                        set_backbone_requires_grad(model, True)
                    print(f"Epoch {epoch}: backbone unfrozen")

                # ---- Train timing ----
                _sync()
                t0 = time.perf_counter()
                if CFG.USE_UCVME:
                    if CFG.UCVME_SHARED_BACKBONE:
                        tr_loss = train_epoch_ucvme_shared(
                            model_shared,
                            tr_loader,
                            ulb_loader,
                            optimizer_shared,
                            scheduler_shared,
                            CFG.DEVICE,
                            ema=ema_shared,
                            ulb_active=ucvme_ulb_active,
                        )
                    else:
                        tr_loss = train_epoch_ucvme(
                            model_a,
                            model_b,
                            tr_loader,
                            ulb_loader,
                            optimizer_a,
                            optimizer_b,
                            scheduler_a,
                            scheduler_b,
                            CFG.DEVICE,
                            ema_a=ema_a,
                            ema_b=ema_b,
                            ulb_active=ucvme_ulb_active,
                        )
                    tr_loss_unweighted = None
                else:
                    tr_out = train_epoch(model, tr_loader, optimizer, scheduler, CFG.DEVICE, ema, group_dro=group_dro)
                    if isinstance(tr_out, tuple):
                        tr_loss, tr_loss_unweighted = tr_out
                    else:
                        tr_loss, tr_loss_unweighted = tr_out, None
                _sync()
                t1 = time.perf_counter()

                if CFG.USE_UCVME:
                    if CFG.UCVME_SHARED_BACKBONE:
                        eval_model_shared = ema_shared.module if ema_shared is not None else model_shared
                    else:
                        eval_model_a = ema_a.module if ema_a is not None else model_a
                        eval_model_b = ema_b.module if ema_b is not None else model_b
                else:
                    eval_model = ema.module if ema is not None else model

                # ---- Val timing (TTA) ----
                _sync()
                t2 = time.perf_counter()
                if CFG.USE_UCVME:
                    if CFG.UCVME_SHARED_BACKBONE:
                        val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta_ucvme_shared(
                            eval_model_shared,
                            val_loaders,
                            CFG.DEVICE,
                        )
                    else:
                        val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta_ucvme(
                            eval_model_a,
                            eval_model_b,
                            val_loaders,
                            CFG.DEVICE,
                        )
                else:
                    val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta(
                        eval_model,
                        val_loaders,
                        CFG.DEVICE,
                    )
                _sync()
                t3 = time.perf_counter()

                time_tr = t1 - t0
                time_val = t3 - t2
                time_ep = t3 - t0

                per_r2_str = " | ".join(
                    [f"{CFG.ALL_TARGET_COLS[i][:5]}: {r2:.3f}" for i, r2 in enumerate(per_r2)]
                )
                if CFG.USE_UCVME:
                    if CFG.UCVME_SHARED_BACKBONE:
                        lrs_s = [pg["lr"] for pg in optimizer_shared.param_groups]
                        lr_str = " ".join([f"lrs{i}={lr:.3e}" for i, lr in enumerate(lrs_s)])
                    else:
                        lrs_a = [pg["lr"] for pg in optimizer_a.param_groups]
                        lrs_b = [pg["lr"] for pg in optimizer_b.param_groups]
                        lr_str = " ".join([f"lra{i}={lr:.3e}" for i, lr in enumerate(lrs_a)])
                        lr_str += " " + " ".join([f"lrb{i}={lr:.3e}" for i, lr in enumerate(lrs_b)])
                else:
                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    lr_str = " ".join([f"lr{i}={lr:.3e}" for i, lr in enumerate(lrs)])

                group_str = ""
                macro = None
                worst = None
                n_groups = 0
                if CFG.REPORT_GROUP_METRICS and val_group_ids is not None:
                    macro, worst, n_groups = compute_group_r2_metrics(
                        true_fold,
                        preds_fold,
                        val_group_ids,
                        CFG.GROUP_METRIC_MIN_SAMPLES,
                    )
                    if macro is not None:
                        group_str = f" | GroupR2 macro={macro:.4f} worst={worst:.4f} (n={n_groups})"

                score = global_r2
                if save_metric in ("avg", "avg_r2"):
                    score = avg_r2
                elif save_metric in ("robust", "group", "group_worst"):
                    if worst is not None:
                        score = save_worst_w * worst + save_global_w * global_r2
                    else:
                        if not warned_no_group_metric:
                            print(
                                "  ! SAVE_METRIC=robust but group metrics unavailable; "
                                "falling back to GlobalR²."
                            )
                            warned_no_group_metric = True
                        score = global_r2

                save_score_str = f" | SaveScore {score:.4f}"
                loss_diag_str = ""
                if tr_loss_unweighted is not None:
                    loss_diag_str = f" | TUnw {tr_loss_unweighted:.5f}"
                phase_note = " | ULB" if ucvme_ulb_active else ""
                print(
                    f"Fold {fold} | Epoch {epoch:02d} | "
                    f"TLoss {tr_loss:.5f}{loss_diag_str} | VLoss {val_loss:.5f} | "
                    f"avgR2 {avg_r2:.4f} | GlobalR² {global_r2:.4f}"
                    f"{save_score_str}{group_str}{phase_note} "
                    f'{"[BEST]" if score > best_score else ""} | '
                    f"{lr_str} | time_tr={time_tr:.1f}s time_val={time_val:.1f}s time_ep={time_ep:.1f}s"
                )
                print(f"  → {per_r2_str}")

                if score > best_score:
                    best_score = score
                    best_global_r2 = global_r2
                    best_avg_r2 = avg_r2

                    if CFG.USE_UCVME:
                        if CFG.UCVME_SHARED_BACKBONE:
                            torch.save(eval_model_shared.state_dict(), save_path_shared)
                            print(
                                f"  → SAVED EMA weights to {save_path_shared} "
                                f"(SaveScore: {best_score:.4f}, GlobalR²: {best_global_r2:.4f})"
                            )
                        else:
                            torch.save(eval_model_a.state_dict(), save_path_a)
                            torch.save(eval_model_b.state_dict(), save_path_b)
                            print(
                                f"  → SAVED EMA weights to {save_path_a} / {save_path_b} "
                                f"(SaveScore: {best_score:.4f}, GlobalR²: {best_global_r2:.4f})"
                            )
                    else:
                        torch.save(eval_model.state_dict(), save_path)
                        print(
                            f"  → SAVED EMA weights to {save_path} "
                            f"(SaveScore: {best_score:.4f}, GlobalR²: {best_global_r2:.4f})"
                        )

                    patience = 0
                    best_fold_preds = preds_fold
                    best_fold_true = true_fold
                else:
                    patience += 1
                    if (
                        CFG.USE_UCVME
                        and CFG.UCVME_ENABLE_AFTER_PATIENCE
                        and not ucvme_ulb_active
                        and ulb_loader is not None
                        and patience >= CFG.PATIENCE
                    ):
                        ucvme_ulb_active = True
                        patience = 0
                        print("  → UCVME unlabeled phase enabled")
                    elif patience >= CFG.PATIENCE:
                        print(f"  → EARLY STOP (no improvement in {CFG.PATIENCE} epochs)")
                        break

                # keep memory tidy but avoid heavy cache/gc churn
                del preds_fold, true_fold

            if best_fold_preds is not None:
                oof_true.append(best_fold_true)
                oof_pred.append(best_fold_preds)
                fold_entry = {
                    "fold": fold,
                    "save_score": best_score,
                    "global_r2": best_global_r2,
                    "avg_r2": best_avg_r2,
                }
                fold_summary.append(fold_entry)

            if CFG.USE_UCVME:
                eval_path = save_path_shared if CFG.UCVME_SHARED_BACKBONE else save_path_a
            else:
                eval_path = save_path
            if CFG.EVAL_BAD_MARKS and bad_mark_set and os.path.exists(eval_path):
                if CFG.USE_UCVME and CFG.UCVME_SHARED_BACKBONE:
                    print("Skipping bad-mark evaluation for UCVME shared-backbone model.")
                else:
                    print(f"\nEvaluating best checkpoint on bad-mark samples (fold {fold})...")
                    eval_model_ckpt = BiomassModel(
                        CFG.MODEL_NAME,
                        pretrained=False,
                        backbone_path=backbone_path,
                        meta_info=META_INFO,
                    )
                    state = torch.load(eval_path, map_location="cpu")
                    eval_model_ckpt.load_state_dict(state, strict=False)
                    eval_model_ckpt.to(CFG.DEVICE)
                    eval_model_ckpt.eval()

                    evaluate_bad_marks(
                        eval_model_ckpt,
                        bad_df,
                        CFG.TRAIN_IMAGE_DIR,
                        title=f"Bad-mark samples fold {fold}",
                        print_limit=CFG.BAD_MARKS_PRINT_LIMIT,
                        meta_info=META_INFO,
                        return_meta=RETURN_META,
                    )

                    val_print_limit = CFG.BAD_MARKS_PRINT_LIMIT if CFG.BAD_MARKS_PRINT_VAL else None
                    evaluate_bad_marks(
                        eval_model_ckpt,
                        val_df,
                        CFG.TRAIN_IMAGE_DIR,
                        title=f"Validation fold {fold} (best checkpoint)",
                        print_limit=val_print_limit,
                        meta_info=META_INFO,
                        return_meta=RETURN_META,
                    )

                    del eval_model_ckpt
                    _sync()
                    torch.cuda.empty_cache()
                    gc.collect()

            # Cleanup for this fold
            if CFG.USE_UCVME:
                if CFG.UCVME_SHARED_BACKBONE:
                    del model_shared, optimizer_shared, scheduler_shared, ema_shared, eval_model_shared
                else:
                    del model_a, model_b, optimizer_a, optimizer_b, scheduler_a, scheduler_b, ema_a, ema_b, eval_model_a, eval_model_b
            else:
                del model, optimizer, scheduler, ema, eval_model
            del tr_loader, val_loaders
            _sync()
            torch.cuda.empty_cache()
            gc.collect()

        if oof_true:
            oof_true_arr = np.concatenate(oof_true, axis=0)
            oof_pred_arr = np.concatenate(oof_pred, axis=0)
            oof_global_r2, oof_avg_r2, oof_per_r2 = weighted_r2_score_global(oof_true_arr, oof_pred_arr)

            print(f"\nTraining complete ({tfm_name}, {cm_mode})! Models saved in:", run_model_dir)
            print("Fold summary:")
            for fs in fold_summary:
                print(
                    f"  Fold {fs['fold']}: SaveScore = {fs.get('save_score', float('nan')):.4f}, "
                    f"Global R² = {fs['global_r2']:.4f}, Avg R² = {fs.get('avg_r2', float('nan')):.4f}"
                )
            print(f"OOF Global Weighted R²: {oof_global_r2:.4f} | OOF Avg Target R²: {oof_avg_r2:.4f}")
            print("OOF Per-target:", dict(zip(CFG.ALL_TARGET_COLS, [f"{r:.4f}" for r in oof_per_r2])))
        else:
            print(f"No OOF predictions collected for preset: {tfm_name}, cutmix={cm_mode}")

# Inference (NO TTA)

# ===============================================================
# 5. CREATE TEST DATASET
# ===============================================================
def clean_image(img):
    # Safe crop (remove bottom artifacts) + inpaint orange date stamp
    h, w = img.shape[:2]
    img = img[0:int(h * 0.90), :]
    return _inpaint_orange_stamp(img)

class BiomassTestDataset(Dataset):
    """
    Test dataset for biomass images.
    Splits each 2000×1000 image into left and right 1000×1000 halves.
    """
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.filenames = [os.path.basename(p) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # Read image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}, using blank image")
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)

        # Split into left and right halves
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid].copy()
        right = img[:, mid:].copy()

        return left, right, self.filenames[idx]

print("✓ Test dataset class defined")

# ===============================================================
# 6. DEFINE INFERENCE TRANSFORM (NO TTA)
# ===============================================================
def get_inference_transform():
    """
    Single deterministic inference pipeline (no TTA).
    """
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

print("✓ Inference transform defined (no TTA)")

# ===============================================================
# 8. RUN INFERENCE (NO TTA) (UPDATED to honor CFG.FOLDS_TO_TRAIN)
# ===============================================================
@torch.no_grad()
def predict_single(model, left_np, right_np, transform, kernel_pred=None, return_pack=False, return_logvar=False):
    """
    Predict using a SINGLE deterministic view (no TTA).

    Args:
        model: Single trained model
        left_np: Left half of image (numpy array, HWC RGB uint8/float)
        right_np: Right half of image (numpy array, HWC RGB uint8/float)
        transform: Albumentations Compose

    Returns:
        numpy array: [total, gdm, green] predictions
    """
    left_tensor = transform(image=left_np)['image'].unsqueeze(0).to(CFG.DEVICE)
    right_tensor = transform(image=right_np)['image'].unsqueeze(0).to(CFG.DEVICE)

    outputs = model(left_tensor, right_tensor)
    target_out, _, feat, logvar_comp = _split_outputs(outputs)
    total, gdm, green, clover, dead = target_out

    pred_pack = _pred_pack(total, gdm, green, clover, dead, 1)
    if kernel_pred is not None:
        kpred = torch.as_tensor(kernel_pred, device=pred_pack.device, dtype=pred_pack.dtype).view(1, -1)
        kmask = torch.ones((1, 1), device=pred_pack.device, dtype=pred_pack.dtype)
        pred_pack = _apply_kernel_reg(pred_pack, kpred, kmask, model, feat)
    if return_pack:
        if return_logvar and CFG.USE_HETERO:
            logvar_pack = _build_logvar_pack(logvar_comp)
            return pred_pack.cpu().numpy()[0], (logvar_pack.cpu().numpy()[0] if logvar_pack is not None else None)
        return pred_pack.cpu().numpy()[0]

    pt, pgdm, pg, _, _ = _unpack_pred_pack(pred_pack)
    return np.array(
        [pt.cpu().item(), pgdm.cpu().item(), pg.cpu().item()],
        dtype=np.float32
    )


def run_inference():
    """
    Main inference function.
    Returns: (predictions_array, image_filenames)

    Notes:
      - Respects `CFG.FOLDS_TO_TRAIN` (if set) and averages only over successfully loaded folds.
      - If no fold weights are found for the requested folds, an error is raised.
      - No TTA: one forward pass per image per fold.
    """
    print("\n" + "="*70)
    print("STARTING INFERENCE (NO TTA)")
    print("="*70)

    # Create dataset and loader
    dataset = BiomassTestDataset(CFG.TEST_IMAGE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )

    transform = get_inference_transform()

    kernel_test_pred = None
    if (
        CFG.USE_KERNEL_REG
        and CFG.PREDICT_TARGETS
        and KERNEL_TRAIN_EMB is not None
        and KERNEL_TRAIN_LABELS is not None
    ):
        test_df = pd.DataFrame({"image_path": dataset.filenames})
        test_emb = compute_kernel_embeddings(test_df, CFG.TEST_IMAGE_DIR, CFG.DEVICE)
        kernel_test_pred = kernel_regression_predict(
            train_emb=KERNEL_TRAIN_EMB,
            train_labels=KERNEL_TRAIN_LABELS,
            query_emb=test_emb,
            device=CFG.DEVICE,
            topk=CFG.KERNEL_REG_TOPK,
            tau=CFG.KERNEL_REG_TAU,
            batch_size=CFG.KERNEL_REG_PRED_BATCH_SIZE,
            exclude_self=False,
        )

    use_kernel = kernel_test_pred is not None
    use_pack = bool(CFG.PREDICT_TARGETS)
    pred_cols = len(CFG.ALL_TARGET_COLS) if use_pack else 3

    # Initialize accumulator for predictions
    accumulated_preds = np.zeros((len(dataset), pred_cols), dtype=np.float32)

    # Use configured folds, fallback to full range if not set or empty
    folds_to_use = getattr(CFG, 'FOLDS_TO_TRAIN', list(range(CFG.N_FOLDS)))
    if not folds_to_use:
        folds_to_use = list(range(CFG.N_FOLDS))

    print(f"Folds requested for inference: {folds_to_use}")

    # Use filenames from dataset (guaranteed consistent ordering with loader because shuffle=False)
    filenames = dataset.filenames.copy()

    successful_folds = 0

    # Loop over requested folds only
    for fold in folds_to_use:
        print(f"\nProcessing Fold {fold}...")

        model_dir = CFG.MODEL_DIR
        backbone_path = getattr(CFG, 'BACKBONE_PATH', None)
        models = []
        model_paths = []
        if CFG.USE_UCVME:
            if getattr(CFG, "UCVME_SHARED_BACKBONE", False):
                model_paths = [
                    ("shared", os.path.join(model_dir, f"best_model_fold{fold}_shared.pth")),
                ]
            else:
                path_a = os.path.join(model_dir, f"best_model_fold{fold}_a.pth")
                path_b = os.path.join(model_dir, f"best_model_fold{fold}_b.pth")
                if os.path.exists(path_a) or os.path.exists(path_b):
                    model_paths = [("a", path_a), ("b", path_b)]
                else:
                    model_paths = [("single", os.path.join(model_dir, f"best_model_fold{fold}.pth"))]
        else:
            model_paths = [("single", os.path.join(model_dir, f"best_model_fold{fold}.pth"))]

        for label, weight_path in model_paths:
            if not os.path.exists(weight_path):
                print(f"Warning: Model file {weight_path} not found! Skipping {label} for fold {fold}.")
                continue
            model = BiomassModel(
                CFG.MODEL_NAME,
                pretrained=False,
                backbone_path=backbone_path,
                meta_info=META_INFO,
            )
            state = torch.load(weight_path, map_location='cpu')

            if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
                key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
                sd = state[key]
            else:
                sd = state

            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"  Warning: missing keys={len(missing)} unexpected keys={len(unexpected)} ({label})")
            model.to(CFG.DEVICE)
            model.eval()
            models.append(model)

        if not models:
            print(f"Warning: No model weights found for fold {fold}. Skipping fold.")
            torch.cuda.empty_cache(); gc.collect()
            continue

        # Run inference for this fold
        for i, (left, right, filename) in enumerate(tqdm(loader, desc=f"Fold {fold}")):
            # left and right are batches of size 1, convert to numpy for transform function
            left_np = left[0].numpy()
            right_np = right[0].numpy()

            preds = []
            for model in models:
                pred = predict_single(
                    model,
                    left_np,
                    right_np,
                    transform,
                    kernel_pred=(kernel_test_pred[i] if use_kernel else None),
                    return_pack=use_pack,
                )
                preds.append(pred)
            pred = np.mean(np.stack(preds, axis=0), axis=0)
            accumulated_preds[i] += pred

        successful_folds += 1

        # Cleanup model to save memory
        for model in models:
            del model
        torch.cuda.empty_cache(); gc.collect()

    if successful_folds == 0:
        raise FileNotFoundError(f"No model weights found for requested folds: {folds_to_use}")

    # Average predictions over the number of successfully loaded folds
    final_predictions = accumulated_preds / successful_folds

    print(f"\nInference complete. Successfully used {successful_folds} fold(s) out of {len(folds_to_use)} requested.")
    return final_predictions, filenames

# ===============================================================
# 9. POST-PROCESS PREDICTIONS
# ===============================================================
def postprocess_predictions(preds_direct):
    """
    Calculate derived targets from direct predictions.

    Input: (n_samples, 3) array with [total, gdm, green]
    Output: (n_samples, 5) array with [green, dead, clover, gdm, total]
    """
    print("\nPost-processing predictions...")
    if preds_direct.shape[1] == len(CFG.ALL_TARGET_COLS):
        return preds_direct

    # Extract direct predictions
    pred_total = preds_direct[:, 0]
    pred_gdm = preds_direct[:, 1]
    pred_green = preds_direct[:, 2]

    # Calculate derived targets with non-negativity constraint
    pred_clover = np.maximum(0, pred_gdm - pred_green)
    pred_dead = np.maximum(0, pred_total - pred_gdm)

    # Stack in the order of ALL_TARGET_COLS
    # ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    preds_all = np.stack([
        pred_green,
        pred_dead,
        pred_clover,
        pred_gdm,
        pred_total
    ], axis=1)

    print(f"✓ Post-processing complete")
    print(f"  Output shape: {preds_all.shape}")
    print(f"\nPrediction statistics:")
    for i, col in enumerate(CFG.ALL_TARGET_COLS):
        print(f"  {col:15s}: mean={preds_all[:, i].mean():.2f}, "
              f"std={preds_all[:, i].std():.2f}, "
              f"min={preds_all[:, i].min():.2f}, "
              f"max={preds_all[:, i].max():.2f}")

    return preds_all

# ===============================================================
# 10. CREATE SUBMISSION FILE (FIXED)
# ===============================================================
def create_submission(predictions, filenames):
    """
    Create submission file in the required format.

    Args:
        predictions: (n_images, 5) array with all target predictions
        filenames: list of test image filenames
    """
    print("\n" + "="*70)
    print("CREATING SUBMISSION FILE")
    print("="*70)

    # Step 0: Load test.csv first to check the image_path format
    test_df = pd.read_csv(CFG.TEST_CSV)
    print(f"\nTest CSV loaded: {len(test_df)} rows")
    print(f"Sample image_path from test.csv: {test_df['image_path'].iloc[0]}")
    print(f"Sample filename from predictions: {filenames[0]}")

    # Step 1: Fix image_path format to match test.csv
    # If test.csv has "test/ID123.jpg" but we have "ID123.jpg", add the prefix
    test_path_example = test_df['image_path'].iloc[0]
    if '/' in test_path_example:
        prefix = test_path_example.rsplit('/', 1)[0] + '/'
        corrected_filenames = [prefix + fn for fn in filenames]
        print(f"Corrected path format: {corrected_filenames[0]}")
    else:
        corrected_filenames = filenames

    # Step 2: Create wide-format DataFrame with corrected paths
    preds_wide = pd.DataFrame(predictions, columns=CFG.ALL_TARGET_COLS)
    preds_wide.insert(0, 'image_path', corrected_filenames)

    print(f"\nWide format predictions:")
    print(preds_wide.head())

    # Step 3: Convert to long format (melt)
    preds_long = preds_wide.melt(
        id_vars=['image_path'],
        value_vars=CFG.ALL_TARGET_COLS,
        var_name='target_name',
        value_name='target'
    )

    print(f"\nLong format predictions (first 10 rows):")
    print(preds_long.head(10))

    # Step 4: Debug the merge
    print(f"\nDebug: Checking if paths match...")
    print(f"Unique paths in test_df: {test_df['image_path'].nunique()}")
    print(f"Unique paths in preds_long: {preds_long['image_path'].nunique()}")

    common_paths = set(test_df['image_path'].unique()) & set(preds_long['image_path'].unique())
    print(f"Common paths found: {len(common_paths)}")

    if len(common_paths) == 0:
        print("\n❌ ERROR: No matching paths found!")
        print(f"Test CSV paths sample: {list(test_df['image_path'].unique()[:3])}")
        print(f"Prediction paths sample: {list(preds_long['image_path'].unique()[:3])}")
        raise ValueError("Path mismatch between test.csv and predictions")

    # Step 5: Merge to get sample_ids
    submission = pd.merge(
        test_df[['sample_id', 'image_path', 'target_name']],
        preds_long,
        on=['image_path', 'target_name'],
        how='left'
    )

    # Step 6: Keep only required columns
    submission = submission[['sample_id', 'target']]

    # Step 7: Check for missing values
    missing_count = submission['target'].isna().sum()
    if missing_count > 0:
        print(f"\n⚠ Warning: {missing_count} missing predictions found!")
        print("Sample missing entries:")
        print(submission[submission['target'].isna()].head())
        submission.loc[submission['target'].isna(), 'target'] = 0.0

    # Step 8: Sort by sample_id
    submission = submission.sort_values('sample_id').reset_index(drop=True)

    # Step 9: Save to CSV
    output_path = os.path.join(CFG.SUBMISSION_DIR, 'submission.csv')
    submission.to_csv(output_path, index=False)

    print(f"\n✓ Submission file saved: {output_path}")
    print(f"  Total rows: {len(submission)}")
    print(f"\nPrediction statistics:")
    print(f"  Min: {submission['target'].min():.4f}")
    print(f"  Max: {submission['target'].max():.4f}")
    print(f"  Mean: {submission['target'].mean():.4f}")
    print(f"  Non-zero values: {(submission['target'] > 0).sum()}/{len(submission)}")

    print(f"\nFirst 10 rows:")
    print(submission.head(10))
    print(f"\nLast 10 rows:")
    print(submission.tail(10))

    # Step 10: Validation checks
    print(f"\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    print(f"✓ Expected rows: {len(test_df)}")
    print(f"✓ Actual rows: {len(submission)}")
    print(f"✓ Match: {len(submission) == len(test_df)}")
    print(f"✓ No missing values: {not submission['target'].isna().any()}")
    print(f"✓ All sample_ids unique: {submission['sample_id'].is_unique}")
    print(f"✓ Has non-zero predictions: {(submission['target'] > 0).any()}")

    return submission

# ===============================================================
# CREATE SUBMISSION
# ===============================================================
if CFG.CREATE_SUBMISSION:
    predictions_direct, test_filenames = run_inference()
    predictions_all = postprocess_predictions(predictions_direct)
    submission_df = create_submission(predictions_all, test_filenames)
