#!/usr/bin/env python3
# Inference v2: shared backbone tokens + headset-only A/B per fold

import os
import gc
import math
from contextlib import nullcontext

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from timm.utils import ModelEmaV2


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
_LOCAL_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
_load_env_file(os.path.join(_SCRIPT_DIR, ".env"))


class CFG:
    CREATE_SUBMISSION = True
    TTT_MODE = False
    UCVME_MODE = True
    USE_TQDM = False
    DIAG_NONFINITE = True

    BASE_PATH = os.environ.get(
        "CSIRO_BIOMASS_DIR",
        _LOCAL_DATA_DIR if os.path.isdir(_LOCAL_DATA_DIR) else "/data",
    )

    TRAIN_CSV = os.environ.get("CSIRO_TRAIN_CSV", os.path.join(BASE_PATH, "train.csv"))
    TRAIN_IMAGE_DIR = os.environ.get("CSIRO_TRAIN_IMAGE_DIR", os.path.join(BASE_PATH, "train"))
    TEST_IMAGE_DIR = os.environ.get("CSIRO_TEST_IMAGE_DIR", os.path.join(BASE_PATH, "test"))
    TEST_CSV = os.environ.get("CSIRO_TEST_CSV", os.path.join(BASE_PATH, "test.csv"))
    SUBMISSION_DIR = os.environ.get("CSIRO_SUBMISSION_DIR", os.getcwd())

    MODEL_DIR = os.environ.get("CSIRO_MODEL_DIR", os.path.join(os.getcwd(), "models"))
    MODEL_NAME = os.environ.get("CSIRO_MODEL_NAME", "vit_huge_plus_patch16_dinov3.lvd1689m")
    BACKBONE_PATH = os.environ.get("CSIRO_BACKBONE_PATH", None)
    BACKBONE_FROM_CKPT = True
    PRETRAINED_BACKBONE = True

    N_FOLDS = 5
    FOLDS_TO_USE = [0, 1, 2, 3, 4]
    REQUIRE_AB = True

    IMG_SIZE = 512
    NUM_WORKERS = 4
    TTA_STEPS = 4

    USE_UCVME = True
    UCVME_BACKBONE_DROP_RATE = 0.05
    UCVME_BACKBONE_DROP_PATH_RATE = 0.10

    USE_HETERO = True
    HETERO_MIN_LOGVAR = -10.0
    HETERO_MAX_LOGVAR = 4.0
    HETERO_EPS = 1e-6
    EMA_DECAY = 0.9

    PREDICT_DATE = True
    PREDICT_STATE = False
    PREDICT_SPECIES = False
    PREDICT_NDVI = True
    PREDICT_HEIGHT = True
    DATE_MODE = "dayofyear_sincos"

    LOSS_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    ALL_TARGET_COLS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

    TTT_EPOCHS = 1
    TTT_LR = 2e-4
    TTT_WD = 0.0
    TTT_GRAD_ACC = 4
    TTT_CLIP_NORM = 1.0
    TTT_VAR_FLOOR = 1e-3
    TTT_VAR_CEIL = 1e6
    TTT_RESTORE_PROB = 0.01
    TTT_BATCH_SIZE = 2
    TTT_PASSES = 1
    UCVME_MC_SAMPLES = 5
    UCVME_ULB_WEIGHT = 2.0
    UCVME_EPOCHS = 1

    USE_BF16 = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


AMP_DTYPE = torch.bfloat16 if CFG.USE_BF16 else torch.float16
scaler = torch.amp.GradScaler(
    "cuda", enabled=(torch.cuda.is_available() and AMP_DTYPE == torch.float16)
)


def autocast_ctx():
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE)


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


def clean_image(img):
    h, w = img.shape[:2]
    img = img[0:int(h * 0.90), :]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img


def get_inference_transform():
    return A.Compose(
        [
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(num_transforms):
    num = max(1, int(num_transforms))
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    to_tensor = ToTensorV2()
    all_tta_transforms = [
        A.Compose([A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE), normalize, to_tensor]),
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            normalize,
            to_tensor,
        ]),
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            normalize,
            to_tensor,
        ]),
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            normalize,
            to_tensor,
        ]),
    ]
    return all_tta_transforms[:num]


class BiomassUnlabeledTensorDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.paths = sorted(
            [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.filenames = [os.path.basename(p) for p in self.paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)

        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid].copy()
        right = img[:, mid:].copy()

        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        return idx, left_t, right_t, self.filenames[idx]


class BiomassRawTestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.paths = sorted(
            [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.filenames = [os.path.basename(p) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid].copy()
        right = img[:, mid:].copy()
        return idx, left, right, self.filenames[idx]


class BiomassLabeledTensorDataset(Dataset):
    def __init__(self, df, img_dir, transform, meta_info=None, return_meta=False):
        self.df = df
        self.img_dir = img_dir
        self.paths = df["image_path"].values
        self.labels = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)
        self.transform = transform
        self.meta_info = meta_info or {}
        self.return_meta = bool(return_meta) and bool(self.meta_info.get("enabled"))
        if self.return_meta:
            self.meta_arrays = build_meta_arrays(df, self.meta_info)
        else:
            self.meta_arrays = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)

        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid].copy()
        right = img[:, mid:].copy()

        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        label_t = torch.from_numpy(self.labels[idx])
        if not self.return_meta:
            return idx, left_t, right_t, label_t
        meta = {}
        if "date" in self.meta_arrays:
            meta["date"] = torch.from_numpy(self.meta_arrays["date"][idx])
            meta["date_mask"] = torch.from_numpy(self.meta_arrays["date_mask"][idx])
        if "state" in self.meta_arrays:
            meta["state"] = torch.tensor(self.meta_arrays["state"][idx], dtype=torch.long)
            meta["state_mask"] = torch.tensor(self.meta_arrays["state_mask"][idx], dtype=torch.float32)
        if "species" in self.meta_arrays:
            meta["species"] = torch.from_numpy(self.meta_arrays["species"][idx])
            meta["species_mask"] = torch.from_numpy(self.meta_arrays["species_mask"][idx])
        if "ndvi" in self.meta_arrays:
            meta["ndvi"] = torch.from_numpy(self.meta_arrays["ndvi"][idx])
            meta["ndvi_mask"] = torch.from_numpy(self.meta_arrays["ndvi_mask"][idx])
        if "height" in self.meta_arrays:
            meta["height"] = torch.from_numpy(self.meta_arrays["height"][idx])
            meta["height_mask"] = torch.from_numpy(self.meta_arrays["height_mask"][idx])
        return idx, left_t, right_t, label_t, meta


def _load_train_wide():
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df_wide = (
        df_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
    )
    if df_wide["image_path"].duplicated().any():
        raise ValueError("Duplicate image_path values found in train.csv.")

    meta_cols = ["Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    available_meta_cols = [c for c in meta_cols if c in df_long.columns]
    if available_meta_cols:
        meta_df = df_long[["image_path"] + available_meta_cols].drop_duplicates()
        df_wide = df_wide.merge(meta_df, on="image_path", how="left")

    if "Dry_Clover_g" not in df_wide.columns:
        if "GDM_g" in df_wide.columns and "Dry_Green_g" in df_wide.columns:
            df_wide["Dry_Clover_g"] = np.maximum(
                0.0, df_wide["GDM_g"].to_numpy() - df_wide["Dry_Green_g"].to_numpy()
            )
    if "Dry_Dead_g" not in df_wide.columns:
        if "Dry_Total_g" in df_wide.columns and "GDM_g" in df_wide.columns:
            df_wide["Dry_Dead_g"] = np.maximum(
                0.0, df_wide["Dry_Total_g"].to_numpy() - df_wide["GDM_g"].to_numpy()
            )

    missing = [c for c in CFG.ALL_TARGET_COLS if c not in df_wide.columns]
    if missing:
        raise ValueError(f"train.csv missing target columns: {missing}")
    keep_meta_cols = [c for c in meta_cols if c in df_wide.columns]
    return df_wide[["image_path"] + keep_meta_cols + CFG.ALL_TARGET_COLS]


def _split_species_tokens(value):
    if pd.isna(value):
        return []
    text = str(value)
    parts = [p for p in text.split("_") if p]
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
        "species_multi_label": True,
    }

    if CFG.PREDICT_DATE and "Sampling_Date" in df.columns:
        meta["use_date"] = True
    if CFG.PREDICT_STATE and "State" in df.columns:
        states = sorted(df["State"].dropna().astype(str).unique().tolist())
        if states:
            meta["use_state"] = True
            meta["state_to_idx"] = {s: i for i, s in enumerate(states)}
            meta["num_states"] = len(states)
    if CFG.PREDICT_SPECIES and "Species" in df.columns:
        tokens = set()
        for val in df["Species"].dropna().astype(str).tolist():
            tokens.update(_split_species_tokens(val))
        species = sorted(tokens)
        if species:
            meta["use_species"] = True
            meta["species_to_idx"] = {s: i for i, s in enumerate(species)}
            meta["num_species"] = len(species)
    if CFG.PREDICT_NDVI and "Pre_GSHH_NDVI" in df.columns:
        meta["use_ndvi"] = True
    if CFG.PREDICT_HEIGHT and "Height_Ave_cm" in df.columns:
        meta["use_height"] = True

    meta["enabled"] = any(
        [
            meta["use_date"],
            meta["use_state"],
            meta["use_species"],
            meta["use_ndvi"],
            meta["use_height"],
        ]
    )
    return meta


def _parse_dates(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce")
    if CFG.DATE_MODE == "dayofyear_sincos":
        doy = dt.dt.dayofyear.astype(float)
        angle = 2.0 * math.pi * (doy - 1.0) / 365.25
        sin = np.sin(angle)
        cos = np.cos(angle)
        values = np.stack([sin, cos], axis=1)
    elif CFG.DATE_MODE == "dayofyear":
        values = dt.dt.dayofyear
    elif CFG.DATE_MODE == "month":
        values = dt.dt.month
    else:
        values = dt.map(lambda d: d.toordinal() if pd.notna(d) else np.nan)
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    return values.astype(np.float32)


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


class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class BiomassHeadset(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head_green_raw = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_clover_raw = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_dead_raw = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_gdm_raw = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_total_raw = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(nf // 2, 1), nn.Softplus()
        )

        self.use_hetero = bool(CFG.USE_HETERO)
        if self.use_hetero:
            self.head_green_logvar = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )
            self.head_clover_logvar = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )
            self.head_dead_logvar = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )
            self.head_gdm_logvar = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )
            self.head_total_logvar = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )

        self.use_date = bool(CFG.PREDICT_DATE)
        self.use_state = bool(CFG.PREDICT_STATE)
        self.use_species = bool(CFG.PREDICT_SPECIES)
        self.use_ndvi = bool(CFG.PREDICT_NDVI)
        self.use_height = bool(CFG.PREDICT_HEIGHT)

        if self.use_date:
            date_out = 2 if CFG.DATE_MODE == "dayofyear_sincos" else 1
            self.head_date = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, date_out)
            )
        if self.use_state:
            self.head_state = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )
        if self.use_species:
            self.head_species = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )
        if self.use_ndvi:
            self.head_ndvi = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )
        if self.use_height:
            self.head_height = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(nf // 2, 1)
            )

    def forward_tokens(self, x_l, x_r):
        x_cat = torch.cat([x_l, x_r], dim=1)
        x_fused = self.fusion(x_cat)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)

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

        out = {"targets": (total, gdm, green, clover, dead)}
        if meta_out:
            out["meta"] = meta_out
        if logvar_comp is not None:
            out["target_logvar"] = logvar_comp
        return out


def _as_col(x: torch.Tensor, bs: int) -> torch.Tensor:
    if x.ndim == 1:
        return x.view(bs, 1)
    if x.ndim == 2:
        return x
    return x.view(bs, 1)


def _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs: int) -> torch.Tensor:
    pg = _as_col(p_green, bs)
    pd = _as_col(p_dead, bs)
    pc = _as_col(p_clover, bs)
    pgdm = _as_col(p_gdm, bs)
    pt = _as_col(p_total, bs)
    return torch.cat([pg, pd, pc, pgdm, pt], dim=1)


def _clamp_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        logvar,
        min=float(CFG.HETERO_MIN_LOGVAR),
        max=float(CFG.HETERO_MAX_LOGVAR),
    )


def _sanitize_logvar(logvar: torch.Tensor | None) -> torch.Tensor | None:
    if logvar is None:
        return None
    logvar = logvar.float()
    logvar = torch.nan_to_num(
        logvar,
        nan=0.0,
        posinf=float(CFG.HETERO_MAX_LOGVAR),
        neginf=float(CFG.HETERO_MIN_LOGVAR),
    )
    return _clamp_logvar(logvar)


def _build_logvar_pack(logvar_comp):
    if logvar_comp is None:
        return None
    logvar_comp = _clamp_logvar(logvar_comp.float())
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
    logvar_pack = torch.log(var_pack + float(CFG.HETERO_EPS))
    return _sanitize_logvar(logvar_pack)


def _split_outputs(outputs):
    if isinstance(outputs, dict):
        return (
            outputs.get("targets"),
            outputs.get("meta"),
            outputs.get("target_logvar"),
        )
    return outputs, None, None


def forward_pred_and_alevar_tokens(headset, x_l, x_r):
    outputs = headset.forward_tokens(x_l, x_r)
    target_out, _, logvar_comp = _split_outputs(outputs)
    p_total, p_gdm, p_green, p_clover, p_dead = target_out
    bs = x_l.size(0)
    pred_pack = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs)
    _check_nonfinite("pred_pack", pred_pack, "forward_alevar")
    _check_nonfinite("logvar_comp", logvar_comp, "forward_alevar")
    if not CFG.USE_HETERO:
        return pred_pack, None
    logvar_pack = _build_logvar_pack(logvar_comp)
    _check_nonfinite("logvar_pack", logvar_pack, "forward_alevar")
    ale_var = torch.exp(torch.clamp(logvar_pack, min=-20.0, max=20.0))
    _check_nonfinite("ale_var", ale_var, "forward_alevar")
    return pred_pack, ale_var


def forward_pred_and_logvar_tokens(headset, x_l, x_r):
    outputs = headset.forward_tokens(x_l, x_r)
    target_out, _, logvar_comp = _split_outputs(outputs)
    p_total, p_gdm, p_green, p_clover, p_dead = target_out
    bs = x_l.size(0)
    pred_pack = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs)
    _check_nonfinite("pred_pack", pred_pack, "forward_logvar")
    _check_nonfinite("logvar_comp", logvar_comp, "forward_logvar")
    if not CFG.USE_HETERO:
        return pred_pack, None
    logvar_pack = _build_logvar_pack(logvar_comp)
    _check_nonfinite("logvar_pack", logvar_pack, "forward_logvar")
    return pred_pack, _sanitize_logvar(logvar_pack)


def forward_pred_logvar_meta(headset, x_l, x_r):
    outputs = headset.forward_tokens(x_l, x_r)
    target_out, meta_out, logvar_comp = _split_outputs(outputs)
    p_total, p_gdm, p_green, p_clover, p_dead = target_out
    bs = x_l.size(0)
    pred_pack = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs)
    if not CFG.USE_HETERO:
        return pred_pack, None, meta_out
    logvar_pack = _build_logvar_pack(logvar_comp)
    return pred_pack, _sanitize_logvar(logvar_pack), meta_out


def _load_checkpoint(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and ("model_state_dict" in sd or "state_dict" in sd):
        return sd.get("model_state_dict", sd.get("state_dict"))
    return sd


def _load_backbone_from_ckpt(backbone, path):
    sd = _load_checkpoint(path)
    backbone_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not backbone_sd:
        return False
    backbone.load_state_dict(backbone_sd, strict=False)
    return True


def build_backbone():
    backbone_kwargs = {}
    if CFG.USE_UCVME:
        backbone_kwargs["drop_rate"] = float(CFG.UCVME_BACKBONE_DROP_RATE)
        backbone_kwargs["drop_path_rate"] = float(CFG.UCVME_BACKBONE_DROP_PATH_RATE)
    backbone = timm.create_model(
        CFG.MODEL_NAME, pretrained=False, num_classes=0, global_pool="", **backbone_kwargs
    )
    return backbone


def load_backbone_weights(backbone, ckpt_paths):
    if CFG.BACKBONE_FROM_CKPT:
        for path in ckpt_paths:
            if _load_backbone_from_ckpt(backbone, path):
                print(f"Loaded backbone weights from {path}")
                return
    if CFG.BACKBONE_PATH and os.path.exists(CFG.BACKBONE_PATH):
        sd = _load_checkpoint(CFG.BACKBONE_PATH)
        if isinstance(sd, dict) and ("model" in sd or "state_dict" in sd):
            sd = sd.get("model", sd.get("state_dict"))
        backbone.load_state_dict(sd, strict=False)
        print(f"Loaded backbone weights from {CFG.BACKBONE_PATH}")
        return
    if CFG.PRETRAINED_BACKBONE:
        print("Loading backbone weights from timm pretrained.")
        sd = timm.create_model(CFG.MODEL_NAME, pretrained=True, num_classes=0, global_pool="").state_dict()
        backbone.load_state_dict(sd, strict=False)
        return
    print("Warning: backbone weights not loaded from checkpoint or pretrained.")


def _get_fold_pair_paths(fold):
    path_a = os.path.join(CFG.MODEL_DIR, f"best_model_fold{fold}_a.pth")
    path_b = os.path.join(CFG.MODEL_DIR, f"best_model_fold{fold}_b.pth")
    base = os.path.join(CFG.MODEL_DIR, f"best_model_fold{fold}.pth")
    paths = []
    if os.path.exists(path_a) and os.path.exists(path_b):
        return [path_a, path_b]
    if os.path.exists(path_a) or os.path.exists(path_b):
        paths = [p for p in (path_a, path_b) if os.path.exists(p)]
        if CFG.REQUIRE_AB:
            print(f"Warning: missing A/B pair for fold {fold} in {CFG.MODEL_DIR}")
            return []
        return paths
    if os.path.exists(base):
        return [base]
    return []


def load_headset_from_ckpt(path, nf, device):
    headset = BiomassHeadset(nf)
    sd = _load_checkpoint(path)
    head_sd = {k: v for k, v in sd.items() if not k.startswith("backbone.")}
    missing, unexpected = headset.load_state_dict(head_sd, strict=False)
    if missing or unexpected:
        print(f"  {os.path.basename(path)} missing={len(missing)} unexpected={len(unexpected)}")
    headset.to(device)
    headset.eval()
    return headset


def load_fold_headsets(folds):
    fold_headsets = {}
    ckpt_paths = []
    for fold in folds:
        paths = _get_fold_pair_paths(fold)
        if not paths:
            continue
        ckpt_paths.extend(paths)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {CFG.MODEL_DIR}")

    backbone = build_backbone()
    load_backbone_weights(backbone, ckpt_paths)
    backbone.to(CFG.DEVICE)
    backbone.eval()

    nf = backbone.num_features
    for fold in folds:
        paths = _get_fold_pair_paths(fold)
        if not paths:
            continue
        headsets = [load_headset_from_ckpt(p, nf, CFG.DEVICE) for p in paths]
        fold_headsets[fold] = headsets
    if not fold_headsets:
        raise FileNotFoundError(f"No valid fold headsets found in {CFG.MODEL_DIR}")
    return backbone, fold_headsets


def forward_fold_headsets(headsets, x_l, x_r):
    preds = []
    vars_ = []
    for headset in headsets:
        pred_pack, ale_var = forward_pred_and_alevar_tokens(headset, x_l, x_r)
        preds.append(pred_pack)
        if ale_var is not None:
            vars_.append(ale_var)
    pred_pack = torch.mean(torch.stack(preds, dim=0), dim=0)
    ale_var = torch.mean(torch.stack(vars_, dim=0), dim=0) if vars_ else None
    _check_nonfinite("pred_pack", pred_pack, "forward_fold_headsets")
    _check_nonfinite("ale_var", ale_var, "forward_fold_headsets")
    return pred_pack, ale_var


def _predict_tta_sum(backbone, fold_headsets, folds_to_use, left_np, right_np, tta_transforms):
    pred_sum = None
    for tfm in tta_transforms:
        left_tensor = tfm(image=left_np)["image"].unsqueeze(0).to(CFG.DEVICE)
        right_tensor = tfm(image=right_np)["image"].unsqueeze(0).to(CFG.DEVICE)
        with autocast_ctx():
            x_l = backbone(left_tensor)
            x_r = backbone(right_tensor)
        for fold in folds_to_use:
            headsets = fold_headsets.get(fold)
            if not headsets:
                continue
            with autocast_ctx():
                pred_pack, _ = forward_fold_headsets(headsets, x_l, x_r)
            pred_sum = pred_pack if pred_sum is None else (pred_sum + pred_pack)
    if pred_sum is None:
        raise RuntimeError("No valid headsets found for TTA prediction.")
    return pred_sum


@torch.no_grad()
def run_inference_shared_backbone():
    dataset = BiomassRawTestDataset(CFG.TEST_IMAGE_DIR)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True
    )
    tta_transforms = get_tta_transforms(CFG.TTA_STEPS)
    print(f"Using TTA with {len(tta_transforms)} views.")

    folds_to_use = CFG.FOLDS_TO_USE or list(range(CFG.N_FOLDS))
    backbone, fold_headsets = load_fold_headsets(folds_to_use)

    pred_cols = len(CFG.ALL_TARGET_COLS)
    accumulated_preds = np.zeros((len(dataset), pred_cols), dtype=np.float32)

    successful_folds = 0
    for fold in folds_to_use:
        if fold not in fold_headsets:
            print(f"Skipping fold {fold} (no headsets found).")
            continue
        successful_folds += 1

    if successful_folds == 0:
        raise FileNotFoundError("No fold headsets available for inference.")

    denom = float(successful_folds * len(tta_transforms))
    for idx, left, right, _ in tqdm(loader, desc="inference"):
        idx_np = idx.numpy()
        left_np = left[0].numpy()
        right_np = right[0].numpy()
        pred_sum = _predict_tta_sum(
            backbone, fold_headsets, folds_to_use, left_np, right_np, tta_transforms
        )
        accumulated_preds[idx_np] += pred_sum.float().cpu().numpy()

    final_predictions = accumulated_preds / denom
    return final_predictions, dataset.filenames


def loo_pseudo_targets(mus, ales=None, leave_out=0):
    M, N, _ = mus.shape
    if M < 2:
        raise ValueError("Need at least 2 folds for LOO pseudo targets.")
    mu_sum = mus.sum(axis=0)
    mu_sumsq = (mus * mus).sum(axis=0)
    mu_i = mus[leave_out]
    mu_t = (mu_sum - mu_i) / float(M - 1)
    ex2 = (mu_sumsq - (mu_i * mu_i)) / float(M - 1)
    var_epi = np.maximum(ex2 - (mu_t * mu_t), 0.0)

    if ales is not None and np.isfinite(ales).any():
        ale_sum = np.nansum(ales, axis=0)
        ale_i = ales[leave_out]
        ale_t = (ale_sum - np.nan_to_num(ale_i, nan=0.0)) / float(M - 1)
        ale_t = np.maximum(ale_t, 0.0)
    else:
        ale_t = np.zeros_like(var_epi, dtype=np.float32)

    var_t = var_epi + ale_t
    return mu_t.astype(np.float32), var_t.astype(np.float32)


def fixed_var_nll(pred, target, var, w=None):
    var = torch.clamp(var, min=float(CFG.TTT_VAR_FLOOR), max=float(CFG.TTT_VAR_CEIL))
    loss = 0.5 * ((pred - target) ** 2 / var + torch.log(var))
    if w is not None:
        w_t = torch.as_tensor(w, device=loss.device, dtype=loss.dtype).view(1, -1)
        loss = loss * w_t
    return loss.mean()


def hetero_nll_loss(pred_pack, logvar_pack, labels, w=None):
    if logvar_pack is None:
        return None
    diff = pred_pack - labels
    logvar = logvar_pack.to(device=pred_pack.device, dtype=pred_pack.dtype)
    loss = 0.5 * (torch.exp(-logvar) * diff * diff + logvar)
    if w is not None:
        w_t = torch.as_tensor(w, device=loss.device, dtype=loss.dtype).view(1, -1)
        loss = loss * w_t
    return loss.mean()


def _ucvme_reg_fixed_logvar(pred_pack, target, logvar_target):
    logvar_target = _clamp_logvar(logvar_target)
    diff = pred_pack - target
    loss = 0.5 * (torch.exp(-logvar_target) * diff * diff + logvar_target)
    return loss.mean()


def _ucvme_uncertainty_consistency(logvar_a, logvar_b):
    if logvar_a is None or logvar_b is None:
        return None
    diff = logvar_a - logvar_b
    return (diff * diff).mean()


def _ucvme_unc_to_target(logvar_pred, logvar_target):
    if logvar_pred is None or logvar_target is None:
        return None
    logvar_pred = _sanitize_logvar(logvar_pred)
    logvar_target = _sanitize_logvar(logvar_target)
    if logvar_pred is None or logvar_target is None:
        return None
    diff = logvar_pred - logvar_target
    return (diff * diff).mean()


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
        total = l_date if total is None else total + l_date

    if "state" in meta_out and "state" in meta_labels:
        l_state = _masked_ce(meta_out["state"], meta_labels["state"], meta_labels.get("state_mask"))
        total = l_state if total is None else total + l_state

    if "species" in meta_out and "species" in meta_labels:
        l_species = _masked_bce(meta_out["species"], meta_labels["species"], meta_labels.get("species_mask"))
        total = l_species if total is None else total + l_species

    if "ndvi" in meta_out and "ndvi" in meta_labels:
        l_ndvi = _masked_smooth_l1(meta_out["ndvi"], meta_labels["ndvi"], meta_labels.get("ndvi_mask"))
        total = l_ndvi if total is None else total + l_ndvi

    if "height" in meta_out and "height" in meta_labels:
        l_height = _masked_smooth_l1(meta_out["height"], meta_labels["height"], meta_labels.get("height_mask"))
        total = l_height if total is None else total + l_height

    return total


@torch.no_grad()
def _ucvme_pseudo_labels_headsets(headset_a, headset_b, x_l, x_r, mc_samples):
    preds = []
    logvars = []
    for _ in range(int(mc_samples)):
        pred_a, logvar_a = forward_pred_and_logvar_tokens(headset_a, x_l, x_r)
        pred_b, logvar_b = forward_pred_and_logvar_tokens(headset_b, x_l, x_r)
        if logvar_a is None or logvar_b is None:
            raise RuntimeError("UCVME requires USE_HETERO=True.")
        preds.extend([pred_a, pred_b])
        logvars.extend([_sanitize_logvar(logvar_a), _sanitize_logvar(logvar_b)])
    y_e = torch.stack(preds, dim=0).mean(dim=0)
    z_e = torch.stack(logvars, dim=0).mean(dim=0)
    z_e = _clamp_logvar(z_e)
    _check_nonfinite("pseudo_y", y_e, "ucvme_pseudo")
    _check_nonfinite("pseudo_z", z_e, "ucvme_pseudo")
    return y_e.detach(), z_e.detach()


def maybe_restore_heads(model, source_state, p_restore=0.01):
    if source_state is None or p_restore <= 0.0:
        return
    if np.random.rand() >= p_restore:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("fusion.") or name.startswith("head_"):
                if name in source_state:
                    p.copy_(source_state[name].to(p.device, dtype=p.dtype))


def set_trainable_for_ttt(headset):
    for name, p in headset.named_parameters():
        if name.startswith("fusion."):
            p.requires_grad = True
        elif name.startswith("head_"):
            if "logvar" in name:
                p.requires_grad = False
            elif any(k in name for k in ("date", "state", "species", "ndvi", "height")):
                p.requires_grad = False
            else:
                p.requires_grad = True
        else:
            p.requires_grad = False


def set_trainable_for_ucvme(headset):
    for name, p in headset.named_parameters():
        if name.startswith("fusion.") or name.startswith("head_"):
            p.requires_grad = True
        else:
            p.requires_grad = False


@torch.no_grad()
def collect_fold_preds(backbone, fold_headsets, loader, folds_to_use):
    M = len(folds_to_use)
    N = len(loader.dataset)
    D = len(CFG.ALL_TARGET_COLS)
    mus = np.zeros((M, N, D), dtype=np.float32)
    ales = np.full((M, N, D), np.nan, dtype=np.float32)
    filenames = [None] * N

    for idx, left, right, fn in tqdm(loader, desc="collect", leave=False):
        idx_np = idx.numpy()
        left = left.to(CFG.DEVICE, non_blocking=True)
        right = right.to(CFG.DEVICE, non_blocking=True)
        with autocast_ctx():
            x_l = backbone(left)
            x_r = backbone(right)
        for m, fold in enumerate(folds_to_use):
            headsets = fold_headsets.get(fold)
            if not headsets:
                continue
            with autocast_ctx():
                pred_pack, ale_var = forward_fold_headsets(headsets, x_l, x_r)
            mus[m, idx_np] = pred_pack.float().cpu().numpy()
            if ale_var is not None:
                ales[m, idx_np] = ale_var.float().cpu().numpy()
        for j, ii in enumerate(idx_np.tolist()):
            if filenames[ii] is None:
                filenames[ii] = fn[j]

    return mus, ales, filenames


def ttt_adapt_all_folds(backbone, fold_headsets, loader, mus, ales, folds_to_use):
    M = len(folds_to_use)
    mu_t_list = []
    var_t_list = []
    for i in range(M):
        mu_t, var_t = loo_pseudo_targets(mus, ales, leave_out=i)
        mu_t_list.append(torch.from_numpy(mu_t))
        var_t_list.append(torch.from_numpy(var_t))

    optimizers = {}
    source_states = {}
    for fold in folds_to_use:
        headsets = fold_headsets.get(fold)
        if not headsets:
            continue
        for headset in headsets:
            set_trainable_for_ttt(headset)
            headset.eval()
        opt_a = torch.optim.AdamW(
            [p for p in headsets[0].parameters() if p.requires_grad],
            lr=float(CFG.TTT_LR),
            weight_decay=float(CFG.TTT_WD),
        )
        opt_b = torch.optim.AdamW(
            [p for p in headsets[-1].parameters() if p.requires_grad],
            lr=float(CFG.TTT_LR),
            weight_decay=float(CFG.TTT_WD),
        )
        optimizers[fold] = (opt_a, opt_b)
        source_states[fold] = (
            {k: v.detach().cpu().clone() for k, v in headsets[0].state_dict().items()},
            {k: v.detach().cpu().clone() for k, v in headsets[-1].state_dict().items()},
        )

    for epoch in range(int(CFG.TTT_EPOCHS)):
        for it, (idx, left, right, _) in enumerate(loader):
            idx_np = idx.numpy()
            left = left.to(CFG.DEVICE, non_blocking=True)
            right = right.to(CFG.DEVICE, non_blocking=True)

            with torch.no_grad():
                with autocast_ctx():
                    x_l = backbone(left)
                    x_r = backbone(right)
            x_l = x_l.detach()
            x_r = x_r.detach()

            for m, fold in enumerate(folds_to_use):
                headsets = fold_headsets.get(fold)
                if not headsets:
                    continue
                opt_a, opt_b = optimizers[fold]
                src_a, src_b = source_states[fold]
                target = mu_t_list[m][idx_np].to(CFG.DEVICE, non_blocking=True)
                var = var_t_list[m][idx_np].to(CFG.DEVICE, non_blocking=True)

                with autocast_ctx():
                    pred_a, _ = forward_pred_and_alevar_tokens(headsets[0], x_l, x_r)
                    pred_b, _ = forward_pred_and_alevar_tokens(headsets[-1], x_l, x_r)
                    loss_a = fixed_var_nll(pred_a, target, var, w=CFG.LOSS_WEIGHTS) / float(CFG.TTT_GRAD_ACC)
                    loss_b = fixed_var_nll(pred_b, target, var, w=CFG.LOSS_WEIGHTS) / float(CFG.TTT_GRAD_ACC)

                if scaler.is_enabled():
                    scaler.scale(loss_a).backward()
                    scaler.scale(loss_b).backward()
                else:
                    loss_a.backward()
                    loss_b.backward()

                do_step = ((it + 1) % int(CFG.TTT_GRAD_ACC) == 0) or ((it + 1) == len(loader))
                if do_step:
                    if scaler.is_enabled():
                        scaler.unscale_(opt_a)
                        scaler.unscale_(opt_b)
                    torch.nn.utils.clip_grad_norm_(headsets[0].parameters(), float(CFG.TTT_CLIP_NORM))
                    torch.nn.utils.clip_grad_norm_(headsets[-1].parameters(), float(CFG.TTT_CLIP_NORM))
                    if scaler.is_enabled():
                        scaler.step(opt_a)
                        scaler.step(opt_b)
                        scaler.update()
                    else:
                        opt_a.step()
                        opt_b.step()
                    opt_a.zero_grad(set_to_none=True)
                    opt_b.zero_grad(set_to_none=True)
                    maybe_restore_heads(headsets[0], src_a, p_restore=float(CFG.TTT_RESTORE_PROB))
                    maybe_restore_heads(headsets[-1], src_b, p_restore=float(CFG.TTT_RESTORE_PROB))


def ucvme_adapt_all_folds(backbone, fold_headsets, loader_lb, loader_ulb, folds_to_use):
    if not CFG.USE_HETERO:
        raise RuntimeError("UCVME requires USE_HETERO=True.")
    optimizers = {}
    source_states = {}
    ema_trackers = {}
    for fold in folds_to_use:
        headsets = fold_headsets.get(fold)
        if not headsets:
            continue
        for headset in headsets:
            set_trainable_for_ucvme(headset)
            headset.train()
        opt_a = torch.optim.AdamW(
            [p for p in headsets[0].parameters() if p.requires_grad],
            lr=float(CFG.TTT_LR),
            weight_decay=float(CFG.TTT_WD),
        )
        opt_b = torch.optim.AdamW(
            [p for p in headsets[-1].parameters() if p.requires_grad],
            lr=float(CFG.TTT_LR),
            weight_decay=float(CFG.TTT_WD),
        )
        optimizers[fold] = (opt_a, opt_b)
        source_states[fold] = (
            {k: v.detach().cpu().clone() for k, v in headsets[0].state_dict().items()},
            {k: v.detach().cpu().clone() for k, v in headsets[-1].state_dict().items()},
        )
        same_head = headsets[0] is headsets[-1]
        ema_a = ModelEmaV2(headsets[0], decay=float(CFG.EMA_DECAY), device=CFG.DEVICE)
        ema_b = ema_a if same_head else ModelEmaV2(headsets[-1], decay=float(CFG.EMA_DECAY), device=CFG.DEVICE)
        ema_trackers[fold] = (ema_a, ema_b, same_head)

    grad_acc = int(CFG.TTT_GRAD_ACC)
    ucvme_epochs = int(getattr(CFG, "UCVME_EPOCHS", 1))
    for epoch in range(ucvme_epochs):
        ulb_iter = iter(loader_ulb) if loader_ulb is not None else None
        for it, batch in enumerate(loader_lb):
            if len(batch) == 5:
                idx, left, right, labels, meta = batch
            else:
                idx, left, right, labels = batch
                meta = None
            left = left.to(CFG.DEVICE, non_blocking=True)
            right = right.to(CFG.DEVICE, non_blocking=True)
            labels = labels.to(CFG.DEVICE, non_blocking=True)
            meta = _move_meta(meta, CFG.DEVICE)

            with torch.no_grad():
                with autocast_ctx():
                    x_l = backbone(left)
                    x_r = backbone(right)
            x_l = x_l.detach()
            x_r = x_r.detach()

            if ulb_iter is not None:
                try:
                    _, ulb_left, ulb_right, _ = next(ulb_iter)
                except StopIteration:
                    ulb_iter = iter(loader_ulb)
                    _, ulb_left, ulb_right, _ = next(ulb_iter)
                ulb_left = ulb_left.to(CFG.DEVICE, non_blocking=True)
                ulb_right = ulb_right.to(CFG.DEVICE, non_blocking=True)
                with torch.no_grad():
                    with autocast_ctx():
                        x_ulb_l = backbone(ulb_left)
                        x_ulb_r = backbone(ulb_right)
                x_ulb_l = x_ulb_l.detach()
                x_ulb_r = x_ulb_r.detach()
            else:
                x_ulb_l = None
                x_ulb_r = None

            for fold in folds_to_use:
                headsets = fold_headsets.get(fold)
                if not headsets:
                    continue
                opt_a, opt_b = optimizers[fold]
                src_a, src_b = source_states[fold]

                with autocast_ctx():
                    pred_a, logvar_a, meta_a = forward_pred_logvar_meta(headsets[0], x_l, x_r)
                    pred_b, logvar_b, meta_b = forward_pred_logvar_meta(headsets[-1], x_l, x_r)
                    loss_reg_a = hetero_nll_loss(pred_a, logvar_a, labels, w=CFG.LOSS_WEIGHTS)
                    loss_reg_b = hetero_nll_loss(pred_b, logvar_b, labels, w=CFG.LOSS_WEIGHTS)
                    loss_reg_lb = 0.5 * (loss_reg_a + loss_reg_b)
                    loss_unc_lb = _ucvme_uncertainty_consistency(logvar_a, logvar_b)
                    loss = loss_reg_lb
                    if loss_unc_lb is not None:
                        loss = loss + loss_unc_lb
                    loss_meta_a = compute_meta_loss(meta_a, meta)
                    loss_meta_b = compute_meta_loss(meta_b, meta)
                    if loss_meta_a is not None and loss_meta_b is not None:
                        loss = loss + 0.5 * (loss_meta_a + loss_meta_b)
                    elif loss_meta_a is not None:
                        loss = loss + loss_meta_a
                    elif loss_meta_b is not None:
                        loss = loss + loss_meta_b

                    if x_ulb_l is not None and x_ulb_r is not None:
                        pseudo_y, pseudo_z = _ucvme_pseudo_labels_headsets(
                            headsets[0], headsets[-1], x_ulb_l, x_ulb_r, int(CFG.UCVME_MC_SAMPLES)
                        )
                        pred_u_a, logvar_u_a = forward_pred_and_logvar_tokens(headsets[0], x_ulb_l, x_ulb_r)
                        pred_u_b, logvar_u_b = forward_pred_and_logvar_tokens(headsets[-1], x_ulb_l, x_ulb_r)
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
                    opt_a.zero_grad(set_to_none=True)
                    opt_b.zero_grad(set_to_none=True)
                    continue

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                do_step = ((it + 1) % grad_acc == 0) or ((it + 1) == len(loader_lb))
                if do_step:
                    if scaler.is_enabled():
                        scaler.unscale_(opt_a)
                        scaler.unscale_(opt_b)
                    torch.nn.utils.clip_grad_norm_(headsets[0].parameters(), float(CFG.TTT_CLIP_NORM))
                    torch.nn.utils.clip_grad_norm_(headsets[-1].parameters(), float(CFG.TTT_CLIP_NORM))
                    if scaler.is_enabled():
                        scaler.step(opt_a)
                        scaler.step(opt_b)
                        scaler.update()
                    else:
                        opt_a.step()
                        opt_b.step()
                    ema_a, ema_b, same_head = ema_trackers[fold]
                    ema_a.update(headsets[0])
                    if not same_head:
                        ema_b.update(headsets[-1])
                    opt_a.zero_grad(set_to_none=True)
                    opt_b.zero_grad(set_to_none=True)
                    maybe_restore_heads(headsets[0], src_a, p_restore=float(CFG.TTT_RESTORE_PROB))
                    maybe_restore_heads(headsets[-1], src_b, p_restore=float(CFG.TTT_RESTORE_PROB))

    for fold in folds_to_use:
        headsets = fold_headsets.get(fold)
        if not headsets:
            continue
        ema_a, ema_b, same_head = ema_trackers.get(fold, (None, None, False))
        if ema_a is not None:
            headsets[0].load_state_dict(ema_a.module.state_dict(), strict=False)
        if not same_head and ema_b is not None:
            headsets[-1].load_state_dict(ema_b.module.state_dict(), strict=False)
        for headset in headsets:
            headset.eval()


def run_inference_ttt_shared_backbone():
    transform = get_inference_transform()
    ds_test = BiomassUnlabeledTensorDataset(CFG.TEST_IMAGE_DIR, transform)
    loader_ulb = DataLoader(
        ds_test,
        batch_size=int(CFG.TTT_BATCH_SIZE),
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )
    loader_ttt = DataLoader(
        ds_test,
        batch_size=int(CFG.TTT_BATCH_SIZE),
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    folds_to_use = CFG.FOLDS_TO_USE or list(range(CFG.N_FOLDS))
    backbone, fold_headsets = load_fold_headsets(folds_to_use)

    if getattr(CFG, "UCVME_MODE", False):
        train_df = _load_train_wide()
        meta_info = build_meta_info(train_df)
        ds_train = BiomassLabeledTensorDataset(
            train_df,
            CFG.TRAIN_IMAGE_DIR,
            transform,
            meta_info=meta_info,
            return_meta=bool(meta_info.get("enabled")),
        )
        loader_lb = DataLoader(
            ds_train,
            batch_size=int(CFG.TTT_BATCH_SIZE),
            shuffle=True,
            drop_last=True,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
        )
        ucvme_adapt_all_folds(backbone, fold_headsets, loader_lb, loader_ulb, folds_to_use)
    if getattr(CFG, "TTT_MODE", False):
        ttt_passes = int(getattr(CFG, "TTT_PASSES", 1))
        if ttt_passes < 1:
            ttt_passes = 1
        for _ in range(ttt_passes):
            mus, ales, _ = collect_fold_preds(backbone, fold_headsets, loader_ttt, folds_to_use)
            ttt_adapt_all_folds(backbone, fold_headsets, loader_ttt, mus, ales, folds_to_use)

    pred_dataset = BiomassRawTestDataset(CFG.TEST_IMAGE_DIR)
    filenames = pred_dataset.filenames
    tta_transforms = get_tta_transforms(CFG.TTA_STEPS)
    print(f"Using TTA with {len(tta_transforms)} views.")

    pred_cols = len(CFG.ALL_TARGET_COLS)
    accumulated_preds = np.zeros((len(pred_dataset), pred_cols), dtype=np.float32)
    successful_folds = 0
    for fold in folds_to_use:
        if fold in fold_headsets:
            successful_folds += 1

    if successful_folds == 0:
        raise FileNotFoundError("No fold headsets available for TTT inference.")

    pred_loader = DataLoader(
        pred_dataset, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True
    )
    denom = float(successful_folds * len(tta_transforms))
    with torch.no_grad():
        for idx, left, right, _ in tqdm(pred_loader, desc="predict"):
            idx_np = idx.numpy()
            left_np = left[0].numpy()
            right_np = right[0].numpy()
            pred_sum = _predict_tta_sum(
                backbone, fold_headsets, folds_to_use, left_np, right_np, tta_transforms
            )
            accumulated_preds[idx_np] += pred_sum.float().cpu().numpy()

    final_predictions = accumulated_preds / denom
    return final_predictions, filenames


def postprocess_predictions(preds_direct):
    if preds_direct.shape[1] == len(CFG.ALL_TARGET_COLS):
        preds_all = preds_direct.copy()
    else:
        pred_total = preds_direct[:, 0]
        pred_gdm = preds_direct[:, 1]
        pred_green = preds_direct[:, 2]
        pred_clover = np.maximum(0, pred_gdm - pred_green)
        pred_dead = np.maximum(0, pred_total - pred_gdm)
        preds_all = np.stack([pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)

    preds_all = np.maximum(preds_all, 0.0)
    try:
        clover_idx = CFG.ALL_TARGET_COLS.index("Dry_Clover_g")
    except ValueError:
        clover_idx = 2
    clover = preds_all[:, clover_idx]
    preds_all[:, clover_idx] = np.where(clover < 1.0, 0.0, clover)
    return _apply_consistency_with_fixed(preds_all)


_CONSIST_A = np.array(
    [
        [-1.0, 0.0, -1.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, -1.0, 1.0],
    ],
    dtype=np.float32,
)


def _apply_consistency_with_fixed(preds_all: np.ndarray) -> np.ndarray:
    if preds_all.ndim != 2 or preds_all.shape[1] != len(CFG.ALL_TARGET_COLS):
        raise ValueError(f"Expected (N, {len(CFG.ALL_TARGET_COLS)}) preds, got {preds_all.shape}")

    try:
        clover_idx = CFG.ALL_TARGET_COLS.index("Dry_Clover_g")
    except ValueError:
        clover_idx = None

    preds_all = preds_all.astype(np.float32, copy=True)
    for i in range(preds_all.shape[0]):
        x = preds_all[i]
        fixed = x == 0.0
        for _ in range(x.size):
            free_idx = np.flatnonzero(~fixed)
            if free_idx.size == 0:
                break
            A_f = _CONSIST_A[:, free_idx]
            if A_f.size == 0:
                break
            if fixed.any():
                A_c = _CONSIST_A[:, fixed]
                b = -(A_c @ x[fixed])
            else:
                b = np.zeros((_CONSIST_A.shape[0],), dtype=x.dtype)
            x_f0 = x[free_idx]
            M = A_f @ A_f.T
            if not np.isfinite(M).all() or np.allclose(M, 0.0):
                break
            diff = A_f @ x_f0 - b
            lam = np.linalg.pinv(M) @ diff
            x_f = x_f0 - A_f.T @ lam
            x[free_idx] = x_f
            if clover_idx is not None and not fixed[clover_idx] and x[clover_idx] < 1.0:
                x[clover_idx] = 0.0
                fixed[clover_idx] = True
                continue
            neg = x[free_idx] < 0.0
            if not np.any(neg):
                break
            neg_idx = free_idx[neg]
            x[neg_idx] = 0.0
            fixed[neg_idx] = True
        x[x < 0.0] = 0.0
        preds_all[i] = x
    return preds_all


def create_submission(predictions, filenames):
    """
    Create submission file in the required format.

    Args:
        predictions: (n_images, 5) array with all target predictions
        filenames: list of test image filenames
    """
    print("\n" + "=" * 70)
    print("CREATING SUBMISSION FILE")
    print("=" * 70)

    test_df = pd.read_csv(CFG.TEST_CSV)
    print(f"\nTest CSV loaded: {len(test_df)} rows")
    print(f"Sample image_path from test.csv: {test_df['image_path'].iloc[0]}")
    print(f"Sample filename from predictions: {filenames[0]}")

    test_path_example = test_df["image_path"].iloc[0]
    if "/" in test_path_example:
        prefix = test_path_example.rsplit("/", 1)[0] + "/"
        corrected_filenames = [prefix + fn for fn in filenames]
        print(f"Corrected path format: {corrected_filenames[0]}")
    else:
        corrected_filenames = filenames

    preds_wide = pd.DataFrame(predictions, columns=CFG.ALL_TARGET_COLS)
    preds_wide.insert(0, "image_path", corrected_filenames)

    print(f"\nWide format predictions:")
    print(preds_wide.head())

    preds_long = preds_wide.melt(
        id_vars=["image_path"],
        value_vars=CFG.ALL_TARGET_COLS,
        var_name="target_name",
        value_name="target",
    )

    print(f"\nLong format predictions (first 10 rows):")
    print(preds_long.head(10))

    print(f"\nDebug: Checking if paths match...")
    print(f"Unique paths in test_df: {test_df['image_path'].nunique()}")
    print(f"Unique paths in preds_long: {preds_long['image_path'].nunique()}")

    common_paths = set(test_df["image_path"].unique()) & set(preds_long["image_path"].unique())
    print(f"Common paths found: {len(common_paths)}")

    if len(common_paths) == 0:
        print("\n❌ ERROR: No matching paths found!")
        print(f"Test CSV paths sample: {list(test_df['image_path'].unique()[:3])}")
        print(f"Prediction paths sample: {list(preds_long['image_path'].unique()[:3])}")
        raise ValueError("Path mismatch between test.csv and predictions")

    submission = pd.merge(
        test_df[["sample_id", "image_path", "target_name"]],
        preds_long,
        on=["image_path", "target_name"],
        how="left",
    )

    submission = submission[["sample_id", "target"]]

    missing_count = submission["target"].isna().sum()
    if missing_count > 0:
        print(f"\n⚠ Warning: {missing_count} missing predictions found!")
        print("Sample missing entries:")
        print(submission[submission["target"].isna()].head())
        submission.loc[submission["target"].isna(), "target"] = 0.0

    submission = submission.sort_values("sample_id").reset_index(drop=True)

    output_path = os.path.join(CFG.SUBMISSION_DIR, "submission.csv")
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

    print(f"\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    print(f"✓ Expected rows: {len(test_df)}")
    print(f"✓ Actual rows: {len(submission)}")
    print(f"✓ Match: {len(submission) == len(test_df)}")
    print(f"✓ No missing values: {not submission['target'].isna().any()}")
    print(f"✓ All sample_ids unique: {submission['sample_id'].is_unique}")
    print(f"✓ Has non-zero predictions: {(submission['target'] > 0).any()}")

    return submission


if __name__ == "__main__":
    if CFG.CREATE_SUBMISSION:
        if getattr(CFG, "TTT_MODE", False) or getattr(CFG, "UCVME_MODE", False):
            predictions_direct, test_filenames = run_inference_ttt_shared_backbone()
        else:
            predictions_direct, test_filenames = run_inference_shared_backbone()
        predictions_all = postprocess_predictions(predictions_direct)
        create_submission(predictions_all, test_filenames)
