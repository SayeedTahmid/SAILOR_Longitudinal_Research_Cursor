"""Locked scientific and filesystem constants."""

from __future__ import annotations

DATA_VERSION = "v2.0"
PREPROCESSING_VERSION = "p2.0"
IMPLEMENTATION_ID = "cursor_primary"
PROJECT_NAME = "SAILOR_Longitudinal_Research_Cursor"
PRODUCTION_DATASET_ROOT = f"/content/drive/MyDrive/{PROJECT_NAME}"
PRODUCTION_LEGACY_ROOT = "/content/drive/MyDrive/sailor_v1"

PRIMARY_TARGET_MASK = "CL"
PRIMARY_TARGET_COMPONENT = "enhancing_t1wc"
SECONDARY_TARGET_MASK = "ONCO"
SENSITIVITY_TARGETS = ("CL:t2wflair_hyperintensity",)

PRIMARY_INPUT_SEQUENCE = "T1c-icor"
FOLD_SCHEME = "5fold_x3seeds_nested4"
OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 4
MIN_HISTORY_SCANS = 2

CANONICAL_FILES = (
    "data-descriptor_a866425efff8.pdf",
    "README.txt",
    "SHA512.txt",
    "overview.tsv",
    "missing.tsv",
    "src-to-raw.yaml",
    "code.tar.bz2",
    "rawdata_BIDS.tar.bz2",
    "derivatives.tar.bz2",
)

KNOWN_MISSING_CANONICAL = (
    "sourcedata.tar.bz2",
    "rawdata.tar.bz2",
    "rawdata_BIDS_ext.tar.bz2",
)

QUARANTINE_NAMES = (
    "tadiff_npy",
    "ckpt_dose",
    "ckpt_nodose",
    "ckpt_finetune",
    "_workdir",
    "unet_v1.pt",
    "unet_timecond_v1.pt",
    "unet_timecond_v2.pt",
    "autoencoder_v1.pt",
    "latents_v1.npz",
    "pairs_v1.npz",
    "sailor_slices_v1.h5",
    "sub-17_image.npy",
    "split_v1.json",
    "split_tadiff.json",
    "session_whitelist.json",
    "persistence_baseline.json",
)

AMBIGUOUS_NAMES = ("raw_needed.tar", "dosemaps.tar")

OUTPUT_DIRECTORIES = (
    "00_CANONICAL",
    "00_QUARANTINE",
    "01_DATA_FOUNDATION",
    "02_PREPROCESSED_MRI",
    "03_TUMOR_MASKS",
    "04_LONGITUDINAL_WINDOWS",
    "05_TREATMENT_DATA",
    "06_QC_REPORTS",
    "07_BASELINE_RESULTS",
    "08_FEATURES",
    "09_MODEL_OUTPUTS",
    "10_EXPERIMENTS",
    "CHECKPOINTS",
    "LOGS",
    "RESULTS",
)

SECTION_STAGE = {
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
}
