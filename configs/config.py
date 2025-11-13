"""
Configuration file for WatchSleepNet Reproduction
Centralizes all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PARAMETERS
# ============================================================================
# Sampling rates from DREAMT dataset
SAMPLING_RATES = {
    'timestamp': 64,  # Hz
    'bvp': 64,        # Hz
    'acc': 32,        # Hz (triaxial)
    'temp': 4,        # Hz
    'eda': 4,         # Hz
    'hr': 1,          # Hz
    'ibi': None,      # Variable (beat-to-beat)
    'sleep_stage': 1/30  # Every 30 seconds
}

# Sleep stage mapping
SLEEP_STAGES = {
    'P': -1,      # Preparation (will be filtered out)
    'W': 0,       # Wake
    'N1': 1,      # NREM Stage 1
    'N2': 2,      # NREM Stage 2
    'N3': 3,      # NREM Stage 3
    'R': 4,       # REM
    'Missing': -2 # Missing labels
}

# Simplified 3-class mapping (like original WatchSleepNet paper)
SLEEP_STAGES_SIMPLIFIED = {
    'W': 0,       # Wake
    'N1': 1,      # NREM (combine N1, N2, N3)
    'N2': 1,
    'N3': 1,
    'R': 2        # REM
}

# Features to use
FEATURES_TO_USE = ['ACC_X', 'ACC_Y', 'ACC_Z', 'HR']  # Primary features from paper
OPTIONAL_FEATURES = ['BVP', 'TEMP', 'EDA']  # For extensions

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
# Window parameters for creating sequences
EPOCH_LENGTH = 30  # seconds (matches sleep stage annotation frequency)
WINDOW_SIZE = 20   # epochs (10 minutes = 20 * 30 seconds) - from paper
OVERLAP = 0.5      # 50% overlap between windows

# Target sampling rate for resampling (if needed)
TARGET_ACC_RATE = 32  # Hz
TARGET_HR_RATE = 1    # Hz

# Filtering parameters
FILTER_PREP_STAGE = True  # Remove 'P' labels
FILTER_MISSING = True     # Remove 'Missing' labels

# ============================================================================
# MODEL PARAMETERS (matching your WatchSleepNet implementation)
# ============================================================================
# Input dimensions
NUM_FEATURES = 1  # Will be 4 after combining ACC + HR
NUM_SAMPLES_PER_SEGMENT = 960  # 30 seconds at 32Hz
MAX_NUM_SEGMENTS = 20  # WINDOW_SIZE from above

# WatchSleepNet architecture parameters
NUM_CHANNELS = 64  # TCN channels
KERNEL_SIZE = 3
HIDDEN_DIM = 128  # LSTM hidden dimension
NUM_HEADS = 4  # Multi-head attention heads
NUM_LAYERS = 2  # LSTM layers
TCN_LAYERS = 3  # Number of TCN layers

# Ablation options
USE_TCN = True  # Use Temporal Convolutional Network
USE_ATTENTION = True  # Use Multi-head Attention

# Output
N_CLASSES = 3  # Wake, NREM, REM (simplified)
# N_CLASSES = 5  # W, N1, N2, N3, R (detailed) - uncomment for full classification

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Optimizer
OPTIMIZER = 'adam'  # 'adam', 'sgd', 'adamw'
WEIGHT_DECAY = 1e-4

# Loss function
LOSS_FUNCTION = 'cross_entropy'  # Can try 'focal_loss' for imbalanced data

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'reduce_on_plateau'  # 'step', 'cosine', 'reduce_on_plateau'
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa']
COMPUTE_CONFUSION_MATRIX = True

# Cross-validation
USE_CROSS_VALIDATION = True
N_FOLDS = 5  # With 5-6 patients, use leave-one-out or 5-fold

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
SAVE_MODEL = True
SAVE_PREDICTIONS = True
SAVE_FIGURES = True

# Random seed for reproducibility
RANDOM_SEED = 42

# Device
DEVICE = 'cuda'  # Will auto-detect in training script

# ============================================================================
# LOGGING
# ============================================================================
LOG_INTERVAL = 10  # Log every N batches
VERBOSE = True

print(f"Configuration loaded successfully!")
print(f"Project root: {PROJECT_ROOT}")
print(f"Using {N_CLASSES} sleep stage classes")
print(f"Primary features: {FEATURES_TO_USE}")