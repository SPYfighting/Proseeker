# ========= Basic Settings ==========
import os

BASE_ESM_MODEL = os.getenv("BASE_ESM_MODEL", "facebook/esm2_t33_650M_UR50D")
DEVICE = os.getenv("DEVICE", "cuda")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# ========== LoRA Settings =========
LORA_ENABLED = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]

# ========== Paths ==========
DATA_DIR = os.getenv("DATA_DIR", "data")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")

PATH_HOMOLOGOUS_FASTA = os.path.join(DATA_DIR, "homologous_sequences.fasta")
PATH_LABELED_DATA = os.path.join(DATA_DIR, "labeled_data.csv")
PATH_TRAINING_PAIRS = os.path.join(DATA_DIR, "training_pairs.csv")
PATH_CANDIDATE_SEQUENCES = os.path.join(DATA_DIR, "candidates.csv")
PATH_MEASURED_DATA = os.path.join(DATA_DIR, "measured_pairs_round1.csv")

PATH_FINAL_PREDICTIONS = os.path.join(OUTPUTS_DIR, "predictions.csv")
PATH_BEST_HYPERPARAMS = os.path.join(OUTPUTS_DIR, "best_hparams.json")

DIR_MLM_TUNED_MODEL = os.getenv("DIR_MLM_TUNED_MODEL", os.path.join(OUTPUTS_DIR, "mlm_finetune_lora"))
DIR_ENSEMBLE_MODELS = os.getenv("DIR_ENSEMBLE_MODELS", os.path.join(OUTPUTS_DIR, "ensemble"))

# Training Control
MAX_LEN = 512

# Hyperparameter Search
HSEARCH_N_TRIALS = int(os.getenv("HSEARCH_N_TRIALS", 30))
HSEARCH_EPOCHS_PER_TRIAL = int(os.getenv("HSEARCH_EPOCHS_PER_TRIAL", 3))

# Ensemble
FINETUNE_N_ENSEMBLE = int(os.getenv("FINETUNE_N_ENSEMBLE", 5))
FINETUNE_FINAL_EPOCHS = int(os.getenv("FINETUNE_FINAL_EPOCHS", 10))
PREDICTION_BATCH_SIZE = int(os.getenv("PREDICTION_BATCH_SIZE", 16))

# MC Dropout & Acquisition
USE_MC_DROPOUT = os.getenv("USE_MC_DROPOUT", "1") == "1"
MC_DROPOUT_PASSES = int(os.getenv("MC_DROPOUT_PASSES", 10))
ACQ_TEMPERATURE = float(os.getenv("ACQ_TEMPERATURE", 0.5))

# Iterative Optimization
ITER_MICRO_STEPS = int(os.getenv("ITER_MICRO_STEPS", 200))
ITER_PARENT_SEQUENCE = os.getenv("ITER_PARENT_SEQUENCE", "SLPLNMPALEMPAFIATKVSQYSCQ...")
