# ==========================================
#         BATTLE CITY AI CONFIG
# ==========================================

# --- SYSTEM & HARDWARE ---
# Number of parallel environments. 
# On Colab (2 cores), keep this around 8-16. 
# On powerful PC (24 cores), go for 48-64.
NUM_CPU = 48 

# --- TRAINING DURATION ---
# Total number of frames/steps to train for.
# 1 Million = ~1 hour on fast PC.
# 5 Million = Good for "Master" level.
TOTAL_TIMESTEPS = 5_000_000 

# --- SAVING ---
# How often to save the model (in steps).
CHECKPOINT_FREQ = 10_000
MODEL_DIR = "models"
LOG_DIR = "logs"

# --- PPO HYPERPARAMETERS ---
# Advanced users only.
LEARNING_RATE = 0.00001  # Slow & Careful
N_STEPS       = 2048     # Increased for higher FPS (less frequent updates)
BATCH_SIZE    = 512      # Increased Batch size
ENT_COEF      = 0.1     # Entropy: Lowered to 0.05 to stabilize mastery.
GAMMA         = 0.99     # Discount factor

# --- SAFETY ---
ALLOW_NEW_MODEL = False # If True, will restart from scratch if no model found. If False, CRASHES.
