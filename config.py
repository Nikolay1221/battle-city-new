# --- SYSTEM & HARDWARE ---
import multiprocessing
import os

NUM_CPU = 8  # Используем все ядра для сбора опыта

# --- HYBRID CONFIGURATION ---
NUM_VIRTUAL = 2

HEADLESS_MODE = False

# --- TRAINING DURATION ---
TOTAL_TIMESTEPS = 1_400_000

# --- OBSERVATION ---
STACK_SIZE = 4
FRAME_SKIP = 4
ROM_PATH = "BattleCity.nes"
TARGET_STAGE = 0

USE_RECURRENT = False
USE_TRANSFORMER = False

# --- REWARDS ---
REW_KILL = 1.0
REW_DEATH = -1.0
REW_BASE = -20.0
REW_EXPLORE = 0.01
REW_WIN = 20.0

# --- TRAINING MODES ---
# Define presets for environment variants
ENV_VARIANTS = {
    "STANDARD":        {"enemy_count": 20, "no_shooting": False},
    "PEACEFUL":        {"enemy_count": 0,  "no_shooting": True},
    "TARGET_PRACTICE": {"enemy_count": 0,  "no_shooting": False}, # Walls only
    "VERY_EASY":       {"enemy_count": 2,  "no_shooting": False}, # Requested: 2 enemies
    "LIGHT_COMBAT":    {"enemy_count": 5,  "no_shooting": False},
    "MEDIUM_COMBAT":   {"enemy_count": 10, "no_shooting": False},
    "VIRTUAL":         "VIRTUAL" 
}

TRAIN_MODE = "HYBRID"

# Hybrid Distribution (Sum must be 8)
# Curriculum-style distribution:
HYBRID_CONFIG = {
    "PEACEFUL": 1,          # #1: Just walking
    "TARGET_PRACTICE": 1,   # #2: Shooting walls
    "VERY_EASY": 1,         # #3: 2 Enemies
    "LIGHT_COMBAT": 1,      # #4: 5 Enemies
    "MEDIUM_COMBAT": 1,     # #5: 10 Enemies
    "STANDARD": 3           # #6-8: Full War (20 Enemies)
}

# --- OBSOLETE FLAGS ---
USE_VIRTUAL = False
USE_HYBRID = False

# --- SAVING ---
CHECKPOINT_FREQ = 2_000
MODEL_DIR = "models"
LOG_DIR = "logs"

# --- PPO HYPERPARAMETERS (КОНФИГ ДЛЯ ЧИСТОГО СТАРТА) ---
# Adaptive Learning Rate Settings
# Для старта с нуля 0.00005 - это слишком мало. Агент не поймет связи действий и наград.
LR_START = 0.0003  # (3e-4) Золотой стандарт Карпатого. Идеально для старта.
LR_MIN = 0.0001  # (1e-4) Если станет слишком сложно, он притормозит.
LR_MAX = 0.0005  # (5e-4) Позволяем разгоняться, если пойдет хорошо.
TARGET_KL = 0.03   

# --- MEMORY TRICK (VRAM SAVER) ---
# N_STEPS = 4096 критически важен для старта с нуля!
# В начале агент делает рандом. Чем больше буфер, тем выше шанс,
# что в рандоме попадется убийство врага, и сеть научится.
N_STEPS       = 2048    
BATCH_SIZE    = 1024 
# Если вылетит ошибка, ставь 128, но 256 лучше.

N_EPOCHS = 2  # Учимся усердно на каждом куске данных.
ENT_COEF = 0.01  # Для старта с нуля оставляем 0.05!
# Ему нужно много экспериментировать в начале.
GAMMA = 0.99
CLIP_RANGE = 0.2  # Даем свободу изменений.
ALLOW_NEW_MODEL = True