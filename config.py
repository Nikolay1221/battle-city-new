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

USE_RECURRENT = True
USE_TRANSFORMER = False

# --- REWARDS ---
REW_KILL = 1.0
REW_DEATH = -1.0
REW_BASE = -20.0
REW_EXPLORE = 0.01
REW_WIN = 50.0

# --- VIRTUAL TRAINING ---
USE_VIRTUAL = False
USE_HYBRID = True

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
TARGET_KL = 0.015  # Цель: менять мозг на 1.5% за шаг.

# --- MEMORY TRICK (VRAM SAVER) ---
# N_STEPS = 4096 критически важен для старта с нуля!
# В начале агент делает рандом. Чем больше буфер, тем выше шанс,
# что в рандоме попадется убийство врага, и сеть научится.
N_STEPS = 4096  # Хранится в RAM (ОЗУ). Дает стабильность.

BATCH_SIZE = 128  # Хранится в VRAM (Видеокарта 4GB).
# Если вылетит ошибка, ставь 128, но 256 лучше.

N_EPOCHS = 10  # Учимся усердно на каждом куске данных.
ENT_COEF = 0.05  # Для старта с нуля оставляем 0.05!
# Ему нужно много экспериментировать в начале.
GAMMA = 0.99
CLIP_RANGE = 0.2  # Даем свободу изменений.
ALLOW_NEW_MODEL = True