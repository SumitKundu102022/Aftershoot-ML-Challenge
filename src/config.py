import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VAL_DIR = os.path.join(DATA_DIR, "Validation")

TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "Images")
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "Images")

TRAIN_CSV = os.path.join(TRAIN_DIR, "sliders.csv")
VAL_CSV = os.path.join(VAL_DIR, "sliders_input.csv")


# Model / artifacts
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessors.pkl")

# Image / training config
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

BATCH_SIZE = 16
EPOCHS = 20
RANDOM_STATE = 42
