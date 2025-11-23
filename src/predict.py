import pandas as pd
import numpy as np
import tensorflow as tf

from src.config import (
    VAL_CSV,
    VAL_IMAGES_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RESULTS_DIR,
)
from src.preprocessing import load_preprocessors, transform_metadata
from src.data_loader import ImageMetaSequence
import os


def main():
    print("Loading validation CSV:", VAL_CSV)
    val_df = pd.read_csv(VAL_CSV)
    ids = val_df["id_global"].values

    print("Loading preprocessors:", PREPROCESSOR_PATH)
    preprocessors = load_preprocessors(PREPROCESSOR_PATH)
    X_meta_val = transform_metadata(val_df, preprocessors)

    print("Creating validation sequence...")
    val_seq = ImageMetaSequence(
        ids=ids,
        metadata_array=X_meta_val,
        labels=None,
        images_dir=VAL_IMAGES_DIR,
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        shuffle=False,
    )

    print("Loading trained model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)


    print("Predicting on validation set...")
    preds = model.predict(val_seq, verbose=1)

    # preds: [Temperature, Tint]
    submission = pd.DataFrame(
        {
            "id_global": ids,
            "Temperature": np.round(preds[:, 0]).astype(int),
            "Tint": np.round(preds[:, 1]).astype(int),
        }
    )

    out_path = os.path.join(RESULTS_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print("Saved submission file to:", out_path)


if __name__ == "__main__":
    main()
