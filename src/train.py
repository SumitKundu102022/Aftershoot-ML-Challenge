import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_CSV,
    TRAIN_IMAGES_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    EPOCHS,
    RANDOM_STATE,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)
from src.preprocessing import fit_preprocessors, transform_metadata, save_preprocessors
from src.data_loader import ImageMetaSequence
from src.model_def import build_model


def main():
    # Reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print("Loading training CSV:", TRAIN_CSV)
    train_df = pd.read_csv(TRAIN_CSV)

    # Targets
    y = train_df[["Temperature", "Tint"]].values
    ids = train_df["id_global"].values

    # Fit preprocessors on training metadata
    print("Fitting metadata preprocessors...")
    preprocessors = fit_preprocessors(train_df)
    X_meta = transform_metadata(train_df, preprocessors)

    # Save preprocessors for prediction script
    save_preprocessors(preprocessors, PREPROCESSOR_PATH)
    print(f"Saved preprocessors to {PREPROCESSOR_PATH}")

    # Split into train/val
    idx = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.1, random_state=RANDOM_STATE, shuffle=True
    )

    train_seq = ImageMetaSequence(
        ids=ids[train_idx],
        metadata_array=X_meta[train_idx],
        labels=y[train_idx],
        images_dir=TRAIN_IMAGES_DIR,
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        shuffle=True,
    )

    val_seq = ImageMetaSequence(
        ids=ids[val_idx],
        metadata_array=X_meta[val_idx],
        labels=y[val_idx],
        images_dir=TRAIN_IMAGES_DIR,
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        shuffle=False,
    )

    # Build model
    meta_dim = X_meta.shape[1]
    print("Building model with metadata dim:", meta_dim)
    model = build_model(
        meta_input_dim=meta_dim,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
    )
    model.summary()

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_mae", save_best_only=True, mode="min", verbose=1
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=5, restore_best_weights=True
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae", factor=0.5, patience=3, verbose=1
    )

    # Train
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
        verbose=1,
    )

    print(f"Training finished. Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
