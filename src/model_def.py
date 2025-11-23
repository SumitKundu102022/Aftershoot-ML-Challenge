import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(meta_input_dim, img_height=224, img_width=224, img_channels=3):
    # Image branch
    image_input = keras.Input(
        shape=(img_height, img_width, img_channels), name="image_input"
    )

    # Use EfficientNetB0 as a reasonable backbone
    # If you don't have internet to download weights, change weights=None
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        pooling="avg",
    )

    x = base_model.output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Metadata branch
    meta_input = keras.Input(shape=(meta_input_dim,), name="meta_input")
    m = layers.Dense(128, activation="relu")(meta_input)
    m = layers.Dropout(0.2)(m)

    # Combine
    combined = layers.concatenate([x, m])
    z = layers.Dense(128, activation="relu")(combined)
    z = layers.Dropout(0.2)(z)

    # Final 2 outputs: Temperature, Tint
    outputs = layers.Dense(2, activation="linear", name="output")(z)

    model = keras.Model(inputs=[image_input, meta_input], outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mae",
        metrics=["mae"],
    )

    return model
