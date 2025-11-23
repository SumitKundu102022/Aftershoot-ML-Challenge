# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import cv2
# import os

# from sklearn.preprocessing import StandardScaler, LabelEncoder

# train_path = "Train"
# valid_path = "Validation"

# train_df = pd.read_csv(os.path.join(train_path, "sliders.csv"))
# valid_df = pd.read_csv(os.path.join(valid_path, "slider_inputs.csv"))

# IMG_SIZE = 224

# def load_image(id_value, folder):
#     path = os.path.join(folder, "Images", f"{id_value}.tiff")
#     img = cv2.read(path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img / 255.0
#     return img

# X_images = np.array([load_image(i, train_path) for i in train_df.id_global])
# y = train_df[["Temperature", "Tint"]].values

# X_valid_images = np.array([load_image(i, valid_path) for i in valid_df.id_global])

# meta_features = ['WB_original', 'Focal_length', 'ISO', 'Aperture']
# scaler = StandardScaler()
# X_meta = scaler.fit_transform(train_df[meta_features].fillna(0))
# X_valid_meta = scaler.transform(valid_df[meta_features].fillna(0))

# # CNN Model
# base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3), pooling='avg')
# image_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# x = base_model(image_input)
# x = tf.keras.layers.Dense(128, activation='relu')(x)

# # Metadata Model
# meta_input = tf.keras.Input(shape=(X_meta.shape[1],))
# m = tf.keras.layers.Dense(64, activation='relu')(meta_input)

# # Merge
# combined = tf.keras.layers.concatenate([x, m])
# output = tf.keras.layers.Dense(2)(combined)

# model = tf.keras.Model(inputs=[image_input, meta_input], outputs=output)
# model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# model.summary()

# model.fit(
#     [X_images, X_meta], y,
#     validation_split=0.1,
#     epochs=10,
#     batch_size=16
# )

# preds = model.predict([X_valid_images, X_valid_meta])

# # Prepare submission
# submission = valid_df.copy()
# submission["Temperature"] = preds[:,0]
# submission["Tint"] = preds[:,1]

# submission.to_csv("submission.csv", index=False)
# print("Saved submission.csv")
# import os

# from src.config import VAL_DIR

# val_dir = "data/Validation"
# print("Validation folder exists:", os.path.exists(val_dir))
# print("CSV exists:", os.path.exists(os.path.join(VAL_DIR, "sliders_input.csv")))
# print("Files:", os.listdir(val_dir))

import pandas as pd
import numpy as np

df = pd.read_csv("results/submission.csv")

df["Temperature"] = np.round(df["Temperature"]).astype(int)
df["Tint"] = np.round(df["Tint"]).astype(int)

df.to_csv("results/submission.csv", index=False)

print(df.head())
print(df.dtypes)



