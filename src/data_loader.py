import os
import numpy as np
from PIL import Image
import tensorflow as tf


class ImageMetaSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence that loads images from disk + precomputed metadata arrays.
    """

    def __init__(
        self,
        ids,
        metadata_array,
        labels,
        images_dir,
        batch_size,
        img_height,
        img_width,
        shuffle=True,
    ):
        super().__init__()  # avoids the Keras warning
        self.ids = np.array(ids)
        self.metadata_array = np.array(metadata_array)
        self.labels = None if labels is None else np.array(labels)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.shuffle = shuffle

        self.indices = np.arange(len(self.ids))
        self.on_epoch_end()

        # cache for found paths so we don't keep searching extensions
        self._path_cache = {}

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _find_image_path(self, img_id):
        """Try multiple extensions to find the actual image file on disk."""
        if img_id in self._path_cache:
            return self._path_cache[img_id]

        exts = [".tiff", ".tif", ".jpg", ".jpeg", ".png"]
        for ext in exts:
            candidate = os.path.join(self.images_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                self._path_cache[img_id] = candidate
                return candidate

        # If still not found, raise a clear error with some debug info
        files = os.listdir(self.images_dir)
        sample_files = files[:10]
        raise FileNotFoundError(
            f"Image for id_global '{img_id}' not found in '{self.images_dir}'.\n"
            f"Tried extensions: {exts}\n"
            f"Example files in folder: {sample_files}"
        )

    def _load_image(self, img_id):
        path = self._find_image_path(img_id)
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_width, self.img_height))
        img = np.array(img, dtype="float32") / 255.0
        return img

    def __getitem__(self, idx):
        batch_indices = self.indices[
        idx * self.batch_size : (idx + 1) * self.batch_size
       ]

        batch_ids = self.ids[batch_indices]
        batch_meta = self.metadata_array[batch_indices]

        batch_imgs = np.stack([self._load_image(i) for i in batch_ids], axis=0)

        if self.labels is not None:
            batch_labels = self.labels[batch_indices]
            return (
            {
                "image_input": batch_imgs,
                "meta_input": batch_meta,
            },
            batch_labels,
        )
        else:
            return {
                "image_input": batch_imgs,
                "meta_input": batch_meta,
            }