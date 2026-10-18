"""Inference adapter for the bundled Keras 2 SavedModel."""
import logging
import numpy as np
from core.config import asset_path

logger = logging.getLogger(__name__)


class Recognizer:
    def __init__(self):
        import tensorflow as tf
        path = asset_path("nn_model_path", "data/nn-model/data003")
        if not (path / "saved_model.pb").is_file():
            raise FileNotFoundError(f"TensorFlow SavedModel not found: {path}")
        self.model = tf.keras.models.load_model(str(path), compile=False)
        if self.model.input_shape != (None, 32, 32, 1) or self.model.output_shape != (None, 9):
            raise ValueError("Expected a 32x32x1 HAR model with nine output classes.")
        logger.info("Loaded HAR model from %s", path)

    def predict_actions(self, activity_map):
        values = np.asarray(activity_map, dtype=np.float32)
        if values.shape == (1, 32, 32):
            values = values[..., None]
        if values.shape != (1, 32, 32, 1) or not np.isfinite(values).all():
            raise ValueError("Expected a finite activity map with shape (1, 32, 32, 1).")
        return self.model(values, training=False).numpy()
