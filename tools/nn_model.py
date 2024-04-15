import keras
import tensorflow as tf
import logging
from os import environ

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Recognizer:
    def __init__(self, ):
        self.__set_model(environ.get('nn_model_path'))

    def __set_model(self, path_to):
        logger.info('Trying set the keras model...')
        self.__nn_model: keras.Model = tf.keras.models.load_model(path_to)

        adam = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.__nn_model.compile(
            optimizer=adam,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        logger.info('The model is set up')

    def predict_actions(self, activity_map):
        return self.__nn_model.predict(activity_map, verbose=0) if activity_map.shape == (1, 32, 32) else None

