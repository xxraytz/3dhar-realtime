from keras.models import Model, load_model
import logging

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class Recognizer:
    def __init__(self, path_to, ):
        self.__nn_model = self.__set_model(path_to)

    def __set_model(self, path_to):
        logger.info('Trying set the keras model...')
        ...

    def recognize(self):
        ...

