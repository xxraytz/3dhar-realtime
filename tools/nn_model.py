from keras.models import Model, load_model

class Recognizer:
    def __init__(self, path_to, ):
        self.__nn_model = self.__set_model(path_to)

    def __set_model(self, path_to):
        ...

    def recognize(self):
        ...

