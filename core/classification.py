from svm.svc_addon import Svc
from svm.one_class_addon import OneClass


class SvmWrapper:
    __path = ""
    __model_main = None
    __model_ourliers = None
    __averages = []
    __rmsd = []

    def __init__(self):
        self.__path = "data/"
        files = {"modelMain": self.__path + "model-main.txt",
                 "modelOutliers": self.__path + "model-outliers.txt",
                 "averages": self.__path + "average.csv",
                 "RMSD": self.__path + "dispersion.csv"}
        self.__averages = self.load_compensation(files.get("averages"))
        self.__rmsd = self.load_compensation(files.get("RMSD"))
        self.c_svc = Svc(C=10000, gamma=0.0001)
        self.c_svc.load_model(files.get("modelMain"))
        self.one_class_svm = OneClass(gamma=0.0001, nu=0.03)
        self.one_class_svm.load_model(files.get("modelOutliers"))

    def get_instance(self):
        pass

    def classify(self, svm_vector: list):
        """

        :param svm_vector:
            dictionary of svm nodes
        :return:
        """
        svm_vector = self.__compensate(svm_vector)
        outlier = self.__is_outlier(svm_vector)
        if outlier:
            return [bool(outlier), 0]

        predict = self.c_svc.predict([self.svm_vector_to_list(svm_vector)])
        return [bool(outlier), -1 if predict == 2 else 1]

    def __compensate(self, svm_vector):
        # Do with simple for cycle.
        # TODO: Check that dict have index ZERO
        for i in range(len(svm_vector)):
            svm_vector[i] = (svm_vector[i] - self.__averages[i]) / self.__rmsd[i]
        return svm_vector

    def __is_outlier(self, vector):
        predict = self.one_class_svm.predict([vector])
        return predict == -1

    @staticmethod
    def svm_vector_to_list(svm_vector: list):
        list_x = [v for v in svm_vector]
        return list_x

    @staticmethod
    def load_compensation(path):
        with open(path, "r") as file:
            data = file.readline().rstrip(",")
            res = [float(number) for number in data.split(",")]
        return res


