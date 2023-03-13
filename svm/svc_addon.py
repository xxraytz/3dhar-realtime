
import numpy as np
from sklearn import svm
from sklearn.utils.validation import check_array


LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']


class Svc(svm.SVC):
    """SVC with model loading opportunity"""

    def __init__(self, C=10000, gamma=0.0001):
        super().__init__(C=C, gamma=gamma, cache_size=1500)
        self.class_weight_ = np.array([1., 1.])
        self.classes_ = np.array([1, 2])
        self._sparse = False
        self.shape_fit_ = (44514, 417)
        self.n_features_in_ = 417
        self.fit_status_ = 0
        self.__model_from_file = {}
        self.human_data = []

    def load_data(self, file):
        with open(file, 'r') as f:
            line = f.readline()
            data = []
            coefs = []
            while line != "":
                coef, string_vector = line.split(" ", 1)
                vector = string_vector.split(" ")
                data += [{(vec.split(":"))[0]: float((vec.split(":"))[1]) for vec in vector}]
                coefs += [int(coef)]
                line = f.readline()
            self.human_data = [data, coefs]

    def load_model(self, file):
        with open(file, "r") as f:
            svm_type = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["svm_type"] = svm_type
            self.__model_from_file["kernel_type"] = (f.readline().split(" ")[1]).rstrip('\n')
            if self.__model_from_file["kernel_type"] == 'rbf':
                self.__model_from_file["gamma"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["nr_class"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["total_sv"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["rho"] = (f.readline().split(" ")[1]).rstrip('\n')

            self.__model_from_file["label"] = (f.readline().split(" ")[1]).rstrip('\n')
            temp_str = f.readline().split(" ")
            self.__model_from_file["nr_sv"] = [int(temp_str[1]), int(temp_str[2].rstrip("\n"))]
            self.__model_from_file["SV"] = len(self.__model_from_file) + 1
            a = f.readline().rstrip("\n")
            if a == "SV":
                for i in range(int(self.__model_from_file.get("total_sv"))):
                    temp_str = f.readline().rstrip('\n')
                    if temp_str[-1].__eq__(' '):
                        temp_str = temp_str[:-1]
                    self.__model_from_file[str(i)] = temp_str
        self.rec_model()

    def rec_model(self):
        sv_coefs = []
        sv = []

        for x0 in range(0, int(self.__model_from_file.get("total_sv"))):
            arr_str = str(self.__model_from_file.get(str(x0))).split(" ", 1)
            sv_coefs += [float(arr_str[0])]
            vector = arr_str[1].split(" ")
            dct = {(vec.split(":"))[0]: float((vec.split(":"))[1]) for vec in vector}
            sv += [list(dct.values())]

        self.support_vectors_ = check_array(sv)
        # self._dual_coef_ = check_array([sv_coefs])
        self.dual_coef_ = np.asarray([sv_coefs])
        self._dual_coef_ = self.dual_coef_
        self._dual_coef_ = -self._dual_coef_
        self.support_ = np.asarray(list(range(self.support_vectors_.shape[0])), dtype=np.int32)
        self.kernel = self.__model_from_file.get("kernel_type")
        if self.kernel == 'rbf':
            self._gamma = float(self.__model_from_file.get("gamma"))
        self._impl = self.__model_from_file.get("svm_type")
        self.svm_type = LIBSVM_IMPL.index(self._impl)

        n_class = self.support_vectors_.shape[0]
        if self.svm_type == 0 or self.svm_type == 1:
            self._n_support = np.array([self.__model_from_file.get("nr_sv")[1],
                                        self.__model_from_file.get("nr_sv")[0]], dtype=np.int32)

        if self.probability != 0:
            self._probA = np.empty(int(n_class * (n_class - 1) / 2), dtype=np.float64)
            self._probB = np.empty(int(n_class * (n_class - 1) / 2), dtype=np.float64)
        else:
            self._probA = np.empty(0, dtype=np.float64)
            self._probB = np.empty(0, dtype=np.float64)
        self.intercept_ = np.array([float(self.__model_from_file.get("rho"))])
        self._intercept_ = self.intercept_.copy()
        self.intercept_ *= -1
