import numpy as np
from sklearn import svm
from sklearn.utils.validation import check_array


LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']


class OneClass(svm.OneClassSVM):
    """SVC with model loading opportunity"""

    def __init__(self, gamma=0.0001, nu=0.03):
        super().__init__(gamma=gamma, nu=nu)
        self.__model_from_file = {}
        self.class_weight_ = np.array([])
        self.classes_ = np.array([])
        self._sparse = False
        self.shape_fit_ = (0, 417)
        self.n_features_in_ = 417
        self.fit_status_ = 0
        self.human_data = []

    def load_data(self, file):
        with open(file, 'r') as f:
            line = f.readline()
            data = []
            coefficient = []
            while line != "":
                coefficient, string_vector = line.split(" ", 1)
                vector = string_vector.split(" ")
                data += [{(vec.split(":"))[0]: float((vec.split(":"))[1]) for vec in vector}]
                coefficient += [int(coefficient)]
                line = f.readline()
            self.human_data = [data, coefficient]

    def load_model(self, file):
        with open(file) as f:
            self.__model_from_file["svm_type"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["kernel_type"] = (f.readline().split(" ")[1]).rstrip('\n')
            if self.__model_from_file["kernel_type"] == 'rbf':
                self.__model_from_file["gamma"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["nr_class"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["total_sv"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["rho"] = (f.readline().split(" ")[1]).rstrip('\n')
            self.__model_from_file["SV"] = len(self.__model_from_file) + 1
            if f.readline().rstrip("\n") == "SV":
                for i in range(int(self.__model_from_file.get("total_sv"))):
                    temp_str = f.readline().rstrip('\n')
                    if temp_str[-1] == " ":
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
        self._dual_coef_ = np.asarray([sv_coefs])
        self.dual_coef_ = self._dual_coef_
        self.support_ = np.asarray(list(range(self.support_vectors_.shape[0])), dtype=np.int32)
        self.kernel = self.__model_from_file.get("kernel_type")
        if self.kernel == 'rbf':
            self._gamma = float(self.__model_from_file.get("gamma"))
        self._impl = self.__model_from_file.get("svm_type")
        self.svm_type = LIBSVM_IMPL.index(self._impl)

        n_class = self.support_vectors_.shape[0]
        self._n_support = np.array([self.__model_from_file.get('total_sv'), self.__model_from_file.get('total_sv')], dtype=np.int32)
        if self.probability != 0:
            self._probA = np.empty(int(n_class * (n_class - 1) / 2), dtype=np.float64)
            self._probB = np.empty(int(n_class * (n_class - 1) / 2), dtype=np.float64)
        else:
            self._probA = np.empty(0, dtype=np.float64)
            self._probB = np.empty(0, dtype=np.float64)
        self._intercept_ = np.array([-1*float(self.__model_from_file.get("rho"))])
        self.intercept_ = self._intercept_
        self.offset_ = -1 * self._intercept_
