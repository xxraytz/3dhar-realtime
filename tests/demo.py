from libsvm.commonutil import evaluations
# from libsvm.svm import *
from sklearn import svm
from svm.svc_addon import Svc

#
# LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']
#
#
# def foo():
#     svm_parameter('-c 10000')
#     model_ = svm_load_model('model-c10000-gamma0.0001.txt')
#     # _y, _x = svm_read_problem('data11_noOutliers_normed.svm')
#     # svm_predict(_y, _x, model)
#     _y, _x = svm_read_problem('data11_noOutliers_normed.svm')
#     _anw = svm_predict(_y, _x, model_)
#     return evaluations(_y, _anw[0])
#
#
# def foo2():
#     clf2 = Svc()
#     clf2.load_data('data11_noOutliers_normed.svm')
#     clf2.load_model('model-c10000-gamma0.0001.txt')
#     _x = clf2.human_data[1]
#     _x = [list(xx.values()) for xx in _x]
#     anw2 = clf2.predict(_x)
#     real_y = clf2.human_data[0]
#     predict_y = list(anw2)
#     return evaluations(real_y, predict_y)
#
#
# # solve1 = foo()
# # solve2 = foo2()
# # Проверка на простых данных
# x = [[1, 1], [1, 3], [6, 6], [1, 6], [6, 3], [6, 1]]
# y = [1, 1, 2, 1, 2, 2]
# # LIBSVM
#
# # isKernel=True must be set for precomputed kernel
# prob = svm_problem(y, x)
# param = svm_parameter('-t 0')
# m = svm_train(prob, param)
# x0 = [{1: 0, 2: 1}, {1: 0, 2: 2}, {1: 0, 2: 3}, {1: 0, 2: 4}, {1: 0, 2: 5},
#       {1: 3, 2: 1}, {1: 3, 2: 2}, {1: 3, 2: 3}, {1: 4, 2: 4}, {1: 4, 2: 5},
#       {1: 6, 2: 1}, {1: 6, 2: 2}, {1: 6, 2: 3}, {1: 6, 2: 4}, {1: 6, 2: 5}]
# y0 = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
# # anw = svm_predict(y0, x0, m)
# # x0, m_dim = gen_svm_nodearray({1: 6, 2: 4})
# a = svm_predict(y0, x0, m)
# print(a)
# # print(anw)
#
# x1 = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
#       [3, 1], [3, 2], [3, 3], [4, 4], [4, 5],
#       [6, 1], [6, 2], [6, 3], [6, 4], [6, 5]]
# clf = svm.SVC(kernel="linear")
# clf.fit(x, y)
# anw = clf.predict(x1)
# print(evaluations(y0, list(anw)))
#
# # ]новый пример
# x = [[1, 1], [1, 2], [2, 1.5], [2, 3], [2.5, 0.5],
#      [3, 3], [3.5, 2], [4, 1], [4, 3.5], [4.5, 0.5], [5, 2.5]]
# y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
# prob = svm_problem(y, x)
# model = svm_train(prob, param)
# x0 = [[1, 3.5], [2, 1], [2, 2], [2.5, 1], [2.5, 2.5], [3, 0.5],
#       [3, 2], [3.5, 1], [4, 1], [4, 2.5], [4.5, 3]]
# x0_lsvm = [{1: 1, 2: 3.5}, {1: 2, 2: 1}, {1: 2, 2: 2}, {1: 2.5, 2: 1}, {1: 2.5, 2: 2.5}, {1: 3, 2: 0.5},
#            {1: 3, 2: 2}, {1: 3.5, 2: 1}, {1: 4, 2: 1}, {1: 4, 2: 2.5}, {1: 4.5, 2: 3}]
#
# y0 = [1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2]
# if len(x0) == len(y0):
#     # LIBSVM
#     a = svm_predict(y, x0_lsvm, model)
#     print(a)
#
#     # SKLEARN
#     clf.fit(x, y)
#     b = clf.predict(x0)
#     b = evaluations(y0, list(b))
#     print(b)
#
# # Эксперимент на Ирисах Фишера
#
# iris = datasets.load_iris()
# set_versi_data = iris.data[50:]
# y = iris.target[50:]
# y = list(y)
# print(len(set_versi_data))
# print(len(y))
# # iris_lsvm = [{1: vec[0], 2: vec[1], 3: vec[2], 4: vec[3]} for vec in set_versi_data]
# iris_lsvm = [{1: vec[0], 2: vec[1]} for vec in set_versi_data]
# test_data_x = []
# test_data_y = []
# # test_id = []
# # for _i in range(cross):
# #     test_id.append(random.randint(0, 49))
# #     test_id.append(random.randint(50, 99))
# # test_id = [99, 78, 67, 56, 80, 81, 92, 57, 73, 65, 12, 34, 45, 32, 44, 10, 23, 25, 49, 36, 140, 130, 120, 134, 113]
# # test_id = [99, 78, 67, 56, 80, 81, 92, 57, 73, 65, 12, 34, 45, 32, 44, 10, 23, 25, 49, 36]
# test_id = [99, 78, 67, 56, 80, 12, 34, 45, 32, 44]
# test_id.sort(reverse=True)
# for i in range(len(test_id)):
#     test_data_x.append(iris_lsvm.pop(test_id[i]))
#     test_data_y.append(y.pop(test_id[i]))
#
# print(len(iris_lsvm))
# prob = svm_problem(y, iris_lsvm)
# model = svm_train(prob, param)
# a = svm_predict(test_data_y, test_data_x, model)
# print(a)
# x = [list(vec.values()) for vec in iris_lsvm]
# clf.fit(x, y)
# b = clf.predict([list(vec.values()) for vec in test_data_x])
# b = evaluations(test_data_y, list(b))
# print(b)

# def libsvm_x(x: list):
#     dicts_x = []
#     for vec in x:
#         key = [index for index in range(1, 418)]
#
#         dicts_x.append(dict(zip(key, vec)))
#     return dicts_x


# New experiment. Create model with libsvm and with sklearn for oneclass and twoclass

# Two class
# To train a model we should have training set and params
# Training set

svc = Svc()
svc.load_data("no-outliers-normalized-1.svm")
data = svc.human_data
x = [list(d.values()) for d in data[0]]
y = data[1]
svc_standart = svm.SVC(C=10000, gamma=0.0001, cache_size=1500, )
model = svc_standart.fit(x, y)
print("SVC model was trained.")
anw = svc_standart.predict(x)
print(evaluations(y, list(anw)))

svm_svc_new = Svc(C=10000, gamma=0.0001)
svm_svc_new.load_model("model_c10000_g0.0001_2.txt")
anw2 = svm_svc_new.predict(x)
print(evaluations(y, list(anw2)))
print("The end")

# One class classifier
# st = OneClass(gamma=0.0001, nu=0.03)
# st.load_data("matrix-fixed-normalized.svm")
# data = st.human_data
# x = [list(d.values()) for d in data[0]]
# y = data[1]
# st.load_model("matrix-fixed-normalized.svm.model")
# anw = st.predict(x)
# print(evaluations(y, list(anw)))
#
# oneclass_standart = svm.OneClassSVM(gamma=0.0001, nu=0.03)
#
# # m = svm_load_model("model_c10000_g0.0001_2.txt")
# oneclass_standart.fit(x, y)
# anw = oneclass_standart.predict(x)
# print(evaluations(y, list(anw)))
#
# print("end")
