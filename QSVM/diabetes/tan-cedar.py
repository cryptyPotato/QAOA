import sys
sys.path.append('/home/g/gaur/gaur/.local/lib/python3.8/site-packages')

import numpy as np
import torch 
from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.datasets import load_iris,load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold


from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers,IQPEmbedding
from pennylane.operation import Tensor

import matplotlib.pyplot as plt

np.random.seed(42)

X, y = load_diabetes(return_X_y=True, as_frame=False)

n = 442

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
#X = X[:500]
#y = y[:500]
n_features = int(sys.argv[1])


X = SelectKBest(f_classif, k=n_features).fit_transform(X, y)
#print(X.shape)
#print(X[1])

# y is the blood glucose level one year after the base line test
# change y to binary y >= 185 diabetes, no otherwise

for i in range(len(y)):
    if (y[i] > 185):
        y[i] = 1
    else:
        y[i] = 0
        
#print(y)


# scaling the inputs is important since the embedding we use is periodic
#scaler = StandardScaler().fit(X)
#X_scaled = scaler.transform(X)

# scaling the labels to -1, 1 is important for the SVM and the
# definition of a hinge loss
y_scaled = 2 * (y - 0.5)


#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)
X_train, X_test, y_train, y_test = train_test_split(X , y_scaled)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

#clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test = scaler.transform(X_test)

#X  = np.concatenate((X_train, X_test), axis=0)
X.shape

def circuit(feature_vector,feature_vector_2,length):
    qml.AngleEmbedding(features=feature_vector, wires=range(length),rotation='Z')
    qml.adjoint(qml.AngleEmbedding(features=feature_vector_2, wires=range(length),rotation='Z'))
    #qml.IQPEmbedding(features=feature_vector, wires=range(length),n_repeats=2)
    #qml.adjoint(qml.IQPEmbedding(features=feature_vector_2, wires=range(length),n_repeats=2))
    return qml.probs(wires=range(length))
n_qubits = len(X_train[0])
dev_kernel = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev_kernel, interface="autograd")
def kernel(x1, x2):
    """The quantum kernel."""
    u_1 = qml.matrix(circuit)(x1,x2,len(x1))
    u_2 = u_1.conjugate().transpose()
    projector = u_1+u_2
    return qml.expval(qml.Hermitian(projector,wires=range(n_qubits)))

def qsvm(Xtrain,ytrain):
    svm = SVC(kernel=new_kernel_par).fit(Xtrain, ytrain)
    return svm

def kernel_matrix_rectangular(A, B,i):
    #sigma=0.1
    gamma = 1
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    res = np.zeros((1,len(B)))
 #   for i in range(len(A)):
    for j in range(len(B)):
        # if ( i < j):
        # res[0,j] =  np.exp(- gamma*(2 - kernel(A[0], B[j]))**2) # np.exp((-2 + kernel(A[0], B[j])))
        #res[0,j] =  np.exp(- gamma*(2 - kernel(A[0], B[j]))**2 )
        res[0,j] = np.tan(np.pi/(4+gamma*(2-kernel(A[0],B[j]))**2))
         #   else:
             #   res[i,j] = res[j,i]
    return list(res[0])
def kernel_matrix_square(A, B, i):
    gamma = 1
    res = np.zeros((1,len(B)))
    for j in range(i+1,len(B)):
        # res[0,j] =  np.exp(- gamma*(2 - kernel(A[0], B[j]))**2) # np.exp((-2+kernel(A[0],B[j])))
        #res[0,j] =  np.exp(- gamma*(2 - kernel(A[0], B[j]))**2 )
        res[0,j] = np.tan(np.pi/(4+gamma*(2-kernel(A[0],B[j]))**2))
    return list(res[0])
import time
import joblib
def new_kernel_par(A,B):
    ans = np.zeros((len(A),len(B)))
    s = time.time()
    # kernel_matrix(X_train[0], X_train)
    if len(A)==len(B):
        ans = joblib.Parallel(n_jobs=4)(joblib.delayed(kernel_matrix_square)([A[i]],B,i) for i in range(len(A)))
        #print(ans)
        e = time.time()
        ans = np.array(ans)
        #print(ans.shape)
        ans=ans+ans.T+np.eye(len(A),len(B))
        #print(ans)
    else:
        ans = joblib.Parallel(n_jobs=4)(joblib.delayed(kernel_matrix_rectangular)([A[i]],B,i) for i in range(len(A)))
        #print(ans)
        e = time.time()
        ans = np.array(ans)
        #print(ans.shape)
        #ans=ans+ans.T+np.eye(len(A),len(B))
        #print(ans)
   # ti=e-s
    # print((e-s)/(40)* 900*10**6/2 /3600/24)
    # print(1)
    return ans   #, (e-s)/(400-20)* 900*10**6/2 /3600/24

# svc = qsvm(X_train,y_train)

#predictions = svc.predict(X_test)
#accuracy_score(predictions, y_test)

#param_grid = {'C': [0.1, 1, 10],
#              'gamma': [0.1, 1, 10]}
# Create an SVM classifier with a radial basis function (RBF) kernel
#svm = SVC(kernel='rbf')
# Perform grid search with cross-validation
#grid_search = GridSearchCV(svm, param_grid, cv=2)
#grid_search.fit(X_train,y_train)
#best_params = grid_search.best_params_
#print("Best Parameters:", best_params)
# Make predictions on the test data using the best model
#y_pred = grid_search.predict(X_test)
#y_pred = grid_search.predict(X)
#predictions = svm.predict(sxtes)
#accuracy = accuracy_score(y_test, y_pred)
#accuracy = accuracy_score(y, y_pred)
#print("Accuracy for svm:", accuracy)

# RBF kernel


# >>> cross_val_score(clf, X, y, cv=cv)

#X = X[:100]
#y = y[:100]

lasso = SVC(kernel='rbf')
clf = make_pipeline(preprocessing.StandardScaler(), lasso)
# cv = KFold(n_splits=10, random_state=42, shuffle=True)
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

cv_results1 = cross_validate(clf, X, y, cv=cv)
sorted(cv_results1.keys())
['fit_time', 'score_time', 'test_score']
print("acc for 10-fold RBF ",cv_results1['test_score'], np.mean(cv_results1['test_score']))

# angle embedding


# >>> cross_val_score(clf, X, y, cv=cv)

#X = X[:100]
#y = y[:100]

lasso = SVC(kernel=new_kernel_par)#newhttps://jupyter.scinet.utoronto.ca/user/gaur/notebooks/Siddartha/Pima_diabetes.ipynb#_kernel_par)
clf = make_pipeline(preprocessing.StandardScaler(), lasso)
# cv = KFold(n_splits=10, random_state=42, shuffle=True)
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
cv_results = cross_validate(clf, X, y, cv=cv)
sorted(cv_results.keys())
['fit_time', 'score_time', 'test_score']
print("acc for 10-fold Qkernel angle embedding is ",cv_results['test_score'],np.mean(cv_results['test_score']))

# # IQP Embedding
# X = X
# y = y
# lasso = SVC(kernel=new_kernel_par)
# cv = KFold(n_splits=10, random_state=42, shuffle=True)
# cv_results2 = cross_validate(lasso, X, y, cv=cv)
# sorted(cv_results2.keys())
# ['fit_time', 'score_time', 'test_score']
# print("acc for 10-fold Qkernel IQP embedding depth 1 is ",sum(cv_results2['test_score'])/10)

# # IQP Embedding with depth 2
# X = X
# y = y
# lasso = SVC(kernel=new_kernel_par)
# cv = KFold(n_splits=10, random_state=42, shuffle=True)
# cv_results3 = cross_validate(lasso, X, y, cv=cv)
# sorted(cv_results3.keys())
# ['fit_time', 'score_time', 'test_score']
# print("acc for 10-fold Qkernel IQP embedding depth 2 is ",sum(cv_results3['test_score'])/10)

#X.shape