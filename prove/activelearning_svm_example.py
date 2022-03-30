import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import time

start_time = time.time()


#Load dataset
cancer = datasets.load_breast_cancer()

print(cancer.data.shape)

# print the names of the 13 features
#print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
#plaprint("Labels: ", cancer.target_names)

X = cancer.data[:,:2]
y = cancer.target

B_init = 100 #initial batch 

X_init = X[:B_init,:]

y_init = y[:B_init]

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', probability=True) 

#Train the model using the training sets
clf.fit(X_init, y_init)

#PLOTS
plt.figure()
plt.clf()

plt.scatter(
    X_init[:, 0], X_init[:,1], c=y_init, zorder=10, cmap=plt.cm.Paired, edgecolor="k", s=20
)

# create a mesh to plot in
x_min, x_max = X_init[:, 0].min() - 1, X_init[:, 0].max() + 1
y_min, y_max = X_init[:, 1].min() - 1, X_init[:, 1].max() + 1
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

Sup = clf.support_

X_iter = X[Sup[0]]
y_iter = y[Sup[0]]

for k in range(Sup.shape[0]):
    X_iter = np.append(X_iter,X[k,:])
    y_iter = np.append(y_iter,y[k])
    
prob = clf.predict_proba(X[B_init:,:2])
etp = [entropy(prob[i]) for i in range(prob.shape[0])]

B = 10 # batch size
maxindex = np.argpartition(etp, -B)[-B:]

for x in range(B):
    X_iter = np.append(X_iter,X[B_init+maxindex[x],:])
    y_iter = np.append(y_iter,y[B_init+maxindex[x]])
    
X_iter = np.reshape(X_iter, (-1,2))
    
clf.fit(X_iter, y_iter)

#PLOTS
plt.figure()
plt.clf()

plt.scatter(
    X_iter[:, 0], X_iter[:,1], c=y_iter, zorder=10, cmap=plt.cm.Paired, edgecolor="k", s=20
)

# create a mesh to plot in
x_min, x_max = X_iter[:, 0].min() - 1, X_iter[:, 0].max() + 1
y_min, y_max = X_iter[:, 1].min() - 1, X_iter[:, 1].max() + 1
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

print("Execution time: %s seconds" % (time.time() - start_time))

plt.show()
