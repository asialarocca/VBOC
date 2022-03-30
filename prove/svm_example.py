import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import time

start_time = time.time()

#Load dataset
cancer = datasets.load_breast_cancer()

# print the names of the 13 features
#print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
#plaprint("Labels: ", cancer.target_names)

X, X_test, y, y_test = train_test_split(cancer.data[:,:2], cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', probability=True) 

#Train the model using the training sets
clf.fit(X, y)

#Tests
#print(clf.predict_proba(X_test[:,:2]))

#PLOTS
plt.figure()
plt.clf()

plt.scatter(
    X[:, 0], X[:,1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor="k", s=20
)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.show()

print("Execution time: %s seconds" % (time.time() - start_time))
