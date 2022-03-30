import numpy as np
from numpy import nan
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time
from tsid_manipulator_class import TsidManipulator
import config_tsid as conf
import warnings

start_time = time.time()
warnings.filterwarnings("ignore")

#Active learning parameters:
N = 100		    #size of initial labeled set
N_iter = 10		#number of active learning iteration
B = 10          #batch size

#Controller parameters:
q0 = np.zeros(6)	#initial configuration
v0 = np.zeros(6)	#initial velocity

tsid = TsidManipulator(conf, q0, v0, viewer=0)	# initialization of the tsid simulation

v_max = tsid.v_max[1]	
v_min = tsid.v_min[1]
q_max = tsid.q_max[1]
q_min = tsid.q_min[1]

#Generate low-discrepancy unlabeled Xu_iter samples:
sampler = qmc.Halton(d=2, scramble=False)
sample = sampler.random(n=10000)
l_bounds = [q_min, v_min]
u_bounds = [q_max, v_max]
data = qmc.scale(sample, l_bounds, u_bounds)

plt.figure()
plt.scatter(data[:,0], data[:,1], marker=".", alpha=0.5);

##Generate the initial set of labeled samples:
#X_iter = np.empty((N+N_iter*B,2))*nan
#y_iter = np.empty(N+N_iter*B)*nan
#Xu_iter = data

#Generate the initial sets of labeled and unlabeled samples
X_iter = [[0,0]]          
y_iter = [tsid.compute_problem(q0,v0)]   
Xu_iter = data

for i in range(N):
    q0[1] = data[i,0]
    v0[1] = data[i,1]
        
    #X_iter[i] = [q0[1],v0[1]]
    #y_iter[i] = tsid.compute_problem(q0,v0)
    
    X_iter = np.append(X_iter,[[q0[1],v0[1]]], axis = 0)
    res = tsid.compute_problem(q0,v0)
    y_iter = np.append(y_iter, res)
    Xu_iter = np.delete(Xu_iter, i, axis = 0)
    
    #Add intermediate states of succesfull initial conditions
    if res == 1:
        q_traj = tsid.q_res[1,1:]
        v_traj = tsid.v_res[1,1:]
        for l in range(0, q_traj.shape[0], int(q_traj.shape[0]/5)):
            X_iter = np.append(X_iter,[[q_traj[l],v_traj[l]]], axis = 0)
            y_iter = np.append(y_iter, 1)

plt.figure()
plt.scatter(X_iter[:,0], X_iter[:,1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)

print("Initial training set generated")

#Create and train the initial svm classifier:
#clf = svm.SVC(kernel='rbf', probability=True, class_weight='balanced') 
#clf.fit(X_iter[:N,:], y_iter[:N])

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf'], 'probability': [True], 'class_weight': ['balanced']}
clf = GridSearchCV(svm.SVC(), param_grid, refit = True)
clf.fit(X_iter, y_iter)

print("Initial classifier trained")

plt.figure()
x_min, x_max = X_iter[:,0].min(), X_iter[:,0].max()
y_min, y_max = X_iter[:,1].min(), X_iter[:,1].max()
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8) 

#Update the labeled set with active learning and re-train the classifier:
for k in range(N_iter):
	#Compute the shannon entropy of the unlabeled samples:
	prob_xu = clf.predict_proba(Xu_iter)
	etp = np.empty(prob_xu.shape[0])*nan
	for p in range(prob_xu.shape[0]):
		etp[p] = entropy(prob_xu[p])
	
	#Add the B most uncertain samples to the labeled set:
	maxindex = np.argpartition(etp, -B)[-B:]     #indexes of the uncertain samples

	for x in range(B):    
		q0[1] = Xu_iter[maxindex[x],0]
		v0[1] = Xu_iter[maxindex[x],1]

		#X_iter[N+B*(k+1)-(B-x),:] = [q0[1],v0[1]]
		#y_iter[N+B*(k+1)-(B-x)] = tsid.compute_problem(q0, v0)
		
		X_iter = np.append(X_iter,[[q0[1],v0[1]]], axis = 0)
		res = tsid.compute_problem(q0,v0)
		y_iter = np.append(y_iter,res)
		
		#Add intermediate states of succesfull initial conditions
		if res == 1:
			q_traj = tsid.q_res[1,1:]
			v_traj = tsid.v_res[1,1:]
			for l in range(0, q_traj.shape[0], int(q_traj.shape[0]/5)):
				X_iter = np.append(X_iter,[[q_traj[l],v_traj[l]]], axis = 0)
				y_iter = np.append(y_iter, 1)

	Xu_iter = np.delete(Xu_iter, maxindex, axis = 0)
	
	#Re-fit the model with the new selected Xu_iter:
	#clf.fit(X_iter[:N+B*(k+1),:], y_iter[:N+B*(k+1)])
	
	clf.fit(X_iter, y_iter)
	
	print("Classifier", k+1, "trained")

plt.figure()
plt.scatter(X_iter[:,0], X_iter[:,1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)

plt.figure()
x_min, x_max = X_iter[:,0].min(), X_iter[:,0].max()
y_min, y_max = X_iter[:,1].min(), X_iter[:,1].max()
h = .02 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.show()

print("Execution time: %s seconds" % (time.time() - start_time))
