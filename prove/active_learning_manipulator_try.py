import numpy as np
from numpy import nan
from scipy.stats import entropy, qmc
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from sklearn import svm
import time
from tsid_manipulator_class_try import TsidManipulator
import config_tsid as conf

start_time = time.time()

N = 100		    #size of initial labeled set
N_iter = 10		#number of active learning iteration
B = 10          #batch size

q0 = np.zeros(6)	# initial configuration
v0 = np.zeros(6)	# initial velocity

tsid = TsidManipulator(conf, q0, v0, viewer=False)	# initialization of the tsid simulation

#Define states bounds
v_max = tsid.model.velocityLimit[1]	
v_min = -v_max
q_max = tsid.model.upperPositionLimit[1]
q_min = tsid.model.lowerPositionLimit[1]

#Generate low-discrepancy unlabeled data samples
sampler = qmc.Halton(d=2, scramble=False)
sample = sampler.random(n=10000)
l_bounds = [q_min, v_min]
u_bounds = [q_max, v_max]
data = qmc.scale(sample, l_bounds, u_bounds)

plt.figure()
plt.scatter(data[:,0], data[:,1], marker=".", alpha=0.5);

#Generate the initial sets of labeled and unlabeled samples
X_iter = [[0,0]]          
y_iter = [tsid.compute_problem(q0,v0)]   
Xu_iter = np.delete(data, 0, axis = 0)

for i in range(N):
    q0[1] = data[i,0]
    v0[1] = data[i,1]
    
    #tsid = TsidManipulator(conf, q0, v0, viewer=False)
    
    X_iter = np.append(X_iter,[[q0[1],v0[1]]], axis = 0)
    res = tsid.compute_problem(q0,v0)
    y_iter = np.append(y_iter, res)
    Xu_iter = np.delete(Xu_iter, i, axis = 0)

	##Add intermediate states of succesfull initial conditions
    #if res == 1:
    #    q_traj = tsid.q_res[1,1:]
    #    v_traj = tsid.v_res[1,1:]
    #    for l in range(0,q_traj.shape[0],100):
    #        X_iter = np.append(X_iter,[[q_traj[l],v_traj[l]]], axis = 0)
    #        y_iter = np.append(y_iter, 1)

plt.figure()
plt.scatter(X_iter[:, 0], X_iter[:,1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)

#Create the svm classifier
clf = svm.SVC(kernel='rbf', probability=True, class_weight= 'balanced') 

#Train the model using the training sets
clf.fit(X_iter[:N,:], y_iter[:N])

plt.figure()
x_min, x_max = X_iter[:N,0].min() - 1, X_iter[:N,0].max() + 1
y_min, y_max = X_iter[:N,1].min() - 1, X_iter[:N,1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8) 

for k in range(N_iter):
    #Generate the new sets of labeled and unlabeled samples
    
    ##Initialize the set with the support vectors of the original model
	#sup = clf.support_	#support vectors
	#X0 = X_iter
	#y0 = y_iter
	#X_iter = [X0[sup[0],:]]
	#y_iter = y0[sup[0]] 
	
	#for k in range(sup.shape[0]):
	#	X_iter = np.append(X_iter,[X0[k,:]], axis = 0)
	#	y_iter = np.append(y_iter,y0[k])

	#Calculate the entropy of the unlabeled data set
	prob_xu = clf.predict_proba(Xu_iter)
	#prob_x = clf.predict_proba(X_iter[:N+B*k,:])
	
	etp = [entropy(prob_xu[p]) for p in range(prob_xu.shape[0])]
	
	#D = np.empty((prob_x.shape[0],prob_xu.shape[0]))*nan
	#div = np.empty(prob_xu.shape[0])*nan
	
	#for s in range(prob_xu.shape[0]):
	#	for n in range(prob_x.shape[0]):
	#		D[n,s] = -sum(rel_entr(prob_xu[s],prob_x[n]))
	#	div[s] = sum(D[:,s])

	#Add the B most uncertain samples to the set
	maxindex = np.argpartition(etp, -B)[-B:]     #indexes of the uncertain samples

	for x in range(B):    
		q0[1] = Xu_iter[maxindex[x],0]
		v0[1] = Xu_iter[maxindex[x],1]
		
		#tsid = TsidManipulator(conf, q0, v0, viewer=False)
		
		X_iter = np.append(X_iter,[[q0[1],v0[1]]], axis = 0)
		y_iter = np.append(y_iter,tsid.compute_problem(q0,v0))
		Xu_iter = np.delete(Xu_iter, maxindex[x], axis = 0)
	
	#Re-fit the model with the new selected data
	clf.fit(X_iter[:N+B*(k+1),:], y_iter[:N+B*(k+1)])
	
	

plt.figure()
plt.scatter(X_iter[:, 0], X_iter[:,1], c=y_iter, marker=".", alpha=0.5, cmap=plt.cm.Paired)

plt.figure()
x_min, x_max = X_iter[:,0].min() - 1, X_iter[:,0].max() + 1
y_min, y_max = X_iter[:,1].min() - 1, X_iter[:,1].max() + 1
h = .02 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.show()

print("Execution time: %s seconds" % (time.time() - start_time))
