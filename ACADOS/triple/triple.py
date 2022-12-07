import numpy as np
from scipy.stats import entropy
import time
from triple_pendulum_ocp_class import OCPtriplependulumINIT
import random
import math
import warnings
import torch
import torch.nn as nn
from my_nn import NeuralNet
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPtriplependulumINIT()

    ocp_dim = ocp.nx

    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = -ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin
    
    num = 15

    # Hyper-parameters for nn:
    input_size = ocp_dim
    hidden_size = ocp_dim * 50
    output_size = 2
    learning_rate = 0.001

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load('model_save'))
    
#    model = torch.load('model_save').to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(weight=torch.torch.Tensor([10,1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer.load_state_dict('opt_save')

    # Active learning parameters:
    B = pow(10, ocp_dim)  # batch size
    etp_stop = 0.5  # active learning stopping condition
    loss_stop = 0.1  # nn training stopping condition
    beta = 0.8
    n_minibatch = 512
    n_minibatch_model = pow(2,15)
    it_max = int(B / n_minibatch)
    
    sigmoid = nn.Sigmoid()

    with open("mean.txt", "r") as f:
        val = float(f.readlines()[0])
    mean = torch.Tensor([val]).to(device)
    with open("std.txt", "r") as f:
        val = float(f.readlines()[0])
    std = torch.Tensor([val]).to(device)
    X_iter = np.load('X_iter.npy').tolist()
    y_iter = np.load('y_iter.npy').tolist()  
    Xu_iter = np.load('Xu_iter.npy').tolist()   
    
    k=0
    performance_history = []
    
    print('All uploaded')
        
    for iteration in range(100):
        
        X_iter = X_iter[-100*B:]
        y_iter = y_iter[-100*B:]

        rad_q = (q_max - q_min) / num
        rad_v = (v_max - v_min) / num
    
        n_points = 2
    
        etp = np.empty((len(X_iter),))
                    
        with torch.no_grad():
            X_iter_tensor = torch.Tensor(X_iter).to(device)
            X_iter_tensor = (X_iter_tensor - mean) / std
            my_dataloader = DataLoader(X_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
            for (idx, batch) in enumerate(my_dataloader):
                if n_minibatch_model*(idx+1) > len(X_iter):
                    prob_x = sigmoid(model(batch)).cpu()
                    etp[n_minibatch_model*idx:len(X_iter)] = entropy(prob_x, axis=1)
                else:
                    prob_x = sigmoid(model(batch)).cpu()
                    etp[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = entropy(prob_x, axis=1)
                    
#        del my_dataloader, X_iter_tensor,prob_x
        print('prob_x')
        
        xu = np.array(X_iter)[etp > 0.5]
        xu = xu[-100*B:]
#        del etp
        
        print('xu')
    
        Xu_it = np.empty((xu.shape[0], n_points, ocp_dim))
    
        # Generate other random samples:
        for i in range(xu.shape[0]):
            for n in range(n_points):
    
                # random angle
                alpha = math.pi * random.random()
                beta = math.pi * random.random()
                lamb = math.pi * random.random()
                theta = math.pi * random.random()
                gamma = 2 * math.pi * random.random()
    
                # random radius
                tmp = math.sqrt(random.random())
                r_x = rad_q * tmp
                r_y = rad_v * tmp
                # calculating coordinates
                x1 = r_x * math.cos(alpha) + xu[i, 0]
                x2 = r_x * math.sin(alpha) * math.cos(beta) + xu[i, 1]
                x3 = r_y * math.sin(alpha) * math.sin(beta) * math.cos(lamb) + xu[i, 2]
                x4 = r_y * math.sin(alpha) * math.sin(beta) * \
                    math.sin(lamb) * math.cos(theta) + xu[i, 3]
                x5 = r_y * math.sin(alpha) * math.sin(beta) * math.sin(lamb) * \
                    math.sin(theta) * math.cos(gamma) + xu[i, 4]
                x6 = r_y * math.sin(alpha) * math.sin(beta) * math.sin(lamb) * \
                    math.sin(theta) * math.sin(gamma) + xu[i, 5]
    
                Xu_it[i, n, :] = [x1, x2, x3, x4, x5, x6]
    
        Xu_it.shape = (xu.shape[0] * n_points, ocp_dim)
        Xu_iter = Xu_it.tolist()
        
#        del Xu_it, xu
        
        print('xu_iter')
    
        etpmax = 1
    
        while not (etpmax < etp_stop or len(Xu_iter) == 0):
    
            if len(Xu_iter) < B:
                B = len(Xu_iter)
    
            etp = np.empty((len(Xu_iter),))
    
            with torch.no_grad():
                Xu_iter_tensor = torch.Tensor(Xu_iter).to(device)
                Xu_iter_tensor = (Xu_iter_tensor - mean) / std
                my_dataloader = DataLoader(Xu_iter_tensor,batch_size=n_minibatch_model,shuffle=False)
                for (idx, batch) in enumerate(my_dataloader):
                    if n_minibatch_model*(idx+1) > len(Xu_iter):
                        prob_xu = sigmoid(model(batch)).cpu()
                        etp[n_minibatch_model*idx:len(Xu_iter)] = entropy(prob_xu, axis=1)
                    else:
                        prob_xu = sigmoid(model(batch)).cpu()
                        etp[n_minibatch_model*idx:n_minibatch_model*(idx+1)] = entropy(prob_xu, axis=1)
                        
            print('prob_xu')
            
#            del Xu_iter_tensor, my_dataloader, prob_xu
    
            maxindex = np.argpartition(etp, -B)[
                -B:
            ].tolist()  # indexes of the uncertain samples
            maxindex.sort(reverse=True)
    
            etpmax = max(etp[maxindex])  # max entropy used for the stopping condition
            
#            del etp
            performance_history.append(etpmax)
    
            k += 1
    
            # Add the B most uncertain samples to the labeled set:
            for x in range(B):
                x0 = Xu_iter[maxindex[x]]
                del Xu_iter[maxindex[x]]
                q0 = x0[:3]
                v0 = x0[3:]
                
                if x0[0] > ocp.thetamax or x0[0] < ocp.thetamin or x0[1] > ocp.thetamax or x0[1] < ocp.thetamin or x0[2] > ocp.thetamax or x0[2] < ocp.thetamin or abs(x0[0]) > ocp.dthetamax or abs(x0[1]) > ocp.dthetamax or abs(x0[2]) > ocp.dthetamax:
                    X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                    y_iter.append([1, 0])
                else:
                    # Data testing:
                    res = ocp.compute_problem(q0, v0)
                    if res == 1:
                        X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                        y_iter.append([0, 1])
                    elif res == 0:
                        X_iter.append([q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]])
                        y_iter.append([1, 0])
    
            print("CLASSIFIER", k, "IN TRAINING")
    
            it = 0
            val = 1
    
            # Train the model
            while val > loss_stop and it <= it_max:
    
                ind = random.sample(range(len(X_iter) - B), int(n_minibatch / 2))
                ind.extend(
                    random.sample(
                        range(len(X_iter) - B, len(X_iter)),
                        int(n_minibatch / 2),
                    )
                )
    
                X_iter_tensor = torch.Tensor([X_iter[i] for i in ind]).to(device)
                y_iter_tensor = torch.Tensor([y_iter[i] for i in ind]).to(device)
                X_iter_tensor = (X_iter_tensor - mean) / std
    
                # Zero the gradients
                for param in model.parameters():
                    param.grad = None
    
                # Forward pass
                outputs = model(X_iter_tensor).to(device)
                loss = criterion(outputs, y_iter_tensor)
    
                # Backward and optimize
                loss.backward()
                optimizer.step()
    
                val = beta * val + (1 - beta) * loss.item()
    
                it += 1
    
            print("CLASSIFIER", k, "TRAINED")
    
            print("etpmax:", etpmax)
            
            with open('etp.txt','w') as f:
                f.write(str(iteration)+ ' it'+ str(etpmax))
            
    
        torch.save(model.state_dict(), 'model_save_np')
    
        with open('times.txt','w') as f:
            f.write(str(time.time() - start_time))
            
        dimen = len(X_iter)-100*B
        
        if dimen > 0:
            np.save('X_iter_np' + str(iteration) + '.npy',np.asarray(X_iter[:dimen]))
            np.save('y_iter_np' + str(iteration) + '.npy',np.asarray(y_iter[:dimen]))
    
        np.save('X_iter_np.npy',np.asarray(X_iter))
    
        np.save('y_iter_np.npy',np.asarray(y_iter))
    
        np.save('Xu_iter_np.npy',np.asarray(Xu_iter))
            
        with open('mean.txt','w') as f:
            f.write(str(mean.item()))
    
        with open('std.txt','w') as f:
            f.write(str(std.item()))

    print("Execution time: %s seconds" % (time.time() - start_time))
