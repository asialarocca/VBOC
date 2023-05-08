import numpy as np
import matplotlib.pyplot as plt
import time
from pendulum_class_fixedveldir import OCPpendulum
import warnings
import random
import torch
import torch.nn as nn
from my_nn import NeuralNetRegression

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    start_time = time.time()

    # Ocp initialization:
    ocp = OCPpendulum()
    
    # Position and velocity bounds:
    v_max = ocp.dthetamax
    v_min = - ocp.dthetamax
    q_max = ocp.thetamax
    q_min = ocp.thetamin

    # Pytorch params:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_layers = 2
    hidden_layers = 100
    output_layers = 1
    learning_rate = 1e-3

    # Model and optimizer:
    model_dir = NeuralNetRegression(input_layers, hidden_layers, output_layers).to(device)
    criterion_dir = nn.MSELoss()
    optimizer_dir = torch.optim.Adam(model_dir.parameters(), lr=learning_rate)

    # Unviable data generation param:
    eps = 1e-3

    # Initialize training dataset:
    X_save = np.empty((0,2))

    print('Start data generation')

    # Data generation (simplified version): 
    for v_sel in [v_min, v_max]:

        # Reset the time parameters:
        N = ocp.N # number of time steps
        ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
        ocp.ocp_solver.update_qp_solver_cond_N(N)
        
        dt = 1e-2 # time step duration

        # Constraints and cost parameters:
        if v_sel == v_min:
            q_init = q_max
            q_fin = q_min

            lb = np.array([q_min, v_min, 0.])
            ub = np.array([q_max, 0., 1e-2])

            cost_dir = 1.
        else:
            q_init = q_min
            q_fin = q_max

            lb = np.array([q_min, 0., 0.])
            ub = np.array([q_max, v_max, 1e-2])

            cost_dir = -1.

        # Initial guess:
        x_guess = np.empty((N+1, 3))
        u_guess = np.zeros((N, 1))
        q_guess = np.linspace(q_init, q_fin, N+1, endpoint=True)
        for i in range(N+1):
            x_guess[i] = [q_guess[i], v_sel, dt]

        norm_old = v_max

        while True:

            # Reset solver:
            ocp.ocp_solver.reset()

            # Constraints and guess:
            for i in range(N):
                ocp.ocp_solver.set(i, "x", np.array(x_guess[i]))
                ocp.ocp_solver.set(i, "u", np.array(u_guess[i]))
                ocp.ocp_solver.set(i, 'p', np.array([cost_dir, 1.]))
                ocp.ocp_solver.constraints_set(i, "lbx", lb)
                ocp.ocp_solver.constraints_set(i, "ubx", ub)

            ocp.ocp_solver.constraints_set(0, "lbx", np.array([q_init, v_min, 0.]))
            ocp.ocp_solver.constraints_set(0, "ubx", np.array([q_init, v_max, 1e-2]))
            ocp.ocp_solver.constraints_set(N, "lbx", np.array([q_fin, 0., 0.]))
            ocp.ocp_solver.constraints_set(N, "ubx", np.array([q_fin, 0., 1e-2]))  
            ocp.ocp_solver.set(N, "x", np.array(x_guess[N]))      
            ocp.ocp_solver.set(N, 'p', np.array([cost_dir, 1.]))

            # Solve the OCP:
            status = ocp.ocp_solver.solve()

            # Print statistics:
            # ocp.ocp_solver.print_statistics()

            if status == 0:

                # Extract initial optimal state:
                x0 = ocp.ocp_solver.get(0, "x")

                # New velocity norm:
                norm_new = np.linalg.norm(x0[1])

                # Compare the current norm with the previous one:
                if norm_new > norm_old + 1e-4: # solve a new OCP

                    norm_old = norm_new

                    # Update the guess with the current solution:
                    x_guess = np.empty((N+1,3))
                    u_guess = np.empty((N+1,1))
                    for i in range(N):
                        x_guess[i] = ocp.ocp_solver.get(i, "x")
                        u_guess[i] = ocp.ocp_solver.get(i, "u")
                    x_guess[N] = ocp.ocp_solver.get(N, "x")
                    u_guess[N] = np.array([0.])

                    # Increase the number of time steps:
                    N = N + 1
                    ocp.ocp_solver.set_new_time_steps(np.full((N,), 1.))
                    ocp.ocp_solver.update_qp_solver_cond_N(N)

                else:
                    # Extract the optimal trajectory:
                    x_sol = np.empty((N+1,3))
                    u_sol = np.empty((N,1))
                    for i in range(N):
                        x_sol[i] = ocp.ocp_solver.get(i, "x")
                        u_sol[i] = ocp.ocp_solver.get(i, "u")
                    x_sol[N] = ocp.ocp_solver.get(N, "x")

                    dt = x_sol[0][2]

                    break

            else:
                raise Exception("Sorry, the solver failed") 
            
        # Save the initial optimal state:
        x0 = x_sol[0][:2]
        X_save = np.append(X_save, [x0], axis = 0)

        # Generate the corresponding unviable state:
        x_out = np.copy(x0)
        x_out[1] = x_out[1] - eps * cost_dir

        # Check if the optimal state is at velocity limit:
        if x_out[1] > v_max or x_out[1] < v_min:
            x_at_limit = True 
        else:
            x_at_limit = False
            
        # Iterate through the trajectory to verify the location of the states with respect to V (simplified version):
        for f in range(1, N):

            # Check if the optimal state is at a velocity limit:
            if x_at_limit: # the previous state was on dX -> if also the current state is on dX then save the state,
                # alternatively solve a new OCP to verify if it is on dV or inside V

                v_out = x_sol[f][1] - eps * cost_dir
                if v_out > v_max or v_out < v_min:

                    # Save the optimal state:
                    X_save = np.append(X_save, [x_sol[f][:2]], axis = 0)

                else:

                    # Original velocity norm:
                    norm_old = np.linalg.norm(x_sol[f][1])

                    # Reset the time settings:
                    N_test = N - f
                    ocp.ocp_solver.set_new_time_steps(np.full((N_test,), 1.))
                    ocp.ocp_solver.update_qp_solver_cond_N(N_test)

                    # Reset solver:
                    ocp.ocp_solver.reset()

                    # Constraints and guess:
                    for i in range(N_test):
                        ocp.ocp_solver.set(i, "x", x_sol[f+i])
                        ocp.ocp_solver.set(i, "u", u_sol[f+i])
                        ocp.ocp_solver.set(i, 'p', np.array([cost_dir, 1.]))
                        ocp.ocp_solver.constraints_set(i, "lbx", lb)
                        ocp.ocp_solver.constraints_set(i, "ubx", ub)

                    ocp.ocp_solver.constraints_set(0, "lbx", np.array([x_sol[f][0], v_min, 0.]))
                    ocp.ocp_solver.constraints_set(0, "ubx", np.array([x_sol[f][0], v_max, 1e-2]))
                    ocp.ocp_solver.constraints_set(N_test, "lbx", np.array([q_fin, 0., 0.]))
                    ocp.ocp_solver.constraints_set(N_test, "ubx", np.array([q_fin, 0., 1e-2]))
                    ocp.ocp_solver.set(N_test, "x", np.array([q_fin, 0., 1e-2]))      
                    ocp.ocp_solver.set(N_test, 'p', np.array([cost_dir, 1.]))

                    # Solve the OCP:
                    status = ocp.ocp_solver.solve()

                    if status == 0:
                        # Extract the new optimal state:
                        x0_new = ocp.ocp_solver.get(0, "x")

                        # New velocity norm:
                        norm_new = np.linalg.norm(x0_new[1])

                        # Compare the current norm with the previous one:
                        if norm_new > norm_old + 1e-4:

                            # Update the solution:
                            for i in range(N-f):
                                x_sol[i+f] = ocp.ocp_solver.get(i, "x")
                                u_sol[i+f] = ocp.ocp_solver.get(i, "u")

                            # Generate the corresponding unviable state:
                            x_out = np.copy(x0_new[:2])
                            x_out[1] = x_out[1] - eps * cost_dir

                            # Check if the optimal state is at velocity limit:
                            if x_out[1] > v_max or x_out[1] < v_min:
                                x_at_limit = True 
                            else:
                                x_at_limit = False

                        else:
                            x_at_limit = False

                        # Save the initial optimal state:
                        X_save = np.append(X_save, [x_sol[f][:2]], axis = 0)

                    else:
                        raise Exception("Sorry, the solver failed") 

            else: # the previous state was on dV -> save the current state
                
                if abs(x_sol[f][0] - q_fin) <= 1e-3:
                    break

                # Save valid state:
                X_save = np.append(X_save, [x_sol[f][:2]], axis = 0)

    print('Data generation completed')

    # Joint positions mean and variance:
    mean_dir, std_dir = torch.mean(torch.tensor(X_save[:,:1].tolist())).to(device).item(), torch.std(torch.tensor(X_save[:,:1].tolist())).to(device).item()

    # Rewrite data in the form [normalized position, velocity direction, velocity norm]:
    X_save_dir = np.empty((X_save.shape[0],3))
    for i in range(X_save_dir.shape[0]):
        X_save_dir[i][0] = (X_save[i][0] - mean_dir) / std_dir
        vel_norm = abs(X_save[i][1])
        if vel_norm != 0:
            X_save_dir[i][1] = X_save[i][1] / vel_norm
        X_save_dir[i][2] = vel_norm

    # Save data:
    np.save('data_1dof_vboc_10.npy', np.asarray(X_save))

    # Training parameters:
    beta = 0.8 # moving average param
    n_minibatch = 64 # batch size
    B = int(X_save.shape[0]*100/n_minibatch) # number of iterations for 100 epoch
    it_max = B * 100 # max number of iterations
            
    print('Start model training')

    it = 0
    val = max(X_save_dir[:,2])
    training_evol = []

    # Train the model
    while val > 1e-3 and it < it_max:

        # Extract batch of data:
        ind = random.sample(range(len(X_save_dir)), n_minibatch)
        X_iter_tensor = torch.Tensor([X_save_dir[i][:2] for i in ind]).to(device)
        y_iter_tensor = torch.Tensor([[X_save_dir[i][2]] for i in ind]).to(device)

        # Forward pass:
        outputs = model_dir(X_iter_tensor)
        loss = criterion_dir(outputs, y_iter_tensor)

        # Backward and optimize:
        loss.backward()
        optimizer_dir.step()
        optimizer_dir.zero_grad()

        # Moving average of training errors:
        val = beta * val + (1 - beta) * loss.item()

        it += 1

        if it % B == 0: 
            print(val)
            training_evol.append(val)

    print('Model training completed')

    print("Execution time: %s seconds" % (time.time() - start_time))

    # Show the training evolution:
    plt.figure()
    plt.plot(training_evol)
    plt.ylabel('Training error moving average')
    plt.xlabel('Epochs')
    plt.title("Training evolution")

    # Save the model:
    torch.save(model_dir.state_dict(), 'model_1dof_vboc_10')

    with torch.no_grad():
        # Show the resulting RMSE on the training data:
        X_iter_tensor = torch.Tensor(X_save_dir[:,:2]).to(device)
        y_iter_tensor = torch.Tensor(X_save_dir[:,2:]).to(device)
        outputs = model_dir(X_iter_tensor)
        print('RMSE training data: ', torch.sqrt(criterion_dir(outputs, y_iter_tensor))) 
            
        # Show the data and the resulting set approximation:
        plt.figure()
        plt.plot(
            X_save[:,0], X_save[:,1], "ko", markersize=2
        )
        h = 0.01
        xx, yy = np.meshgrid(np.arange(q_min, q_max, h), np.arange(v_min, v_max, h))
        inp = np.c_[xx.ravel(), yy.ravel(), yy.ravel()]
        for i in range(inp.shape[0]):
            inp[i][0] = (inp[i][0] - mean_dir) / std_dir
            vel_norm = abs(inp[i][1])
            if vel_norm != 0:
                inp[i][1] = inp[i][1] / vel_norm
            inp[i][2] = vel_norm
        out = (model_dir(torch.from_numpy(inp[:,:2].astype(np.float32)).to(device))).cpu().numpy() 
        y_pred = np.empty(out.shape)
        for i in range(len(out)):
            if inp[i][2] > out[i]:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        Z = y_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.xlim([q_min, q_max])
        plt.ylim([v_min, v_max])
        plt.ylabel('$\dot{q}$')
        plt.xlabel('$q$')
        plt.subplots_adjust(left=0.14)
        plt.grid()
        plt.title("Classifier")

    plt.show()
