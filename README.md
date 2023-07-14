# Learning the Viability Kernel of Robotic Manipulators

This GitHub repository contains data-driven algorithms for the computation of viability kernels of robotic manipulators. 
The algorithms use Optimal Control Problems (OCPs) to generate training data and employ Neural Networks (NNs) to approximate the viability kernel based on this data.

## Algorithms

### Active Learning (AL)

The Active Learning algorithm, referred to as "AL" solves OCPs for system's initial states to verify from which of these states it is possible to stop (reach the zero-velocity set). 
AL leverages then Active Learning techniques to iteratively select batches of new states to be tested to maximize the resulting NN classifier accuracy.

### Viability-Boundary Optimal Control (VBOC)

The Viability-Boundary Optimal Control algorithm, referred to as "VBOC," utilizes OCPs to directly compute states that lie exactly on the boundary of the viability set and uses an NN regressor to approximate the set.

### Hamilton-Jacoby Reachability (HJR)

The Hamilton-Jacoby Reachability algorithm, referred to as "HJB," is an adaptation of a reachability algorithm presented in the paper "Recursive Regression with Neural Networks: Approximating the HJI PDE Solution" by V. Rubies-Royo and C. Tomlin. 
HJR computes the solution of the Hamilton-Jacoby-Isaacs (HJI) Partial Differential Equation (PDE) through recursive regression. NN classifiers are employed to approximate the set.

## Additional Information

For more details, please refer to the paper "VBOC: Learning the Viability Boundary of a Robot Manipulator using Optimal Control" by A. La Rocca, M. Saveriano, and A. Del Prete. The paper provides analysis and comparison of the algorithms mentioned above. 

## References

- V. Rubies-Royo and C. Tomlin, "Recursive Regression with Neural Networks: Approximating the HJI PDE Solution," in 5th International Conference on Learning Representations, Apr 2017. Available at: http://arxiv.org/abs/1611.02739.
- A. La Rocca, M. Saveriano, A. Del Prete, "VBOC: Learning the Viability Boundary of a Robot Manipulator using Optimal Control," arXiv:2305.07535 [cs.RO], 2023. Available at: http://arxiv.org/abs/2305.07535.

## Usage

The main folder contains scripts for the generation of test data and for the comparison of the performance of the algorithms. The subfolders "AL", "VBOC" and "HJR" contain the implementation of the different algorithms for the computation of the Viability Kernels of 1, 2 and 3 DOFs systems.

To try the algorithms and compare their performance you have to first generate the test data (execute "selectedsystem_testdata.py"), then compute the set with the three algorithms (execute "AL/selectedsystem_al.py", "HJR/selectedsystem_hjr.py" and "VBOV/selectedsystem_vboc_iterative.py") and then compare the results (execute "selectedsystem_comparison.py").