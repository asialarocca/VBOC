# Viability Kernels of Robotic Manipulators

This GitHub repository contains data-driven algorithms for the computation of viability kernels of robotic manipulators. 
The algorithms utilize Optimal Control Problems (OCPs) to generate training data and employ Neural Networks (NNs) to approximate the viability kernel based on this data.

## Algorithms

### Active Learning (AL)

The Active Learning algorithm, referred to as "AL" solves OCPs for system's initial states to verify from which of these states it is possible to stop (reach the zero-velocity set). 
AL leverages then Active Learning techniques to iteratively select batches of new states to be tested to maximize the resulting NN classifier accuracy.

### Hamilton-Jacoby Reachability (HJR)

The Hamilton-Jacoby Reachability algorithm, referred to as "HJB," is an adaptation of a reachability algorithm presented in the paper "Recursive Regression with Neural Networks: Approximating the HJI PDE Solution" by V. Rubies-Royo and C. Tomlin. 
HJR computes the solution of the Hamilton-Jacoby-Isaacs (HJI) Partial Differential Equation (PDE) through recursive regression. NN classifiers are employed to approximate the resulting set.

### Viability-Boundary Optimal Control (VBOC)

The Viability-Boundary Optimal Control algorithm, referred to as "VBOC," utilizes OCPs to directly compute states that lie exactly on the boundary of the viability set and uses an NN regressor to approximate the set.

## Additional Information

For a more detailed description and comparison of these algorithms, please refer to the paper "VBOC: Learning the Viability Boundary of a Robot Manipulator using Optimal Control" by A. La Rocca, M. Saveriano, and A. Del Prete. This paper provides an accurate analysis and comparison of the algorithms mentioned above. The paper is available on arXiv: 2305.07535 [cs.RO] (2023)(http://arxiv.org/abs/2305.07535).

## References

- V. Rubies-Royo and C. Tomlin, "Recursive Regression with Neural Networks: Approximating the HJI PDE Solution," in 5th International Conference on Learning Representations, Apr 2017. [Online]. Available: [http://arxiv.org/abs/1611.02739](http://arxiv.org/abs/1611.02739).
- A. La Rocca, M. Saveriano, A. Del Prete, "VBOC: Learning the Viability Boundary of a Robot Manipulator using Optimal Control," arXiv:2305.07535 [cs.RO] (2023)(http://arxiv.org/abs/2305.07535).
