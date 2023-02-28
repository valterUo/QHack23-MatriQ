# MatriQ project

Valter Uotila\
team: Qumpula Quantum\
PhD student\
University of Helsinki\
[valteruo.github.io](valteruo.github.io)

## Shortly
MatriQ project explores faster matrix multiplication algorithms with quantum computing.

## Bases on paper
> Fawzi, A. et al. [Discovering faster matrix multiplication algorithms with reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4).
*Nature* **610** (Oct 2022). [https://github.com/deepmind/alphatensor](https://github.com/deepmind/alphatensor).

## Key idea
Optimal matrix multiplication algorithms are sequences of suitable tensor configurations. Quantum computing can deal with tensors. Let's combine these two!

## Key contribution
The current MatriQ algorithm is able to discover 6/7 steps of [Strassen's algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm) with a good initial guess.

## Resources
  - Video presentation of the project: https://youtu.be/ux2twxZ2T4c
  - [A detailed scientific document to how computationally explore matrix multiplication algorithms](https://github.com/valterUo/QHack23-MatriQ/blob/main/Project_MatriQ.pdf)
  - Implementation:
    - QUBO formulation for D-wave Leap or simulated annealing (small cases solvable locally)
        - Studies on different sized matrices are located in jupyter notebooks 2x2, 3x3, 4x4
        - utils.py implements useful functions
        - qubos_optimized.py implements functions to construct QUBOs
        - drafts folder contains the very first implementations
    - QAOA formulation with Pennylane so that executable on Amazon Braket (and on Nvidia GPUs in the future)
