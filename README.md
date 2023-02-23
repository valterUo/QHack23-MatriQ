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
MatriQ algorithm is able to rediscover [Strassen's algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm).

## Resources
  - Video presentation of the project: Will appear by the end of the hackathon
  - [A longer and detailed scientific document to how computationally explore matrix multiplication algorithms](https://github.com/valterUo/QHack23-MatriQ/blob/main/Project_MatriQ.pdf) (working on)
  - Implementation (working on):
    - QUBO formulation for Amazon Braket and D-wave Leap
        - Studies on different sized matrices are located in jupyter notebooks 2x2, 3x3, 4x4
        - utils.py implements useful functions
        - qubos.py implements functions to construct QUBOs
    - QAOA formulation with Pennylane
        - TODO
    - Nvidia GPU-accelerated implementation
        - TODO
