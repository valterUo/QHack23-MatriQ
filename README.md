# MatriQ project

Valter Uotila\
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
  - Video presentation of the project:
  - A longer and detailed scientific document with an introduction to how computationally harvest matrix multiplication algorithms
  - Implementation (working on):
    - QUBO formulation for Amazon Braket and D-wave Leap
    - QAOA formulation with Pennylane
    - Nvidia GPU-accelerated implementation
