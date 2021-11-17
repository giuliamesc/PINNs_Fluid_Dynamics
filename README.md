# NAPDE_PINNs_FluidDynamics

# Report

## Introduction
### 1. Introduction
### 2. An Overview on Neural Networks
    1. Why the Neural Networks
    2. Perceptron
    3. Multilayer - FeedForward
    4. Loss Functions
    5. Backpropagation
### 3. Physics Informed Neural Networks
    1. What is a Pinn
    2. Why Pinns
    3. Structure of a Pinn
    4. Some Examples
## First Test Cases
### 1. Poisson Problem  
    1. Loss Dirichlet
    2. Loss Neumann
    3. Loss Pde
    4. Normalization
### 2. Poiseuille Flow
    1. Multidimensional Problem
    2. Stokes Problem
    3. Navier-Stokes Problem
### 3. Colliding Flows
    1. Fitting Points
    2. Press_Mean
    3. Noise on Data
## Navier Stokes - The Lid Drive Cavity
### 1. Steady Case
### 2. Unsteady Case
## An application - the Coronary Flow
## Conclusions
### 1. Conclusions
### 2. Refernces
### 3. Appendix
    1. Adam and BFGS
    2. Nisaba
    3. FeniCs

# Project Progress

## Last Improvements
- Reduced Loss Graph
- Now input can be given by a .txt
- New way of test savings
- Full code refactoring
- Refactor the code for Poiseuille and Colliding

## To do list:
- Generate plots with many epochs (until plateau of losses)
- Implementation of PRESS_0

## Further ideas
- Prediction of the exact value of the BC plateau, noisy case
- Restyle of losses plot (as for June presentation)
- Take the same collocation points of initial time for the unsteady case
- Different type of sampling
