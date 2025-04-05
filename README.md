# PINN-HeatEquation

This repository contains a Physics-Informed Neural Network (PINN) implementation for solving the one-dimensional Heat Equation, using measurement data. 

A typical experience in a physics teaching lab is the study of heat transfer in a copper rod. In this setup, a constant power is applied at one end using a function generator, and a series of thermocouples measure the temperature along the rod. 

With this data, the most common analysis is to estimate the material's diffusivity coefficient. The problem is that the solutions to the heat equation, considering the boundary conditions, are not trivial and always require excessive approximations.

For this reason, one solution using the acquired data is to use a PINN. This is a multilayer neural network trained using a loss function derived from the differential equation itself. For more information see this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125).

