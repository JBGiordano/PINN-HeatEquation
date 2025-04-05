# PINN-HeatEquation

This repository contains a Physics-Informed Neural Network (PINN) implementation for solving the one-dimensional Heat Equation, using measurement data. 

A typical experience in a physics teaching lab is the study of heat transfer in a copper rod. In this setup, a constant power is applied at one end using a function generator, and a series of thermocouples measure the temperature along the rod. 

With this data, the most common analysis is to estimate the material's diffusivity coefficient. The problem is that the solutions to the heat equation, considering the boundary conditions, are not trivial and always require excessive approximations.

For this reason, one solution using the acquired data is to use a PINN. This is a multilayer neural network trained using a loss function derived from the differential equation itself. For more information see this [paper](Raissi, M., P. Perdikaris, and G. E. Karniadakis. “Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.” Journal of Computational Physics 378 (February 1, 2019): 686–707.).




