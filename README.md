# Kalman Filtering – Target Tracking Project

This repository contains the implementation of a Kalman Filter and an Extended Kalman Filter (EKF) for tracking mobile targets from noisy sensor data. The project was developed during the *Statistical Filtering* course at Télécom SudParis.

The main objective is to estimate the state (position and velocity) of a target using a probabilistic state-space model and to evaluate the performance of different filtering strategies in both simulated and real-world settings.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Description](#technical-description)
- [Code Breakdown](#code-breakdown)
- [Data Files](#data-files)
- [Learning Objectives](#learning-objectives)

---

## Project Overview

The project is structured into three parts:

1. **Synthetic Scenario**  
   A mobile target is simulated with known dynamics. Observations are generated with Gaussian noise, and a standard Kalman Filter is used to estimate the trajectory.

2. **Real Data: Aircraft Tracking**  
   Noisy trajectories of a commercial and an aerobatic aircraft are tracked using a Kalman Filter. Missing observations (`NaN`) are handled by skipping update steps.

3. **Nonlinear Observations (EKF)**  
   When the sensor provides angle and distance (instead of Cartesian coordinates), an Extended Kalman Filter (EKF) is used. The observation function is linearized at each step.

---

## Technical Description

### State-Space Model

- Linear system with constant velocity assumption
- State vector: `[px, vx, py, vy]`
- Observation types:
  - Cartesian: position `[px, py]`
  - Polar: angle and distance

### Filtering Methods

- **Standard Kalman Filter**
  - Linear prediction and correction
  - Assumes Gaussian noise in both process and measurement

- **Extended Kalman Filter (EKF)**
  - Handles nonlinear observations via Jacobian-based linearization
  - Used for radar-like sensors (angle, distance)

### Performance Evaluation

- Quadratic error and average RMSE
- Visual comparison: true vs. estimated vs. observed trajectories
- Sensitivity to measurement and process noise

---

## Code Breakdown

### `3.1_Implementation_Kalman_Synthetic.py`

Implements the standard Kalman Filter on simulated data:

- Initializes model matrices (`F`, `H`, `Q`, `R`) and initial state
- Generates a random trajectory (`x`) and noisy observations (`y`)
- Applies Kalman filtering recursively
- Computes error metrics and generates plots

**Goal**: Validate the Kalman filter in a controlled simulation.

---

### `3.2_Application.py`

Applies the Kalman Filter to real aircraft data:

- Loads `.npy` files containing real trajectories and noisy observations
- Manages missing values (`NaN`) in detection
- Reconstructs filtered trajectories
- Compares true, observed, and estimated data

**Goal**: Evaluate the filter’s robustness with real and imperfect data.

---

### `4_Measure_Angle_Distance.py`

Implements an Extended Kalman Filter (EKF) using polar measurements:

- Simulates noisy angle and distance from true Cartesian positions
- Defines nonlinear measurement functions and computes Jacobians
- Applies EKF update using linearized model
- Visualizes tracking performance

**Goal**: Adapt filtering to nonlinear observation models like radar.

---

## Data Files

These `.npy` files contain true or observed positions for the aircraft tracking experiments:

| File                            | Description                              |
|---------------------------------|------------------------------------------|
| `vecteur_x_avion_ligne.npy`     | True X positions of commercial aircraft  |
| `vecteur_y_avion_ligne.npy`     | True Y positions of commercial aircraft  |
| `vecteur_x_avion_voltige.npy`   | True X positions of aerobatic aircraft   |
| `vecteur_y_avion_voltige.npy`   | True Y positions of aerobatic aircraft   |

They are loaded and used in `3.2_Application.py`.

---
