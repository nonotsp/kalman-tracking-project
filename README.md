# Kalman Filtering – Target Tracking Project

This repository presents the implementation of a Kalman Filter and an Extended Kalman Filter (EKF) for mobile object tracking, developed during the *Statistical Filtering* course (MAT4501) at Télécom SudParis.

The objective is to estimate the state (position and velocity) of a moving target from noisy sensor measurements, using a probabilistic state-space model.

---

## Project Overview

The project is organized into three main parts:

1. **Synthetic Scenario**  
   Simulation of target trajectories using a known linear dynamic model. Observations are generated with Gaussian noise, and a Kalman Filter is applied to reconstruct the true trajectory.

2. **Real Data: Aircraft Tracking**  
   Application of the Kalman Filter on real-world data extracted from video recordings of an airshow. Includes handling missing observations (`NaN`) due to detection loss (e.g., occlusion by clouds).

3. **Nonlinear Observations: Angle and Distance (EKF)**  
   Estimation from polar measurements (angle and distance) using an Extended Kalman Filter. The observation model is linearized via first-order Taylor expansion, and the EKF is implemented to track the target under this nonlinear setup.

---

## Technical Description

- **State-Space Model**  
  - Linear dynamics with constant velocity assumption  
  - State vector: `[px, vx, py, vy]`  
  - Noisy observations of position or polar coordinates (angle, distance)

- **Kalman Filter Implementation**  
  - Recursive prediction and correction steps  
  - Noise modeling through covariance matrices `Q` (process noise) and `R` (observation noise)

- **Extended Kalman Filter (EKF)**  
  - Nonlinear observation model  
  - Linearization via Jacobian computation at each prediction step

- **Error Analysis**  
  - Mean squared error (MSE) between estimated and true trajectories  
  - Visual analysis of filter performance under various noise conditions

---
