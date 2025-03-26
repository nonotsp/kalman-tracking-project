# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math 

T_e = 1
T = 100
sigma_Q = 1
sigma_px = 30
sigma_py = 30
F = np.eye(4)
F[0][1] = T_e
F[2][3] = T_e

#print(F)

Q = np.eye(4)
Q[0][0] = T_e*T_e/3
Q[0][1] = T_e/2
Q[1][0] = Q[0][1]
for i in range(0,2):
  for j in range(0,2):
    Q[i+2][j+2] = Q[i][j]
Q = sigma_Q**2 * T_e * Q
#print(Q)

H = np.zeros((2,4))
H[0][0] = 1
H[1][2] = 1
#print(H)

R = np.zeros((2,2))
R[0][0] = sigma_px**2
R[1][1] = sigma_py**2
print(R)

x_init = [3,40,-4,20]
x_kalm = x_init
P_kalm = np.eye(4)

vecteur_x_avion_ligne = np.load('vecteur_x_avion_ligne.npy').T
vecteur_x_avion_voltige = np.load('vecteur_x_avion_voltige.npy').T
#print(vecteur_x_avion_ligne)

vecteur_y_avion_ligne = np.load('vecteur_y_avion_ligne.npy').T
vecteur_y_avion_voltige = np.load('vecteur_y_avion_voltige.npy').T
#print(vecteur_y_avion_ligne)

plt.figure(figsize=(10, 6))
plt.plot(vecteur_x_avion_ligne[0, :], vecteur_x_avion_ligne[2, :], label="Vraie trajectoire", linestyle='-', linewidth=2)
plt.scatter(vecteur_y_avion_ligne[0, :], vecteur_y_avion_ligne[1, :], color='red', s=15, label="Observations bruitées", alpha=0.6)
plt.xlabel("Position en x")
plt.ylabel("Position en y")
plt.title("Avion de ligne : vraie trajectoire vs observations bruitées")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(vecteur_x_avion_voltige[0, :], vecteur_x_avion_voltige[2, :], label="Vraie trajectoire", linestyle='-', linewidth=2)
plt.scatter(vecteur_y_avion_voltige[0, :], vecteur_y_avion_voltige[1, :], color='red', s=15, label="Observations bruitées", alpha=0.6)
plt.xlabel("Position en x")
plt.ylabel("Position en y")
plt.title("Avion de voltige : vraie trajectoire vs observations bruitées")
plt.legend()
plt.grid()
plt.show()


def filtre_de_kalman(F,Q,H,R,y_k,x_kalm_prec,P_kalm_prec):
  # Prédiction
  x_k1_k = F @ x_kalm_prec
  P_k1_k = F @ P_kalm_prec @ F.T + Q

  # Innov
  L_k1 = H @ P_k1_k @ H.T + R

  # Mise à jour
  
  if math.isnan(y_k[0]) or math.isnan(y_k[1]):
      x_kalm_k = x_k1_k
      P_kalm_k = P_k1_k
  else:
      K_k1_k1 = P_k1_k @ H.T @ np.linalg.inv(L_k1)
      x_kalm_k = x_k1_k + K_k1_k1 @ (y_k - H @ x_k1_k)
      P_kalm_k = P_k1_k - K_k1_k1 @ L_k1 @ K_k1_k1.T
  return x_kalm_k, P_kalm_k



x_est_ligne = np.zeros((4,T))
x_kalm = x_init
P_kalm = np.eye(4)

for k in range(T):
    x_kalm, P_kalm = filtre_de_kalman(F, Q, H, R, vecteur_y_avion_ligne[:, k], x_kalm, P_kalm)
    x_est_ligne[:, k] = x_kalm

plt.figure(figsize=(10, 6))
plt.plot(vecteur_x_avion_ligne[0, :], vecteur_x_avion_ligne[2, :], label="Vraie trajectoire", linestyle='-', linewidth=2)
plt.plot(x_est_ligne[0, :], x_est_ligne[2, :], label="Trajectoire estimée", linestyle='--', linewidth=2)
plt.scatter(vecteur_y_avion_ligne[0, :], vecteur_y_avion_ligne[1, :], color='red', s=15, label="Observations bruitées", alpha=0.6)
plt.xlabel("Position en x")
plt.ylabel("Position en y")
plt.title("Avion de ligne : Vraie trajectoire vs trajectoire estimée et observations")
plt.legend()
plt.grid()
plt.show()


def err_quadra(k, vecteur_x, x_est):
  return (vecteur_x[:,k] - x_est[:,k]).T @ (vecteur_x[:,k] - x_est[:,k])


def erreur_moyenne(vecteur_x, x_est): 
    erreur_moyenne = 0
    for i in range(T):
      erreur_moyenne += np.sqrt(err_quadra(i, vecteur_x, x_est))
    erreur_moyenne = erreur_moyenne/T
    return erreur_moyenne

print(erreur_moyenne(vecteur_x_avion_ligne, x_est_ligne))

x_est_voltige = np.zeros((4,T))
x_kalm = x_init
P_kalm = np.eye(4)

for k in range(T):
    x_kalm, P_kalm = filtre_de_kalman(F, Q, H, R, vecteur_y_avion_voltige[:, k], x_kalm, P_kalm)
    x_est_voltige[:, k] = x_kalm

plt.figure(figsize=(10, 6))
plt.plot(vecteur_x_avion_voltige[0, :], vecteur_x_avion_voltige[2, :], label="Vraie trajectoire", linestyle='-', linewidth=2)
plt.plot(x_est_voltige[0, :], x_est_voltige[2, :], label="Trajectoire estimée", linestyle='--', linewidth=2)
plt.scatter(vecteur_y_avion_voltige[0, :], vecteur_y_avion_voltige[1, :], color='red', s=15, label="Observations bruitées", alpha=0.6)
plt.xlabel("Position en x")
plt.ylabel("Position en y")
plt.title("Avion de ligne : Vraie trajectoire vs trajectoire estimée et observations")
plt.legend()
plt.grid()
plt.show()

print(erreur_moyenne(vecteur_x_avion_voltige, x_est_voltige))
