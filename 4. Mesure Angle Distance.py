# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
T_e = 1
T = 100
sigma_Q = 1
sigma_angle = np.pi/180
sigma_dist = 10
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
R[0][0] = sigma_angle**2
R[1][1] = sigma_dist**2
print(R)

x_init = [3,40,-4,20]
x_kalm = x_init
P_kalm = np.eye(4)

"""# Nouvelle section"""

def creer_trajectoire(F,Q,x_init,T):

  x = np.zeros((4, T))
  x[:, 0] = x_init

  for k in range(1, T):
    uk = np.random.multivariate_normal(mean=np.zeros(4), cov=Q)
    x[:, k] = F @ x[:, k-1] + uk
  return x

vecteur_x = creer_trajectoire(F,Q,x_init,T)
# print(vecteur_x)

def creer_observations_radar(R,vecteur_x,T):
  y = np.zeros((2, T))

  for k in range(T):
      vk = np.random.multivariate_normal(mean=np.zeros(2), cov=R)
      y[0, k] = np.arctan(vecteur_x[2,k] / vecteur_x[0, k]) + vk[0]
      y[1, k] = (vecteur_x[0, k]**2 + vecteur_x[2, k]**2)**(1/2) + vk[1]

  return y

vecteur_y = creer_observations_radar(R,vecteur_x,T)
#print(vecteur_y)


def filtre_de_kalman_radar(F,Q,R,y_k,x_kalm_prec,P_kalm_prec):

  # Prédiction
  x_k1_k = F @ x_kalm_prec
  Hk = np.array([[-x_k1_k[2]/(x_k1_k[0]**2 + x_k1_k[2]**2), 0, 
                  x_k1_k[0]/(x_k1_k[0]**2 + x_k1_k[2]**2), 0],
                 [x_k1_k[0]/np.sqrt((x_k1_k[0]**2 + x_k1_k[2]**2)), 0, 
                 x_k1_k[2]/np.sqrt((x_k1_k[0]**2 + x_k1_k[2]**2)), 0]])
  P_k1_k = F @ P_kalm_prec @ F.T + Q

  # Innov
  L_k1 = Hk @ P_k1_k @ Hk.T + R

  # Mise à jour
  K_k1_k1 = P_k1_k @ Hk.T @ np.linalg.inv(L_k1)
  y_k_prime = y_k - np.array([np.arctan(x_k1_k[2]/x_k1_k[0]), 
                              np.sqrt(x_k1_k[0]**2 + x_k1_k[2]**2)]) + Hk @ x_k1_k
  x_kalm_k = x_k1_k + K_k1_k1 @ (y_k_prime - Hk @ x_k1_k)
  P_kalm_k = P_k1_k - K_k1_k1 @ L_k1 @ K_k1_k1.T

  return x_kalm_k, P_kalm_k

x_est = np.zeros((4,T))
x_kalm = x_init
P_kalm = np.eye(4)
for k in range(T):
    x_kalm, P_kalm = filtre_de_kalman_radar(F, Q, R, vecteur_y[:, k], x_kalm, P_kalm)
    x_est[:, k] = x_kalm
    #print(x_kalm)
    #print(k)

plt.figure(figsize=(10, 6))
plt.plot(vecteur_x[0, :], vecteur_x[2, :], label="Vraie trajectoire", linestyle='-', linewidth=2)
plt.plot(x_est[0, :], x_est[2, :], label="Trajectoire estimée", linestyle='--', linewidth=2)
#plt.scatter(vecteur_y[0, :], vecteur_y[1, :], color='red', s=15, label="Observations bruitées", alpha=0.6)
plt.xlabel("Position en x")
plt.ylabel("Position en y")
plt.title("Vraie trajectoire vs trajectoire estimée")
plt.legend()
plt.grid()
plt.show()

def err_quadra(k):
  return (vecteur_x[:,k] - x_est[:,k]).T @ (vecteur_x[:,k] - x_est[:,k])

erreur_moyenne = 0
for i in range(T):
  erreur_moyenne += err_quadra(i)**(1/2)
erreur_moyenne = erreur_moyenne/T
print(erreur_moyenne)

time = np.arange(T)  # Axe des temps

plt.figure(figsize=(10, 6))
plt.plot(time, vecteur_x[0, :], label="Vraie position \(p_x\)", linestyle='-', linewidth=2)
plt.plot(time, x_est[0, :], label="Position estimée \(p_x\)", linestyle='--', linewidth=2)
#plt.scatter(time, vecteur_y[0, :], color='red', s=10, label="Position observée \(p_x\)", alpha=0.6)
plt.xlabel("Temps")
plt.ylabel("Position \(p_x\)")
plt.title("Position \(p_x\) en fonction du temps")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, vecteur_x[2, :], label="Vraie position \(p_y\)", linestyle='-', linewidth=2)
plt.plot(time, x_est[2, :], label="Position estimée \(p_y\)", linestyle='--', linewidth=2)
#plt.scatter(time, vecteur_y[1, :], color='red', s=10, label="Position observée \(p_y\)", alpha=0.6)
plt.xlabel("Temps")
plt.ylabel("Position \(p_y\)")
plt.title("Position \(p_y\) en fonction du temps")
plt.legend()
plt.grid()
plt.show()

