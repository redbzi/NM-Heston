import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import grid
import mat_factory as fac
import solver
import bs


# parameters
K = 100.
S_0 = K
S = 8 * K
V = 5.
V_0 = 0.04
T = 1.

r_d = 0.025                                 # domestic interest rate
r_f = 0.                                    # foreign interest rate
rho = -0.9
sigma = 0.3
kappa = 1.5
eta = 0.04

true_price = 8.8948693600540167             # from Monte Carlo simulation (ref: https://github.com/RedwanBouizi/MC-Heston)

# grid [0, S] x [0, V]
m1 = 50                                     # S
m2 = 25                                     # V
m = (m1 + 1) * (m2 + 1)                     # matrix A and vector U size
c = K / 5
d = V / 500

# line [0, T]
N = 20
delta_t = T / N

# model setup
[Vec_s, Delta_s, Vec_v, Delta_v, X, Y] = grid.make_grid(m1, S, S_0, K, c, m2, V, V_0, d)
[A_0, A_1, A_2, A] = fac.make_matrices(m1, m2, m, rho, sigma, r_d, r_f, kappa, eta, Vec_s, Vec_v, Delta_s, Delta_v)
[B_0, B_1, B_2, B] = fac.make_boundaries(m1, m2, m, r_d, r_f, N, Vec_s, delta_t)

# pricing
print("--True Price", true_price)

theta = 0.8

print("\n--CN Scheme")
UU_0 = np.array([[max(Vec_s[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
U_0 = UU_0.flatten()

[price, time] = solver.CN_scheme(m, N, U_0, delta_t, A, B, r_f)
price = np.reshape(price, (m2 + 1, m1 + 1))
index_s = Vec_s.index(S_0)
index_v = Vec_v.index(V_0)

print(" S_0: ", Vec_s[index_s])
print(" V_0: ", Vec_v[index_v])
print(" price: ", price[index_v, index_s])
print(" Error: ", abs(price[index_v, index_s] - true_price) / true_price)
print(" Computation Time: ", time)


print("\n--Do Scheme")
UU_0 = np.array([[max(Vec_s[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
U_0 = UU_0.flatten()

[price, time] = solver.DO_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, B, B_0, B_1, B_2, r_f)
price = np.reshape(price, (m2 + 1, m1 + 1))
index_s = Vec_s.index(S_0)
index_v = Vec_v.index(V_0)

print(" S_0: ", Vec_s[index_s])
print(" V_0: ", Vec_v[index_v])
print(" price: ", price[index_v, index_s])
print(" Error: ", abs(price[index_v, index_s] - true_price) / true_price)
print(" Computation Time: ", time)


print("\n--CS Scheme")
UU_0 = np.array([[max(Vec_s[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
U_0 = UU_0.flatten()

[price, time] = solver.CS_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, B, B_0, B_1, B_2, r_f)
price = np.reshape(price, (m2 + 1, m1 + 1))
index_s = Vec_s.index(S_0)
index_v = Vec_v.index(V_0)

print(" S_0: ", Vec_s[index_s])
print(" V_0: ", Vec_v[index_v])
print(" price: ", price[index_v, index_s])
print(" Error: ", abs(price[index_v, index_s] - true_price) / true_price)
print(" Computation Time: ", time)


print("\n--MCS Scheme")
UU_0 = np.array([[max(Vec_s[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
U_0 = UU_0.flatten()

[price, time] = solver.MCS_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, B, B_0, B_1, B_2, r_f)
price = np.reshape(price, (m2 + 1, m1 + 1))
index_s = Vec_s.index(S_0)
index_v = Vec_v.index(V_0)

print(" S_0: ", Vec_s[index_s])
print(" V_0: ", Vec_v[index_v])
print(" price: ", price[index_v, index_s])
print(" Error: ", abs(price[index_v, index_s] - true_price) / true_price)
print(" Computation Time: ", time)


print("\n--HV Scheme")
UU_0 = np.array([[max(Vec_s[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
U_0 = UU_0.flatten()

[price, time] = solver.HV_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, B, B_0, B_1, B_2, r_f)
price = np.reshape(price, (m2 + 1, m1 + 1))
index_s = Vec_s.index(S_0)
index_v = Vec_v.index(V_0)

print(" S_0: ", Vec_s[index_s])
print(" V_0: ", Vec_v[index_v])
print(" price: ", price[index_v, index_s])
print(" Error: ", abs(price[index_v, index_s] - true_price) / true_price)
print(" Computation Time: ", time)


# 3d plot
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('Underlying price')
ax.set_ylabel('Variance')
ax.set_zlabel('Call price')
ax.plot_surface(X, Y, price)
plt.show()

# map plot
fig1, ax = plt.subplots()
plt.xlabel('S'); plt.ylabel('V')
map = plt.pcolor(Vec_s, Vec_v, price)
plt.colorbar(map)
plt.show()


# implied volatility
epsilon = 0.01

num_s_inf = 5
num_s_sup = 6
num_v_inf = 1
num_v_sup = 10

IV = np.array(
     [
        [
            bs.reverse_BS_new(1, Vec_s[i], K, r_f, T, 0.5, price[i][j], epsilon)
            for i in range(index_s - num_s_inf, index_s + num_s_sup + 1)
        ]
        for j in range(index_v - num_v_inf, index_v + num_v_sup + 1)
     ]
)

Vec_s_IV = Vec_s[index_s - num_s_inf : index_s + num_s_sup + 1]
Vec_v_IV = Vec_v[index_v - num_v_inf : index_v + num_v_sup + 1]

X_IV, Y_IV = np.meshgrid(Vec_s_IV, Vec_v_IV)

fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.set_xlabel('Underlying price')
ax2.set_ylabel('Variance')
ax2.set_zlabel('Implied Volatility')
ax2.plot_surface(X_IV, Y_IV, IV)
plt.show()


# convergence regarding S-axis meshing

def error_S(m_2, N_, delta_t_, theta_, l_m1_, true_price_):
    l_err_DO    = []
    l_err_CS    = []
    l_err_MCS   = []
    l_err_HV    = []
    for m_1 in l_m1_:
        m_0 = (m_1 + 1) * (m_2 + 1)
        l_s, d_s, l_v, d_v, _, _ = grid.make_grid(m_1, S, S_0, K, c, m_2, V, V_0, d)
        a_0, a_1, a_2, a = fac.make_matrices(m_1, m_2, m_0, rho, sigma, r_d, r_f, kappa, eta, l_s, l_v, d_s, d_v)
        b_0, b_1, b_2, b = fac.make_boundaries(m_1, m_2, m_0, r_d, r_f, N_, l_s, delta_t_)

        uu_0 = np.array([[max(l_s[i] - K, 0) for i in range(m_1 + 1)] for _ in range(m_2 + 1)])
        u_0 = uu_0.flatten()

        idx_s = l_s.index(S_0)
        idx_v = l_v.index(V_0)

        price_DO = solver.DO_scheme(m_0, N_, u_0, delta_t_, theta_, a, a_0, a_1, a_2, b, b_0, b_1, b_2, r_f)[0]
        price_DO = np.reshape(price_DO, (m_2 + 1, m_1 + 1))

        price_CS = solver.CS_scheme(m_0, N_, u_0, delta_t_, theta_, a, a_0, a_1, a_2, b, b_0, b_1, b_2, r_f)[0]
        price_CS = np.reshape(price_CS, (m_2 + 1, m_1 + 1))

        price_MCS= solver.MCS_scheme(m_0, N_, u_0, delta_t_, theta_, a, a_0, a_1, a_2, b, b_0, b_1, b_2, r_f)[0]
        price_MCS = np.reshape(price_MCS, (m_2 + 1, m_1 + 1))

        price_HV = solver.HV_scheme(m_0, N_, u_0, delta_t_, theta_, a, a_0, a_1, a_2, b, b_0, b_1, b_2, r_f)[0]
        price_HV = np.reshape(price_HV, (m_2 + 1, m_1 + 1))

        l_err_DO.append(abs(price_DO[idx_v, idx_s] - true_price_) / true_price_)
        l_err_CS.append(abs(price_CS[idx_v, idx_s] - true_price_) / true_price_)
        l_err_MCS.append(abs(price_MCS[idx_v, idx_s] - true_price_) / true_price_)
        l_err_HV.append(abs(price_HV[idx_v, idx_s] - true_price_) / true_price_)

    return l_m1_, l_err_DO, l_err_CS, l_err_MCS, l_err_HV


l_m1 = [10 * i for i in range(3, 8)]
theta = 0.8

l_m1, V_err_DO, V_err_CS, V_err_MCS, V_err_HV = error_S(m2, N, delta_t, theta, l_m1, true_price)

plt.xlabel('m1')
plt.ylabel('Relative Error')
plt.plot(l_m1, V_err_DO, label='DO')
plt.plot(l_m1, V_err_CS, label='CS')
plt.plot(l_m1, V_err_MCS, label='MCS')
plt.plot(l_m1, V_err_HV, label='HV')
plt.legend(loc='upper right', shadow=True)
plt.show()
