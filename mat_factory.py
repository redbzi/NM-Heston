import numpy as np
from math import exp
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

from coeff import *


def make_matrices(m1, m2, m, rho, sigma, r_d, r_f, kappa, eta, Vec_s, Vec_v, Delta_s, Delta_v):
    A_0 = np.zeros((m, m))
    A_1 = np.zeros((m, m))
    A_2 = np.zeros((m, m))

    l_9a = [-2, -1, 0]
    l_9b = [-1, 0, 1]
    l_9c = [0, 1, 2]
    l_10 = [-1, 0, 1]
    l_11 = [[-1, 0, 1], [-1, 0, 1]]

    # Definition of A_0
    for j in range(1, m2):
        for i in range(1, m1):
            c = rho * sigma * Vec_s[i] * Vec_v[j]
            for k in l_11[0]:
                for l in l_11[1]:
                    A_0[i + j * (m1 + 1), (i + k) + (j + l) * (m1 + 1)] += c * beta_s(i - 1, k, Delta_s) * beta_v(j - 1, l, Delta_v)

    A_0 = csc_matrix(A_0)
    #plt.spy(A_0)
    #plt.show()

    # Definition of A_1
    for j in range(m2 + 1):
        for i in range(1, m1):
            a = 0.5 * Vec_s[i] ** 2 * Vec_v[j]
            b = (r_d - r_f) * Vec_s[i]
            for k in l_10:
                A_1[i + j * (m1 + 1), (i + k) + j * (m1 + 1)] += (a * delta_s(i - 1, k, Delta_s) + b * beta_s(i - 1, k, Delta_s))
            A_1[i + j * (m1 + 1), i + j * (m1 + 1)] += - 0.5 * r_d
        A_1[m1 + j * (m1 + 1), m1 + j * (m1 + 1)] += - 0.5 * r_d


    A_1 = csc_matrix(A_1)
    #plt.spy(A_1)
    #plt.show()

    #Definition of A_2
    for j in range(m2 - 1):
        for i in range(m1 + 1):
            temp = kappa * (eta - Vec_v[j])
            temp2 = 0.5 * sigma ** 2 * Vec_v[j]
            if Vec_v[j] > 1.:
                for k in l_9a:
                    A_2[i + (j + 1) * (m1 + 1), i + (m1 + 1) * (j + 1 + k)] += temp * alpha_v(j, k, Delta_v)
                for k in l_10:
                    A_2[i + (j + 1) * (m1 + 1), i + (m1 + 1) * (j + 1 + k)] += temp2 * delta_v(j - 1, k, Delta_v)
            if j == 0:
                for k in l_9c:
                    A_2[i, i + (m1 + 1) * k] += temp * gamma_v(j, k, Delta_v)
            else:
                for k in l_10:
                    A_2[i + j * (m1 + 1), i + (m1 + 1) * (j + k)] += (temp * beta_v(j - 1, k, Delta_v) + temp2 * delta_v(j - 1, k, Delta_v))
            A_2[i + j * (m1 + 1), i + j * (m1 + 1)] += - 0.5 * r_d

    A_2 = csc_matrix(A_2)
    #plt.spy(A_2)
    #plt.show()

    A = A_0 + A_1 + A_2
    A = csc_matrix(A)
    #plt.spy(A)
    #plt.show()

    return [A_0, A_1, A_2, A]


def make_boundaries(m1, m2, m, r_d, r_f, N, Vec_s, delta_t):
    b_0 = [0.] * m
    b_1 = [0.] * m
    b_2 = [0.] * m

    # Boundary when s = S
    for j in range(m2 + 1):
        b_1[m1 * (j + 1)] = (r_d - r_f) * Vec_s[-1] * exp(-r_f * delta_t * (N - 1))

    # Boundary when v = V
    for i in range(1, m1 + 1):
        b_2[m - m1 - 1 + i] = -0.5 * r_d * Vec_s[i] * exp(-r_f * delta_t * (N - 1))

    b_0 = np.array(b_0)
    b_1 = np.array(b_1)
    b_2 = np.array(b_2)

    b = b_0 + b_1 + b_2

    return [b_0, b_1, b_2, b]
