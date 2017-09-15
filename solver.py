from scipy.sparse.linalg import inv
from datetime import datetime
from scipy.sparse import csc_matrix
import numpy as np
from math import exp


def F(n, omega, A, b, r_f, delta_t):
    return A * omega + b * exp(r_f * delta_t * n)


def F_0(n, omega, A_0, b_0, r_f, delta_t):
    return A_0 * omega + b_0 * exp(r_f * delta_t * n)


def F_1(n, omega, A_1, b_1, r_f, delta_t):
    return A_1 * omega + b_1 * exp(r_f * delta_t * n)


def F_2(n, omega, A_2, b_2, r_f, delta_t):
    return A_2 * omega + b_2 * exp(r_f * delta_t * n)


def CN_scheme(m, N, U_0, delta_t, A, b, r_f):
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    lhs = csc_matrix(I - 0.5 * delta_t * A)
    inv_lhs = inv(lhs)
    for n in range(1, N + 1):
        U = inv_lhs * (U + 0.5 * delta_t * (F(n - 1, U, A, b, r_f, delta_t) + b * exp(r_f * delta_t * n)))
    end = datetime.now()
    time = (end - start).total_seconds()
    return [U, time]


def DO_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f):
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    lhs_1 = csc_matrix(I - theta * delta_t * A_1)
    inv_lhs_1 = inv(lhs_1)
    lhs_2 = csc_matrix(I - theta * delta_t * A_2)
    inv_lhs_2 = inv(lhs_2)
    for n in range(1, N + 1):
        Y_0 = U + delta_t * F(n - 1, U, A, b, r_f, delta_t)
        rhs_1 = Y_0 + theta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1 = inv_lhs_1 * rhs_1
        rhs_2 = Y_1 + theta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))  #we update b_2
        U = inv_lhs_2 * rhs_2
    end = datetime.now()
    time = (end - start).total_seconds()
    return [U, time]


def CS_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f):
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    lhs_1 = csc_matrix(I - theta * delta_t * A_1)
    inv_lhs_1 = inv(lhs_1)
    lhs_2 = csc_matrix(I - theta * delta_t * A_2)
    inv_lhs_2 = inv(lhs_2)
    for n in range(0, N):
        Y_0 = U + delta_t * F(n - 1, U, A, b, r_f, delta_t)
        rhs_1 = Y_0 + theta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1 = inv_lhs_1 * rhs_1
        rhs_2 = Y_1 + theta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))  #we update b_2
        Y_2 = inv_lhs_2 * rhs_2

        Y_0_tilde = Y_0 + 0.5 * delta_t * (F_0(n, Y_2, A_0, b_0, r_f, delta_t) - F_0(n - 1, U, A_0, b_0, r_f, delta_t))

        rhs_1 = Y_0_tilde + theta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1_tilde = inv_lhs_1 * rhs_1
        rhs_2 = Y_1_tilde + theta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))  #we update b_2
        U = inv_lhs_2 * rhs_2
    end = datetime.now()
    time = (end - start).total_seconds()
    return [U, time]


def MCS_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f):
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    lhs_1 = csc_matrix(I - theta * delta_t * A_1)
    inv_lhs_1 = inv(lhs_1)
    lhs_2 = csc_matrix(I - theta * delta_t * A_2)
    inv_lhs_2 = inv(lhs_2)
    for n in range(0, N):
        Y_0 = U + delta_t * F(n - 1, U, A, b, r_f, delta_t)
        rhs_1 = Y_0 + theta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1 = inv_lhs_1 * rhs_1
        rhs_2 = Y_1 + theta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))  #we update b_2
        Y_2 = inv_lhs_2 * rhs_2

        Y_0_hat = Y_0 + theta * delta_t * (F_0(n, Y_2, A_0, b_0, r_f, delta_t) - F_0(n - 1, U, A_0, b_0, r_f, delta_t))
        Y_0_tilde = Y_0_hat + (0.5 - theta) * delta_t * (F(n, Y_2, A, b, r_f, delta_t) - F(n - 1, U, A, b, r_f, delta_t))

        rhs_1 = Y_0_tilde + theta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1_tilde = inv_lhs_1 * rhs_1
        rhs_2 = Y_1_tilde + theta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))  #we update b_2
        U = inv_lhs_2 * rhs_2
    end = datetime.now()
    time = (end - start).total_seconds()
    return [U, time]


def HV_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f):
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    lhs_1 = csc_matrix(I - theta * delta_t * A_1)
    inv_lhs_1 = inv(lhs_1)
    lhs_2 = csc_matrix(I - theta * delta_t * A_2)
    inv_lhs_2 = inv(lhs_2)
    for n in range(0, N):
        Y_0 = U + delta_t * F(n - 1, U, A, b, r_f, delta_t)
        rhs_1 = Y_0 + theta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1 = inv_lhs_1 * rhs_1
        rhs_2 = Y_1 + theta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))  #we update b_2
        Y_2 = inv_lhs_2 * rhs_2

        Y_0_tilde = Y_0 + 0.5 * delta_t * (F(n, Y_2, A, b, r_f, delta_t) - F(n - 1, U, A, b, r_f, delta_t))

        rhs_1 = Y_0_tilde + theta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n, Y_2, A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1_tilde = inv_lhs_1 * rhs_1
        rhs_2 = Y_1_tilde + theta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n, Y_2, A_2, b_2, r_f, delta_t))  #we update b_2
        U = inv_lhs_2 * rhs_2
    end = datetime.now()
    time = (end - start).total_seconds()
    return [U, time]
