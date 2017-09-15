from math import sinh, asinh
import numpy as np
import matplotlib.pyplot as plt


def Map_s(xi, K, c):
    return K + c * sinh(xi)


def Map_v(xi, d):
    return d * sinh(xi)


def make_grid(m1, S, S_0, K, c, m2, V, V_0, d):
    Delta_xi = (1.0 / m1) * (asinh((S - K) / c) - asinh(-K / c))
    Uniform_s = [asinh(-K / c) + i * Delta_xi for i in range(m1 + 1)]
    Vec_s = [Map_s(Uniform_s[i], K, c) for i in range(m1 + 1)]
    Vec_s.append(S_0)
    Vec_s.sort()
    Vec_s.pop(-1)
    Delta_s = [Vec_s[i + 1] - Vec_s[i] for i in range(m1)]

    Delta_eta = (1.0 / m2) * asinh(V / d)
    Uniform_v = [i * Delta_eta for i in range(m2 + 1)]
    Vec_v = [Map_v(Uniform_v[i], d) for i in range(m2 + 1)]
    Vec_v.append(V_0)
    Vec_v.sort()
    Vec_v.pop(-1)
    Delta_v = [Vec_v[i + 1] - Vec_v[i] for i in range(m2)]

    X, Y = np.meshgrid(Vec_s, Vec_v)

    # # grid checking
    # plt.plot(X, Y, '.', color='blue')
    # plt.show()

    return [Vec_s, Delta_s, Vec_v, Delta_v, X, Y]
