from math import exp, log, sqrt, pi
from scipy.stats import norm

# only for implied volatility computation

def d_j(j, S, K, r, v, T):
    return (log(S / K) + (r + ((-1)**(j-1)) * 0.5 * v * v) * T) / (v * (T**0.5))


def call_price(CP, S, K, r, v, T):
    price =     CP * S * norm.cdf(CP * d_j(1, S, K, r, v, T)) \
            -   CP * K * norm.cdf(CP * d_j(2, S, K, r, v, T)) * exp(-r * T)
    return price


def call_vega(CP, S, K, r, v, T):
    d = d_j(1, S, K, r, v, T)
    vega = CP * S * exp(-d ** 2 / 2) * sqrt(T / (2 * pi))
    return vega


def reverse_BS_dic(CP, S, K, r, T, C_target, epsilon, a, b):
    x = (b + a) / 2
    C = call_price(CP, S, K, r, x, T)

    while abs(C - C_target) > epsilon:
        C = call_price(CP, S, K, r, x, T)
        if C > C_target :
            b = x
        else:
            a = x
        x = (b + a) / 2
    return x


def reverse_BS_new(CP, S, K, r, T, v_0, C_target, epsilon):
    x = v_0
    C = call_price(CP, S, K, r, x, T)
    fail = 0

    while abs(C - C_target) > epsilon:
        C = call_price(CP, S, K, r, x, T)
        V = call_vega(CP, S, K, r, x, T)
        if V == 0:
            fail = 1
            print('Newton method fails for {}'.format(S))
            break
        x -= (C - C_target) / V
    if fail == 1:
        a = 0.001
        b = 1.
        x = reverse_BS_dic(CP, S, K, r, T, C_target, epsilon, a, b)
    return x
