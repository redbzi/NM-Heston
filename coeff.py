def delta_s(i, pos, Delta_s):
    if pos == -1:
        return 2 / (Delta_s[i] * (Delta_s[i] + Delta_s[i + 1]))
    elif pos == 0:
        return -2 / (Delta_s[i] * Delta_s[i + 1])
    elif pos == 1:
        return 2 / (Delta_s[i + 1] * (Delta_s[i] + Delta_s[i + 1]))
    else:
        raise ValueError("Wrong pos")


def delta_v(i, pos, Delta_v):
    if pos == -1:
        return 2 / (Delta_v[i] * (Delta_v[i] + Delta_v[i + 1]))
    elif pos == 0:
        return -2 / (Delta_v[i] * Delta_v[i + 1])
    elif pos == 1:
        return 2 / (Delta_v[i + 1] * (Delta_v[i] + Delta_v[i + 1]))
    else:
        raise ValueError("Wrong pos")


def alpha_s(i, pos, Delta_s):
    if pos == -2:
        return Delta_s[i] / (Delta_s[i - 1] * (Delta_s[i - 1] + Delta_s[i]))
    elif pos == -1:
        return (-Delta_s[i - 1] - Delta_s[i]) / (Delta_s[i - 1] * Delta_s[i])
    elif pos == 0:
        return (Delta_s[i - 1] + 2 * Delta_s[i]) / (Delta_s[i] * (Delta_s[i - 1] + Delta_s[i]))
    else:
        raise ValueError("Wrong pos")


def alpha_v(i, pos, Delta_v):
    if pos == -2:
        return Delta_v[i] / (Delta_v[i - 1] * (Delta_v[i - 1] + Delta_v[i]))
    elif pos == -1:
        return (-Delta_v[i - 1] - Delta_v[i]) / (Delta_v[i - 1] * Delta_v[i])
    elif pos == 0:
        return (Delta_v[i - 1] + 2 * Delta_v[i]) / (Delta_v[i] * (Delta_v[i - 1] + Delta_v[i]))
    else:
        raise ValueError("Wrong pos")


def beta_s(i, pos, Delta_s):
    if pos == -1:
        return -Delta_s[i + 1] / (Delta_s[i] * (Delta_s[i] + Delta_s[i + 1]))
    elif pos == 0:
        return (Delta_s[i + 1] - Delta_s[i]) / (Delta_s[i] * Delta_s[i + 1])
    elif pos == 1:
        return Delta_s[i] / (Delta_s[i + 1] * (Delta_s[i] + Delta_s[i + 1]))
    else:
        raise ValueError("Wrong pos")


def beta_v(i, pos, Delta_v):
    if pos == -1:
        return -Delta_v[i + 1] / (Delta_v[i] * (Delta_v[i] + Delta_v[i + 1]))
    elif pos == 0:
        return (Delta_v[i + 1] - Delta_v[i]) / (Delta_v[i] * Delta_v[i + 1])
    elif pos == 1:
        return Delta_v[i] / (Delta_v[i + 1] * (Delta_v[i] + Delta_v[i + 1]))
    else:
        raise ValueError("Wrong pos")


def gamma_s(i, pos, Delta_s):
    if pos == 0:
        return (-2 * Delta_s[i + 1] - Delta_s[i + 2]) / (Delta_s[i + 1] * (Delta_s[i + 1] + Delta_s[i + 2]))
    elif pos == 1:
        return (Delta_s[i + 1] + Delta_s[i + 2]) / (Delta_s[i + 1] * Delta_s[i + 2])
    elif pos == 2:
        return -Delta_s[i + 1] / (Delta_s[i + 2] * (Delta_s[i + 1] + Delta_s[i + 2]))
    else:
        raise ValueError("Wrong pos")


def gamma_v(i, pos, Delta_v):
    if pos == 0:
        return (-2 * Delta_v[i + 1] - Delta_v[i + 2]) / (Delta_v[i + 1] * (Delta_v[i + 1] + Delta_v[i + 2]))
    elif pos == 1:
        return (Delta_v[i + 1] + Delta_v[i + 2]) / (Delta_v[i + 1] * Delta_v[i + 2])
    elif pos == 2:
        return -Delta_v[i + 1] / (Delta_v[i + 2] * (Delta_v[i + 1] + Delta_v[i + 2]))
    else:
        raise ValueError("Wrong pos")
