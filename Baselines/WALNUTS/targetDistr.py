import numpy as np
import scipy.stats as sps


# -------------------------------------
# example target distributions
# -------------------------------------

# standard gaussian distribution

def stdGauss(q, hessian=False):
    lp = -0.5 * np.sum(q * q)
    grad = -q
    return [lp, grad]


# bivariate zero-mean gaussian with unit marginal variances and correlation=rho

def corrGauss(q, hessian=False):
    rho = 0.5
    tmp = 1.0 - rho ** 2
    lp = -0.5 * q[0] ** 2 - (0.5 / tmp) * (q[1] - rho * q[0]) ** 2
    grad = np.array([-(q[0] - rho * q[1]) / tmp,
                     -(q[1] - rho * q[0]) / tmp])
    return [lp, grad]


# distribution q[0] \sim N(0,1), q[1]|q[0] \sim N(q[0]^2,1)
def smileDistr(q, hessian=False):
    lp = -0.5 * q[0] ** 2 - 0.5 * (q[1] - q[0] ** 2) ** 2
    grad = np.array([-q[0] + 2.0 * q[0] * q[1] - 2.0 * q[0] ** 3,
                     q[0] ** 2 - q[1]])
    return [lp, grad]


def modFunnel(q, hessian=False):
    x = q[0]
    y = q[1]
    t1 = np.exp(-3.0 * x)
    t2 = 1.0 + t1
    t3 = 0.1e1 / t2
    t4 = y ** 2
    t5 = -0.1e1 / 0.2e1
    lp = t5 * (t2 * t4 + np.log(t3) + x ** 2)
    grad = np.array([0.3e1 / 0.2e1 * t1 * (t4 - t3) - x, -y * t2])
    return [lp, grad]


def modFunnelH(q, hessian=False):
    if (not hessian): return modFunnel(q)
    q1 = q[0]
    q2 = q[1]
    t1 = np.exp(-3 * q1)
    t2 = 1 + t1
    t3 = q2 ** 2
    t4 = -2 * q1
    t5 = np.exp(-6 * q1)
    t6 = 0.1e1 / t2
    t7 = 0.1e1 / 0.2e1
    t8 = 0.1e1 / t2 ** 2
    cg0 = np.array(
        [-t7 * (t2 * t3 - np.log(t2) + q1 ** 2), t7 * (t1 * (-3 + 3 * t3 + t4) + t4 + 3 * t5 * t3) * t6, -q2 * t2,
         t7 * (t1 * (-9 * t3 + 5) + t3 * (-18 * t5 - 9 * np.exp(-9 * q1)) - 2 - 2 * t5) * t8, 3 * t1 * q2, -t2])
    lp = cg0[0]
    grad = cg0[1:3]
    H = np.array([[cg0[3], cg0[4]], [cg0[4], cg0[5]]])
    return [lp, grad, H]


def funneln(q, hessian=False):
    n_others = len(q) - 1
    grad_constant = -n_others / 2.0
    lp = sps.norm.logpdf(q[0], loc=0.0, scale=3.0) + sum(sps.norm.logpdf(q[1:], loc=0.0, scale=np.exp(0.5 * q[0])))
    grad = np.r_[
        np.array([grad_constant - q[0] / 9.0 + 0.5 * np.exp(-q[0]) * sum(q[1:] * q[1:])]),
        -q[1:] * np.exp(-q[0])
    ]
    return [lp, grad]


def funnelnrescaled(q, hessian=False):
    n = len(q)
    S = np.ones(n)
    S[0] = 3.0
    qb = S * q
    lp, g = funneln(qb)
    return [lp, S * g]


def funnel1(q, hessian=False):
    lp = sps.norm.logpdf(q[0], loc=0.0, scale=3.0) + sps.norm.logpdf(q[1], loc=0.0, scale=np.exp(0.5 * q[0]))
    grad = np.array([-0.5 - q[0] / 9.0 + 0.5 * q[1] ** 2 * np.exp(-q[0]),
                     -q[1] * np.exp(-q[0])])
    return [lp, grad]