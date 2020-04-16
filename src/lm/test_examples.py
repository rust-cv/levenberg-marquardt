import numpy as np
from scipy.optimize import leastsq

np.set_printoptions(precision=20)

def report(out):
    print("params = \n  {}".format(out[0]))
    print("objective_function = {}".format((out[2]['fvec'] ** 2).sum() * 0.5))
    print("terminate {}, {}".format(out[3], out[4]))

def linear_full_rank(n, m, factor=1.):
    def func(params):
        s = params.sum()
        temp = 2. * s / m + 1
        vec = np.zeros(m);
        vec[:] = -temp
        vec[:params.size] += params
        return vec

    def jac(params):
        jac = np.zeros((m, n))
        jac.fill(-2. / m)
        for i in range(n):
            jac[i,i] += 1
        return jac

    return func, jac, np.ones(n) * factor

def linear_rank1(n, m, factor=1.):
    def func(params):
        vec = np.zeros(m)
        s = 0
        for j in range(n):
            s += (j + 1) * params[j]
        for i in range(m):
            vec[i] = (i + 1) * s - 1
        return vec

    def jac(params):
        jac = np.zeros((m, n))
        for i in range(n):
            for j in range(m):
                jac[j, i] = (i + 1) * (j + 1)
        return jac

    return func, jac, np.ones(n) * factor

def linear_rank1_zero(n, m, factor=1.):
    def func(params, vec=np.zeros(m)):
        s = 0
        for j in range(1, n - 1):
            s += (j + 1) * params[j]
        for i in range(m):
            vec[i] = i * s - 1
        vec[m-1] = -1
        return vec

    def jac(params, jac=np.zeros((m, n))):
        jac.fill(0)
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                jac[j, i] = j * (i + 1)
        return jac

    return func, jac, np.ones(n) * factor

f, jac, x0 = linear_rank1(5, 10)
tol = 1.49012e-08
report(leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))
