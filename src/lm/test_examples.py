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


def rosenbruck():
    def func(params, vec=np.zeros(2)):
        vec[0] = 10 * (params[1] - params[0]**2)
        vec[1] = 1 - params[0]
        return vec

    def jac(params, jac=np.zeros((2, 2))):
        jac[0,0] = -20 * params[0]
        jac[0,1] = -1
        jac[1,0] = 10
        jac[1,1] = 0
        return jac.T

    return func, jac, np.asfarray([-1.2, 1])


def helical_valley():
    tpi = 2 * np.pi

    def func(params, vec=np.zeros(3)):
        if params[0] == 0:
            tmp1 = np.copysign(0.25, params[1])
        elif params[0] > 0:
            tmp1 = np.arctan(params[1] / params[0]) / tpi
        else:
            tmp1 = np.arctan(params[1] / params[0]) / tpi + 0.5

        tmp2 = np.sqrt(params[0]**2 + params[1]**2)

        vec[0] = 10 * (params[2] - 10 * tmp1)
        vec[1] = 10 * (tmp2 - 1)
        vec[2] = params[2]
        return vec

    def jac(params, jac=np.zeros((3,3))):
        temp = params[0]**2 + params[1]**2
        tmp1 = tpi * temp
        tmp2 = np.sqrt(temp)
        jac[0,0] = 100 * params[1] / tmp1
        jac[0,1] = 10 * params[0] / tmp2
        jac[0,2] = 0

        jac[1,0] = -100 * params[0] / tmp1
        jac[1,1] = 10 * params[1] / tmp2
        jac[1,2] = 0
        
        jac[2,0] = 10
        jac[2,1] = 0
        jac[2,2] = 1
        return jac.T

    return func, jac, np.asfarray([-1, 0, 0])


def powell_singular():
    def func(params, vec=np.zeros(4)):
        vec[0] = params[0] + 10 * params[1]
        vec[1] = np.sqrt(5) * (params[2] - params[3])
        vec[2] = (params[1] - 2 * params[2])**2
        vec[3] = np.sqrt(10) * (params[0] - params[3])**2
        return vec

    def jac(params, jac=np.zeros((4,4))):
        jac.fill(0)
        jac[0,0] = 1
        jac[0,3] = 2 * np.sqrt(10) * (params[0] - params[3])
        jac[1,0] = 10
        jac[1,2] = 2 * (params[1] - 2 * params[2])
        jac[2,1] = np.sqrt(5)
        jac[2,2] = -2 * jac[2,1]
        jac[3,1] = -np.sqrt(5)
        jac[3,3] = -jac[3,0]
        return jac.T

    return func, jac, np.asfarray([3, -1, 0, 1])


tol = 1.49012e-08
f, jac, x0 = linear_rank1(5, 10)
report(leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))
