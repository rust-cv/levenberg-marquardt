import numpy as np
from scipy.optimize import leastsq

np.set_printoptions(precision=20)


def enorm(v):
    rdwarf = 3.834e-20
    rgiant = 1.304e19
    agiant = rgiant / v.size

    s1 = s2 = s3 = x1max = x3max = 0.

    for i in range(v.size):
        xabs = abs(v[i])

        if xabs > rdwarf and xabs < agiant:
            s2 += xabs**2
        elif xabs <= rdwarf:
            if xabs <= x3max:
                if xabs != 0.:
                    s3 += (xabs / x3max)**2
            else:
                s3 = 1 + s3 * (x3max / xabs)**2
                x3max = xabs
        else:
            if xabs <= x1max:
                s1 += (xabs / x1max)**2
            else:
                s1 = 1. + s1 * (x1max / xabs)**2
                x1max = xabs

    if s1 != 0.:
        return x1max * np.sqrt(s1 + (s2 / x1max) / x1max)

    if s2 == 0.:
        return x3max * np.sqrt(s3)

    if s2 >= x3max:
        return np.sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)))

    return np.sqrt(x3max * ((s2 / x3max) + (x3max * s3)))


def report(f, out):
    print("params = \n  {}".format(out[0]))
    fvec = out[2]['fvec']
    print("objective_function (fvec) = {}".format((enorm(fvec) ** 2) * 0.5))
    # fvec2 = f(out[0])
    # print("objective_function (fvec2) = {}".format((enorm(fvec2) ** 2) * 0.5))
    print("terminate {}, {}".format(out[3], out[4]))


def linear_full_rank(n, m, factor=1.):
    print("\n" + "=" * 80)
    print("linear_full_rank {} {}".format(n, m))
    print("=" * 80)
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
    print("\n" + "=" * 80)
    print("linear_rank1 {} {}".format(n, m))
    print("=" * 80)
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
    print("\n" + "=" * 80)
    print("linear_rank1_zero {} {}".format(n, m))
    print("=" * 80)
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
    print("\n" + "=" * 80)
    print("rosenbruck")
    print("=" * 80)
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
    print("\n" + "=" * 80)
    print("helical_valley")
    print("=" * 80)
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
    print("\n" + "=" * 80)
    print("powell_singular")
    print("=" * 80)
    def func(params, vec=np.zeros(4)):
        vec[0] = params[0] + 10 * params[1]
        vec[1] = np.sqrt(5) * (params[2] - params[3])
        vec[2] = (params[1] - 2 * params[2])**2
        vec[3] = np.sqrt(10) * (params[0] - params[3])**2
        return vec

    def jac(params, jac=np.zeros((4,4))):
        jac.fill(0)
        f = np.sqrt(5)
        t = np.sqrt(10)
        tmp1 = params[1] - 2 * params[2]
        tmp2 = params[0] - params[3]
        jac[0,0] = 1.
        jac[0,3] = 2. * t * tmp2
        jac[1,0] = 10.
        jac[1,2] = 2. * tmp1
        jac[2,1] = f
        jac[2,2] = -4. * tmp1
        jac[3,1] = -f
        jac[3,3] = -2. * t * tmp2
        return jac.T

    return func, jac, np.asfarray([3, -1, 0, 1])


def freudenstein_roth():
    print("\n" + "=" * 80)
    print("freudenstein_roth")
    print("=" * 80)

    def func(params, vec=np.zeros(2)):
        vec[0] = -13 + params[0] + ((5 - params[1]) * params[1] - 2) * params[1]
        vec[1] = -29 + params[0] + ((1 + params[1]) * params[1] - 14) * params[1]
        return vec

    def jac(params, jac=np.zeros((2,2))):
        jac[0] = 1
        jac[1,0] = params[1] * (10 - 3 * params[1]) - 2
        jac[1,1] = params[1] * (2 + 3 * params[1]) - 14
        return jac.T

    return func, jac, np.asfarray([0.5, -2])


tol = 1.49012e-08

f, jac, x0 = linear_full_rank(5, 10)
report(f, leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))
f, jac, x0 = linear_full_rank(5, 50)
report(f, leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = linear_rank1(5, 10)
report(f, leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))
f, jac, x0 = linear_rank1(5, 50)
report(f, leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = linear_rank1_zero(5, 10)
report(f, leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))
f, jac, x0 = linear_rank1_zero(5, 50)
report(f, leastsq(f, x0, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = rosenbruck()
for scale in (1., 10., 100.):
    print("initial: {}".format(x0))
    print("SCALE = {}".format(scale))
    report(f, leastsq(f, x0 * scale, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = helical_valley()
for scale in (1., 10., 100.):
    print("initial: {}".format(x0))
    print("SCALE = {}".format(scale))
    report(f, leastsq(f, x0 * scale, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = powell_singular()
for scale in (1., 10., 100.):
    print("initial: {}".format(x0))
    print("SCALE = {}".format(scale))
    report(f, leastsq(f, x0 * scale, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = freudenstein_roth()
for scale in (1., 10., 100.):
    print("initial: {}".format(x0))
    print("SCALE = {}".format(scale))
    report(f, leastsq(f, x0 * scale, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))
