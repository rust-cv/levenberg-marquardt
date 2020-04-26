import re
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
    reason = out[3]
    if out[4] == 1:
        reason = "ftol"
    elif out[4] == 2:
        reason = "xtol"
    elif out[4] == 3:
        reason = "ftol, xtol"
    elif out[4] == 4:
        reason = "gtol"
    print("assert_eq!(report.termination, {});".format(reason))
    print("assert_eq!(report.number_of_evaluations, {});".format(out[2]['nfev']))
    fvec = out[2]['fvec']
    print("assert_relative_eq!(report.objective_function, {});".format((enorm(fvec) ** 2) * 0.5))
    params = re.sub(r'(?<=\d)([\n \]]+)', ', ', str(out[0]))[1:]
    print("params = {}".format(params))


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


def bard():
    print("\n" + "=" * 80)
    print("bard")
    print("=" * 80)

    y1 = np.asfarray([0.14, 0.18, 0.22, 0.25, 0.29,
                      0.32, 0.35, 0.39, 0.37, 0.58,
                      0.73, 0.96, 1.34, 2.10, 4.39])

    def func(params, vec=np.zeros(15)):
        for i in range(15):
            tmp2 = 15 - i

            if i > 7:
                tmp3 = tmp2
            else:
                tmp3 = i + 1

            vec[i] = y1[i] - (params[0] + (i + 1) / (params[1] * tmp2 + params[2] * tmp3))
        return vec

    def jac(params, jac=np.zeros((3, 15))):
        for i in range(15):
            tmp2 = 15 - i

            if i > 7:
                tmp3 = tmp2
            else:
                tmp3 = i + 1

            tmp4 = (params[1] * tmp2 + params[2] * tmp3)**2
            jac[0,i] = -1
            jac[1,i] = (i + 1) * tmp2 / tmp4
            jac[2,i] = (i + 1) * tmp3 / tmp4
        return jac.T

    return func, jac, np.asfarray([1, 1, 1])


def kowalik_osborne():
    print("\n" + "=" * 80)
    print("kowalik_osborne")
    print("=" * 80)
    v = np.asfarray([4, 2, 1, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625])
    y2 = np.asfarray([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456,
                      0.0342, 0.0323, 0.0235, 0.0246])

    def func(params, vec=np.zeros(11)):
        tmp1 = v * (v + params[1])
        tmp2 = v * (v + params[2]) + params[3]
        vec[:] = y2 - params[0] * tmp1 / tmp2
        return vec

    def jac(params, jac=np.zeros((4,11))):
        tmp1 = v * (v + params[1])
        tmp2 = v * (v + params[2]) + params[3]
        jac[0] = -tmp1 / tmp2
        jac[1] = -v * params[0] / tmp2
        jac[2] = jac[0] * jac[1]
        jac[3] = jac[2] / v
        return jac.T

    return func, jac, np.asfarray([0.25, 0.39, 0.415, 0.39])


def meyer():
    print("\n" + "=" * 80)
    print("meyer")
    print("=" * 80)
    y3 = np.asarray([3.478e4, 2.861e4, 2.365e4, 1.963e4, 1.637e4, 1.372e4, 1.154e4,
                     9.744e3, 8.261e3, 7.03e3, 6.005e3, 5.147e3, 4.427e3, 3.82e3,
                     3.307e3, 2.872e3])

    def func(params):
        temp = 5 * (np.arange(16) + 1) + 45 + params[2]
        return params[0] * np.exp(params[1] / temp) - y3

    def jac(params):
        jac=np.zeros((3,16))
        temp = 5 * (np.arange(16) + 1) + 45 + params[2]
        tmp1 = params[1] / temp
        tmp2 = np.exp(tmp1)
        jac[0] = tmp2
        jac[1] = params[0] * tmp2 / temp
        jac[2] = -(params[0] * tmp2) * tmp1 / temp
        return jac.T

    return func, jac, np.asfarray([0.02, 4000, 250])


def watson(n_params):
    print("\n" + "=" * 80)
    print("watson {}".format(n_params))
    print("=" * 80)

    def func(params):
        div = (np.arange(29) + 1.) / 29
        s1 = 0
        dx = 1

        for j in range(1, params.size):
            s1 += (j * params[j]) * dx
            dx *= div

        s2 = 0
        dx = 1

        for j in range(params.size):
            s2 += dx * params[j]
            dx *= div

        vec = np.zeros(31)
        vec[:29] = (s1 - s2**2) - 1
        vec[29] = params[0]
        vec[30] = params[1] - params[0]**2 - 1
        return vec

    def jac(params):
        jac = np.zeros((n_params, 31))
        div = (np.arange(29) + 1.) / 29
        s2 = 0
        dx = 1

        for j in range(params.size):
            s2 += dx * params[j]
            dx *= div

        temp = 2 * div * s2
        dx = 1. / div

        for j in range(params.size):
            jac[j,:29] = -dx * temp + j * dx
            dx *= div

        jac[0,29] = 1
        jac[0,30] = -2 * params[0]
        jac[1,30] = 1
        return jac.T

    return func, jac, np.zeros(n_params)


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

f, jac, x0 = bard()
for scale in (1., 10., 100.):
    print("initial: {}".format(x0))
    print("SCALE = {}".format(scale))
    report(f, leastsq(f, x0 * scale, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = kowalik_osborne()
for scale in (1., 10., 100.):
    print("initial: {}".format(x0))
    print("SCALE = {}".format(scale))
    report(f, leastsq(f, x0 * scale, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = meyer()
for scale in (1., 10.):
    print("initial: {}".format(x0))
    print("SCALE = {}".format(scale))
    report(f, leastsq(f, x0 * scale, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = watson(6)
for offset in (0., 10., 100.):
    print("\nOFFSET: {}".format(offset))
    report(f, leastsq(f, x0 + offset, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = watson(9)
for offset in (0., 10., 100.):
    print("\nOFFSET: {}".format(offset))
    report(f, leastsq(f, x0 + offset, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))

f, jac, x0 = watson(12)
for offset in (0., 10., 100.):
    print("\nOFFSET: {}".format(offset))
    report(f, leastsq(f, x0 + offset, Dfun=jac, ftol=tol, xtol=tol, gtol=0., full_output=True))