import re
import numpy as np
from scipy.optimize import leastsq

np.set_printoptions(precision=20)


def sq(x):
    # We use this because x ** 2 != x * x
    return x * x


def enorm(v):
    rdwarf = 3.834e-20
    rgiant = 1.304e19
    agiant = rgiant / v.size

    s1 = s2 = s3 = x1max = x3max = 0.

    for i in range(v.size):
        xabs = abs(v[i])

        if xabs >= agiant or xabs <= rdwarf:
            if xabs > rdwarf:
                # sum for large components
                if xabs > x1max:
                    s1 = 1. + s1 * sq(x1max / xabs)
                    x1max = xabs;
                else:
                    s1 += sq(xabs / x1max)
            else:
                # sum for small components
                if xabs > x3max:
                    s3 = 1. + s3 * sq(x3max / xabs)
                    x3max = xabs;
                elif xabs != 0.:
                    s3 += sq(xabs / x3max)
        else:
            s2 += xabs * xabs

    if s1 != 0.:
        return x1max * np.sqrt(s1 + (s2 / x1max) / x1max)

    if s2 == 0.:
        return x3max * np.sqrt(s3)

    if s2 >= x3max:
        return np.sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)))

    return np.sqrt(x3max * ((s2 / x3max) + (x3max * s3)))


def linear_full_rank(m, n=5, factor=1.):
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


def linear_rank1(m, n=5, factor=1.):
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


def linear_rank1_zero_columns(m, n=5, factor=1.):
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
        vec[0] = 10 * (params[1] - sq(params[0]))
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

        tmp2 = np.sqrt(sq(params[0]) + sq(params[1]))

        vec[0] = 10 * (params[2] - 10 * tmp1)
        vec[1] = 10 * (tmp2 - 1)
        vec[2] = params[2]
        return vec

    def jac(params, jac=np.zeros((3,3))):
        temp = sq(params[0]) + sq(params[1])
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
        vec[2] = sq(params[1] - 2 * params[2])
        vec[3] = np.sqrt(10) * sq(params[0] - params[3])
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

            tmp4 = sq(params[1] * tmp2 + params[2] * tmp3)
            jac[0,i] = -1
            jac[1,i] = (i + 1) * tmp2 / tmp4
            jac[2,i] = (i + 1) * tmp3 / tmp4
        return jac.T

    return func, jac, np.asfarray([1, 1, 1])


def kowalik_osborne():
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
        vec[:29] = (s1 - sq(s2)) - 1
        vec[29] = params[0]
        vec[30] = params[1] - sq(params[0]) - 1
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


def beale():
    def func(params):
        x = params[0]
        y = params[1]
        return np.array([
            sq(1.5 - x + x*y) + sq(2.25 - x + x * sq(y)),
            0.,
        ]);

    def jac(params):
        x = params[0]
        y = params[1]
        y3 = y * y * y
        dx = 0.5 * (-1 + y) * (15 + 9 * y + 4 * x * (-2 + y * y + y3))
        dy = x * (3 + 9 * y + x * (-2 - 2 * y + 4 * y3))
        return np.array([
            [dx, dy],
            [0., 0.],
        ])

    return func, jac, np.array([2.5, 1.])


def generate_test_case(problem_function, arg_sets=[()], offsets=[('')]):
    TOL = 1.49012e-08
    struct_name = re.sub(r'(?:^|_)([a-z])', lambda x: x[1].upper(), problem_function.__name__)

    code = '#[test]\nfn test_{}() {{\n'.format(problem_function.__name__)

    def format_np(arr):
        arr = re.sub(r'^\[\s*', '[', str(arr))
        arr = re.sub(r'e[-+]?0+(?=[^\d])', '', arr)
        return re.sub(r'(?<!\[)(e[+-]?\d+)?(0*[\n \]]+)', r'\1, ', arr)[:-2] + ']'

    nm = None
    for arg_set in arg_sets:
        f, jac, x0 = problem_function(*arg_set)
        n = len(x0)
        m = len(f(x0))
        for i, offset in enumerate(offsets):
            start = eval('x0 ' + offset)
            minpack_output = leastsq(f, start, Dfun=jac, ftol=TOL, xtol=TOL, gtol=0., full_output=True)
            first = nm is None
            n_vec = 'VectorN::<f64, U{}>::from_column_slice(&{{}})'.format(len(start))
            if first or nm != (n, m):
                nm = (n, m)
                if not first:
                    code += '\n'
                code += '    let mut problem = {}'.format(struct_name)
                v = 'VectorN::<f64, U{}>::zeros()'.format(n)
                if arg_set:
                    code += '::new(\n        {},\n'.format(v)
                    code += ',\n'.join(' ' * 8 + str(arg) for arg in arg_set)
                    code += ',\n'
                    code += '    );\n'
                else:
                    code += ' { params: ' + v + ' };\n'

                initial = n_vec.format(format_np(x0))
                code += '    let initial = {};\n\n'.format(initial)
            
            if first:
                # create unit test for Jacobian using finite differences
                params = n_vec.format(format_np(np.random.rand(n)))
                code += '    // check derivative implementation\n'
                code += '    problem.set_params(&{});\n'.format(params);
                code += '    let jac_num = differentiate_numerically(&mut problem).unwrap();\n'
                code += '    let jac_trait = problem.jacobian().unwrap();\n'
                code += '    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-5);\n\n'

            x0_code = 'initial.map(|x| x {})'.format(offset) if offset else 'initial.clone()'
            code += '    problem.set_params(&{});\n'.format(x0_code)
            have_more_offsets = i < len(offsets) - 1
            mut = 'mut ' if have_more_offsets else ''
            code += '    let ({}problem, report) = LevenbergMarquardt::new().with_tol(TOL).minimize(problem.clone());\n'.format(mut)

            if minpack_output[4] == 1:
                reason = 'Converged { ftol: true, xtol: false }'
            elif minpack_output[4] == 2:
                reason = 'Converged { ftol: false, xtol: true }'
            elif minpack_output[4] == 3:
                reason = 'Converged { ftol: true, xtol: true }'
            elif minpack_output[4] == 4:
                reason = 'Orthogonal'
            elif minpack_output[4] == 5:
                reason = 'LostPatience'
            elif minpack_output[4] == 8:
                reason = 'NoImprovementPossible("gtol")'
            else:
                raise ValueError('unknown termination reason {}'.format(minpack_output[4]))
            code += '    assert_eq!(report.termination, TerminationReason::{});\n'.format(reason)
            code += '    assert_eq!(report.number_of_evaluations, {});\n'.format(minpack_output[2]['nfev'])
            objective_function = sq(enorm(f(minpack_output[0]))) * 0.5
            code += '    assert_fp_eq!(report.objective_function, {});\n'.format(objective_function)
            params = format_np(minpack_output[0])
            code += '    assert_fp_eq!(problem.params, {});\n'.format(n_vec.format(params))
    code += '}\n'
    return code


if __name__ == '__main__':
    np.random.seed(0)
    print('// This was file was generated by test_examples.py\n')
    print(generate_test_case(linear_full_rank, [(10,), (50,)]))
    print(generate_test_case(linear_rank1, [(10,), (50,)]))
    print(generate_test_case(linear_rank1_zero_columns, [(10,), (50,)]))
    print(generate_test_case(rosenbruck, offsets=['', '* 10.', '* 100.']))
    print(generate_test_case(helical_valley, offsets=['', '* 10.', '* 100.']))
    print(generate_test_case(powell_singular, offsets=['', '* 10.', '* 100.']))
    print(generate_test_case(freudenstein_roth, offsets=['', '* 10.', '* 100.']))
    print(generate_test_case(bard, offsets=['', '* 10.', '* 100.']))
    print(generate_test_case(kowalik_osborne, offsets=['', '* 10.', '* 100.']))
    print(generate_test_case(meyer, offsets=['', '* 10.']))
    print(generate_test_case(watson, [(6,), (9,), (12,)], offsets=['', '+ 10.', '+ 100.']))
    print(generate_test_case(beale, offsets=['', '- 0.5']), end='')