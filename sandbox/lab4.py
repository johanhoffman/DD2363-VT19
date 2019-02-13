import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, linalg

def riemann_sum(f, a, b, n):
    h = (b-a) / n
    x = np.linspace(a, b, n+1)
    return f(x[:-1]+h/2).sum()/n * (b-a)

def riemann_sum_2D(f, a, b, c, d, n):
    h_x = (b-a) / n
    h_y = (d-c) / n
    x = np.linspace(a, b, n+1)
    y = np.linspace(a, b, n+1)
    grid_x, grid_y = np.meshgrid(x, y)
    return f(grid_x[:-1,:-1]+h_x/2, grid_y[:-1,:-1] + h_y/2).sum()/n**2 * (b-a) * (d-c)


def trap_sum(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    return (y.sum() + y[1:-1].sum())/(2*n) * (b-a)

def gauss_quad(f, n = 1):
    if n == 1:
        return f(.5)
    if n == 2:
        return .5*(f(.5 - .5*3**-.5) + f(.5 + .5*3**-.5))
    if n == 3:
        return 4/9*f(.5) + 5/18 * (f(.5 - .5*(3/5)**.5) + f(.5 + .5*(3/5)**.5))

def gauss_quad_2D(f, n = 1):
    if n == 1:
        return 1/2 * f(1/3, 1/3)
    if n == 2:
        return 1/4 * (f(1/3 *(1 + .5**.5), 1/3 *(1 + .5**.5)) 
            + f(1/3 *(1 - .5**.5), 1/3 *(1 - .5**.5)))


def l2_project(f, nodes, quad_points):
    # number of inner nodes
    m = len(nodes)-2
    # array of pairwise differences
    h = np.diff(nodes)
    M = np.zeros((m+2, m+2))
    b = np.zeros(m+2)

    # fill values of M
    np.fill_diagonal(M[1:,:], h/6)
    np.fill_diagonal(M[:,1:], h/6)
    np.fill_diagonal(M[1:,1:], h/3)
    M[np.diag_indices_from(M[:-1,:-1])] += h/3

    # fill values of b
    b_0 = lambda x: f(x * h[0] + nodes[0]) * (1-x)
    b_m1 = lambda x: f(x * h[m] + nodes[m]) * x
    b[0] = gauss_quad(b_0, quad_points) * h[0]
    b[m+1] = gauss_quad(b_m1, quad_points) * h[m]

    for k in range(1, m+1):
        b_l = lambda x: f(x * h[k-1] + nodes[k-1])*x
        b_r = lambda x: f(x * h[k] + nodes[k])*(1-x)
        b[k] = gauss_quad(b_l, quad_points) * h[k-1] + gauss_quad(b_r, quad_points) * h[k]

    # solve for x
    x = np.linalg.lstsq(M, b, rcond = None)[0]
    # calculate areas under all piecewise linear functions
    s = .5*(np.dot(x[:-1], h) + np.dot(x[1:], h))
    return s

def l2_project_2D(f, nodes):
    # number of inner nodes
    m_x = nodes[0].shape[1] - 2
    m_y = nodes[1].shape[0] - 2
    n_nodes = (m_x+2)*(m_y+2)
    # array of pairwise differences
    h_y = np.diff(nodes[1], axis = 0)
    h_x = np.diff(nodes[0], axis = 1)
    # print(h_x)
    # print(h_y)
    np.set_printoptions(linewidth = 160)
    M = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            row_i = i // (m_x+2)
            col_i = i % (m_x+2)
            row_j = j // (m_x+2)
            col_j = j % (m_x+2)
            if i == j:
                # print("\ntesting:", row, col)
                M[i,j] += 4
                if 0<row_i<m_x+1:
                    # print("mid row")
                    M[i,j] *= 2
                if 0<col_i<m_y+1:
                    # print("mid col")
                    M[i,j] *= 2
                # print("values:", row, col, M[i,j])
            if row_i == row_j and col_i - col_j == 1:
                M[i,j] += 1
                if 0<row_i<m_x+1:
                    M[i,j] *= 2
                M[j,i] = M[i,j]

            if col_i == col_j and row_i - row_j == 1:
                M[i,j] += 1
                if 0<col_i<m_x+1:
                    M[i,j] *= 2
                M[j,i] = M[i,j]

    M /= 48

    for i in range(0, n_nodes):
        row = i // (m_x+2)
        col = i % (m_x+2)
        if row > 0 and col > 0:
            x0 = np.array([nodes[0][row-1, col], nodes[1][row-1, col]])
            x1 = np.array([nodes[0][row, col-1], nodes[1][row, col-1]])
            x2 = np.array([nodes[0][row, col], nodes[1][row, col]])
            avg = (x0+x1+x2)/3
            b[i] += f(avg[0], avg[1])/6
        if row < m_y+1 and col > 0:
            x0 = np.array([nodes[0][row+1, col], nodes[1][row+1, col]])
            x1 = np.array([nodes[0][row, col-1], nodes[1][row, col-1]])
            x2 = np.array([nodes[0][row, col], nodes[1][row, col]])
            avg = (x0+x1+x2)/3
            b[i] += f(avg[0], avg[1])/6
        if col < m_x+1 and row > 0:
            x0 = np.array([nodes[0][row, col+1], nodes[1][row, col+1]])
            x1 = np.array([nodes[0][row-1, col], nodes[1][row-1, col]])
            x2 = np.array([nodes[0][row, col], nodes[1][row, col]])
            avg = (x0+x1+x2)/3
            b[i] += f(avg[0], avg[1])/6
        if col < m_x+1 and row < m_y+1:
            x0 = np.array([nodes[0][row, col], nodes[1][row, col]])
            x1 = np.array([nodes[0][row, col+1], nodes[1][row, col+1]])
            x2 = np.array([nodes[0][row+1, col], nodes[1][row+1, col]])
            avg = (x0+x1+x2)/3
            b[i] += f(avg[0], avg[1])/6

    # solve for x
    x = np.linalg.lstsq(M, b, rcond = None)[0]
    s = 0

    for i in range(n_nodes):
        row = i // (m_x+2)
        col = i % (m_x+2)
        s_add = 0
        if row > 0 and col > 0:
            s_add += x[i] * h_x[row, col-1] * h_y[row-1, col] / 6
        if row < m_y+1 and col > 0:
            s_add += x[i] * h_x[row, col-1] * h_y[row, col] / 6
        if col < m_x+1 and row > 0:
            s_add += x[i] * h_x[row, col] * h_y[row-1, col] / 6
        if col < m_x+1 and row < m_y+1:
            s_add += x[i] * h_x[row, col] * h_y[row, col] / 6
        s += s_add

    # calculate areas under all piecewise linear functions
    return s * 9/8


def test_gauss_quad_1D():
    # test first degree polynomials
    for _ in range(100):
        a = np.random.randn()
        b = np.random.randn()
        f = lambda x: a*x+b
        true_value = (f(0) + f(1))/2
        assert np.isclose(gauss_quad(f, 1), true_value)

    # test second degree polynomials
    for _ in range(100):
        a = np.random.randn()
        b = np.random.randn()
        c = np.random.randn()

        f = lambda x: a*x**2 + b*x + c
        true_value = a / 3 + b / 2 + c
        test_value = gauss_quad(f, 2)
        assert np.isclose(test_value, true_value)
    print("Test passed!")

def test_gauss_quad_2D():
    # test first degree polynomials
    for _ in range(100):
        a = np.random.randn()
        b = np.random.randn()
        c = np.random.randn()
        f = lambda x, y: a*x + b*y + c
        true_value = a/6 + b/6 + c/2
        test_value = gauss_quad_2D(f, 1)
        assert np.isclose(test_value, true_value)

    # test second degree polynomials
    for _ in range(100):
        a = np.random.randn()
        b = np.random.randn()
        c = np.random.randn()
        d = np.random.randn()
        e = np.random.randn()

        f = lambda x, y: a*x**2 + b*y**2 + c*x + d*y + e
        true_value = a / 12 + b / 12 + c/6 + d/6 + e/2
        test_value = gauss_quad_2D(f, 2)
        assert np.isclose(test_value, true_value)
    print("Test passed!")


def test_l2_project_1D():
    functions = [
        lambda x: x,
        lambda x: x**2,
        lambda x: np.exp(x),
        lambda x: np.sin(x)
    ]
    intervals = [
        (0, 1),
        (1, 2),
        (-1, 1),
        (5, 13),
        (-6, -2),
        (1, 1)
    ]

    for i, f in enumerate(functions):
        for interval in intervals:
            nodes = np.linspace(interval[0], interval[1], 100)
            true_value = trap_sum(f, interval[0], interval[1], 10000)
            test_value = l2_project(f, nodes, 2)
            assert np.isclose(test_value, true_value)

    print("Test passed!")

def test_l2_project_2D():
    functions = [
        lambda x, y: (x+y)*0 + 1,
        lambda x, y: x**2 + y**2,
        lambda x, y: np.exp(x + y),
        lambda x, y: np.sin(x + y)
    ]
    intervals = [
        (0, 1, 0, 1),
        (1, 2, 1, 2),
        (-1, 1, -1, 1),
        (5, 13, 5, 13),
        (-6, -2, -6, -2)
    ]

    for i, f in enumerate(functions):
        for i in intervals:
            grid_x = np.linspace(i[0], i[1], 20)
            grid_y = np.linspace(i[2], i[3], 20)
            nodes = np.meshgrid(grid_x, grid_y)
            # print(nodes_x)
            # print(nodes_y)
            true_value = riemann_sum_2D(f, i[0], i[1], i[2], i[3], 1000)
            test_value = l2_project_2D(f, nodes)
            # print(true_value, test_value, test_value / true_value)
            assert np.isclose(test_value, true_value, rtol = 1e-2)

    print("Test passed!")


def test_l2_project_convergence():
    errs_1 = []
    errs_2 = []
    errs_3 = []
    errs_trap = []
    errs_rie = []
    ns = list(range(10, 500, 10))
    for n in ns:
        f = lambda x: np.exp(x)
        nodes = [1 + 2*i/n for i in range(n+1)]

        true_value = np.exp(nodes[n]) - np.exp(nodes[0])
        test_value_1 = l2_project(f, nodes, 1)
        test_value_2 = l2_project(f, nodes, 2)
        test_value_3 = l2_project(f, nodes, 3)
        test_trap = trap_sum(f, 1, 3, n)
        test_rie = riemann_sum(f, 1, 3, n)
        errs_1.append(abs(test_value_1 - true_value))
        errs_2.append(abs(test_value_2 - true_value))
        errs_3.append(abs(test_value_3 - true_value))
        errs_trap.append(abs(test_trap - true_value))
        errs_rie.append(abs(test_rie - true_value))
        # assert np.isclose(test_value, true_value)

    conv_rate_1 = -(np.log(errs_1[5]) - np.log(errs_1[0])) / (np.log(ns[5]) - np.log(ns[0]))
    conv_rate_2 = -(np.log(errs_2[5]) - np.log(errs_2[0])) / (np.log(ns[5]) - np.log(ns[0]))
    conv_rate_3 = -(np.log(errs_3[5]) - np.log(errs_3[0])) / (np.log(ns[5]) - np.log(ns[0]))
    plt.loglog(ns, errs_1, c = "b", marker = "*", ls = '', label = "L2 1-point")
    plt.loglog(ns, errs_2, c = "b", ls = "--", label = "L2 2-point")
    plt.loglog(ns, errs_3, c = "b", ls = "-.", label = "L2 3-point")
    plt.loglog(ns, errs_trap, c = "r", label = "Trapezoid sum")
    plt.loglog(ns, errs_rie, c = "g", label = "Midpoint sum")
    plt.text(70, 3*10**-3, "q = %.2f" % conv_rate_1)
    plt.text(33, 10**-7, "q = %.2f" % conv_rate_2)
    plt.text(21, 10**-11, "q = %.2f" % conv_rate_3)

    plt.legend()
    plt.show()

def test_l2_project_convergence_2D():
    errs = []
    errs_rie = []
    ns = list(range(10, 30, 2))
    for n in ns:
        f = lambda x, y: np.exp(x + y)
        grid_x = np.linspace(1, 3, n)
        grid_y = np.linspace(1, 3, n)
        nodes = np.meshgrid(grid_x, grid_y)

        true_value = np.exp(2)*(np.exp(2)-1)**2
        test_value = l2_project_2D(f, nodes)
        test_rie = riemann_sum_2D(f, 1, 3, 1, 3, n)
        errs.append(abs(test_value - true_value))
        errs_rie.append(abs(test_rie - true_value))
        # assert np.isclose(test_value, true_value)

    conv_rate = -(np.log(errs[5]) - np.log(errs[0])) / (np.log(ns[5]) - np.log(ns[0]))
    plt.loglog(ns, errs, c = "b", marker = "*", ls = '', label = "L2 2D")
    plt.loglog(ns, errs_rie, c = "g", label = "Midpoint sum")
    plt.text(70, 3*10**-3, "q = %.2f" % conv_rate)

    plt.legend()
    plt.show()



test_gauss_quad_1D()
test_gauss_quad_2D()
test_l2_project_1D()
test_l2_project_2D()
# test_l2_project_convergence()
test_l2_project_convergence_2D()


"""
a = 0
b = 1
ns = [i*10**j for j in range(7) for i in [1, 2, 5]]
err0 = []
err1 = []
err2 = []
for n in ns:
    i0 = riemann_sum(f, a, b, n)
    i1 = trap_sum(f, a, b, n)
    i2 = gauss_quad(f)
    true_value = 8
    err0.append(abs(i0-true_value))
    err1.append(abs(i1-true_value))
    err2.append(abs(i2-true_value))
    print(i0, i1, i2, true_value)

plt.loglog(ns, err0)
plt.loglog(ns, err1)
plt.loglog(ns, err2)
plt.show()
"""