import numpy as np


def calculate_R_s(s, n, sigma=1.6, epsilon=0.002):
    """
    R_s = G_s(G_s + epsilon * I_n)^-1,
    where G_s is the centered gram matrix, and n is the sample size of variable s.
    Besides, epsilon is a regulation constant, and epsilon > 0.

    :param s: samples
    :param n: sample size
    :param epsilon: a small constant
    :param sigma: sigma for Gaussian kernel
    :return:
    """
    # Calculate gram matrix
    alpha = np.matrix([np.sum(np.square(s[_])) for _ in range(n)])
    ones = np.matrix(np.ones(shape=[n]))
    amo = alpha.T * ones

    diff2 = 2 * s * s.T - amo - amo.T
    gm = np.exp(diff2 / sigma ** 2)

    # Calculate centered gram matrix, namely G_s
    I_n = np.eye(n)
    N_0 = I_n - np.full([n, n], 1 / n)
    G_s = N_0 * gm * N_0

    # Calculate R_s
    return G_s * (G_s + epsilon * I_n).I


def ConditionalHSIC(x, y, Z):
    """
    Conditional independence test by HSIC measure.

    :param x: variable x in R^m, [1, 2, 3, ...]
    :param y: variable y in R^m, [2, 3, 4, ...]
    :param Z: condition set Z in R^{m x d}, [[1, 2, 3], [2, 3, 4]...]
    :return: is conditional independence?
    """

    # # Length of samples
    # m = np.minimum(len(x), len(y))
    # x = np.matrix(x[: m])
    # y = np.matrix(y[: m])
    # Z = np.matrix(Z[: m])

    # Define Gaussian kernel
    R_x = calculate_R_s(x, len(x))
    R_y = calculate_R_s(y, len(y))
    R_Z = calculate_R_s(Z, len(Z))

    # (7)
    V_yx_Z = R_y.T * R_x - R_y.T * R_Z * R_Z.T * R_x
    return np.trace(V_yx_Z.T * V_yx_Z)

# """
#     Test
# """
#
# if __name__ == '__main__':
#     import time
#
#     t0 = time.time()
#
#     # #
#     # x = np.random.normal(0, 1, size=[1000, 1])
#     # y = np.random.normal(0, 1, size=[1000, 1])
#     # z = np.random.normal(0, 1, size=[1000, 2])
#     # xz = np.concatenate([x, z], axis=1)
#     # yz = np.concatenate([y, z], axis=1)
#
#     cov = [[1, 0.9, 0.2, 0.2],
#            [0.9, 1, 0.5, 0.3],
#            [0.2, 0.5, 1, 0.2],
#            [0.2, 0.3, 0.2, 1]]
#     samples = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=cov, size=2000)
#     print('sample ==\n', samples)
#
#     mci = ConditionalHSIC(samples[:, [0, 2, 3]], samples[:, [1, 2, 3]], samples[:, [2, 3]])
#     print('mci ==', mci)
#
#     # xx = np.random.normal(0, 1, size=[3, 3])
#     # yy = np.random.normal(0, 1, size=[3, 3])
#     # print('xx ==', xx)
#     # print('yy ==', yy)
#     # print(mm(xx, yy))
#     print('time ==', time.time() - t0)
