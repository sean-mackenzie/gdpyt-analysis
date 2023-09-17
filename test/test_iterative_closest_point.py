

"""
def icp(a, b,
        max_time = 1
    ):
    import cv2
    import numpy
    import copy
    import pylab
    import time
    import sys
    import sklearn.neighbors
    import scipy.optimize



    def res(p ,src ,dst):
        T = numpy.matrix([[numpy.cos(p[2]) ,-numpy.sin(p[2]) ,p[0]],
        [numpy.sin(p[2]), numpy.cos(p[2]) ,p[1]],
        [0 ,0, 1]])
        n = numpy.size(src, 0)
        xt = numpy.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = numpy.zeros(numpy.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        r = numpy.sum(numpy.square(d[:, 0]) + numpy.square(d[:, 1]))
        return r

    def jac(p, src, dst):
        T = numpy.matrix([[numpy.cos(p[2]), -numpy.sin(p[2]), p[0]],
                          [numpy.sin(p[2]), numpy.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = numpy.size(src, 0)
        xt = numpy.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = numpy.zeros(numpy.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        dUdth_R = numpy.matrix([[-numpy.sin(p[2]), -numpy.cos(p[2])],
                                [numpy.cos(p[2]), -numpy.sin(p[2])]])
        dUdth = (src * dUdth_R.T).A
        g = numpy.array([numpy.sum(2 * d[:, 0]),
                         numpy.sum(2 * d[:, 1]),
                         numpy.sum(2 * (d[:, 0] * dUdth[:, 0] + d[:, 1] * dUdth[:, 1]))])
        return g

    def hess(p, src, dst):
        n = numpy.size(src, 0)
        T = numpy.matrix([[numpy.cos(p[2]), -numpy.sin(p[2]), p[0]],
                          [numpy.sin(p[2]), numpy.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = numpy.size(src, 0)
        xt = numpy.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = numpy.zeros(numpy.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        dUdth_R = numpy.matrix([[-numpy.sin(p[2]), -numpy.cos(p[2])], [numpy.cos(p[2]), -numpy.sin(p[2])]])
        dUdth = (src * dUdth_R.T).A
        H = numpy.zeros([3, 3])
        H[0, 0] = n * 2
        H[0, 2] = numpy.sum(2 * dUdth[:, 0])
        H[1, 1] = n * 2
        H[1, 2] = numpy.sum(2 * dUdth[:, 1])
        H[2, 0] = H[0, 2]
        H[2, 1] = H[1, 2]
        d2Ud2th_R = numpy.matrix([[-numpy.cos(p[2]), numpy.sin(p[2])], [-numpy.sin(p[2]), -numpy.cos(p[2])]])
        d2Ud2th = (src * d2Ud2th_R.T).A
        H[2, 2] = numpy.sum(2 * (
                    numpy.square(dUdth[:, 0]) + numpy.square(dUdth[:, 1]) + d[:, 0] * d2Ud2th[:, 0] + d[:, 0] * d2Ud2th[
                                                                                                                :, 0]))
        return H

    t0 = time.time()
    init_pose = (0, 0, 0)
    src = numpy.array([a.T], copy=True).astype(numpy.float32)
    dst = numpy.array([b.T], copy=True).astype(numpy.float32)
    Tr = numpy.array([[numpy.cos(init_pose[2]), -numpy.sin(init_pose[2]), init_pose[0]],
                      [numpy.sin(init_pose[2]), numpy.cos(init_pose[2]), init_pose[1]],
                      [0, 0, 1]])
    print("src", numpy.shape(src))
    print("Tr[0:2]", numpy.shape(Tr[0:2]))
    src = cv2.transform(src, Tr[0:2])
    p_opt = numpy.array(init_pose)
    T_opt = numpy.array([])
    error_max = sys.maxsize
    first = False
    while not (first and time.time() - t0 > max_time):
        distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', p=3).fit(
            dst[0]).kneighbors(src[0])
        p = scipy.optimize.minimize(res, [0, 0, 0], args=(src[0], dst[0, indices.T][0]), method='Newton-CG', jac=jac,
                                    hess=hess).x
        T = numpy.array([[numpy.cos(p[2]), -numpy.sin(p[2]), p[0]], [numpy.sin(p[2]), numpy.cos(p[2]), p[1]]])
        p_opt[:2] = (p_opt[:2] * numpy.matrix(T[:2, :2]).T).A
        p_opt[0] += p[0]
        p_opt[1] += p[1]
        p_opt[2] += p[2]
        src = cv2.transform(src, T)
        Tr = (numpy.matrix(numpy.vstack((T, [0, 0, 1]))) * numpy.matrix(Tr)).A
        error = res([0, 0, 0], src[0], dst[0, indices.T][0])

        if error < error_max:
            error_max = error
            first = True
            T_opt = Tr

    p_opt[2] = p_opt[2] % (2 * numpy.pi)
    return T_opt, error_max


def main():
    import cv2
    import numpy
    import random
    import matplotlib.pyplot
    n1 = 100
    n2 = 75
    bruit = 1 / 10
    center = [random.random() * (2 - 1) * 3, random.random() * (2 - 1) * 3]
    radius = random.random()
    deformation = 2

    template = numpy.array([
        [numpy.cos(i * 2 * numpy.pi / n1) * radius * deformation for i in range(n1)],
        [numpy.sin(i * 2 * numpy.pi / n1) * radius for i in range(n1)]
    ])

    data = numpy.array([
        [numpy.cos(i * 2 * numpy.pi / n2) * radius * (1 + random.random() * bruit) + center[0] for i in range(n2)],
        [numpy.sin(i * 2 * numpy.pi / n2) * radius * deformation * (1 + random.random() * bruit) + center[1] for i in
         range(n2)]
    ])

    T, error = icp(data, template)
    dx = T[0, 2]
    dy = T[1, 2]
    rotation = numpy.arcsin(T[0, 1]) * 360 / 2 / numpy.pi

    print("T", T)
    print("error", error)
    print("rotation°", rotation)
    print("dx", dx)
    print("dy", dy)

    result = cv2.transform(numpy.array([data.T], copy=True).astype(numpy.float32), T).T
    matplotlib.pyplot.plot(template[0], template[1], label="template")
    matplotlib.pyplot.plot(data[0], data[1], label="data")
    matplotlib.pyplot.plot(result[0], result[1], label="result: " + str(rotation) + "° - " + str([dx, dy]))
    matplotlib.pyplot.legend(loc="upper left")
    matplotlib.pyplot.axis('square')
    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()"""

from utils import iterative_closest_point as icp
import numpy as np
import time
# import icp

# Constants
N = 10                                      # number of random points in the dataset
num_tests = 100                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp(B, A, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
        assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return


if __name__ == "__main__":
    test_best_fit()
    test_icp()