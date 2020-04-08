import os
import sys

import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from main import readVTP, getMinSF, getNearpoints
import pymesh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def surface_curvature(X, Y, Z):
    print(X.shape)
    (lr, lb) = X.shape

    print(lr)
    # print("awfshss-------------")
    print(lb)
    # First Derivatives
    Xv, Xu = np.gradient(X)
    Yv, Yu = np.gradient(Y)
    Zv, Zu = np.gradient(Z)
    print(Xu)

    # Second Derivatives
    Xuv, Xuu = np.gradient(Xu)
    Yuv, Yuu = np.gradient(Yu)
    Zuv, Zuu = np.gradient(Zu)

    Xvv, Xuv = np.gradient(Xv)
    Yvv, Yuv = np.gradient(Yv)
    Zvv, Zuv = np.gradient(Zv)

    # 2D to 1D conversion
    # Reshape to 1D vectors
    Xu = np.reshape(Xu, lr * lb)
    Yu = np.reshape(Yu, lr * lb)
    Zu = np.reshape(Zu, lr * lb)
    Xv = np.reshape(Xv, lr * lb)
    Yv = np.reshape(Yv, lr * lb)
    Zv = np.reshape(Zv, lr * lb)
    Xuu = np.reshape(Xuu, lr * lb)
    Yuu = np.reshape(Yuu, lr * lb)
    Zuu = np.reshape(Zuu, lr * lb)
    Xuv = np.reshape(Xuv, lr * lb)
    Yuv = np.reshape(Yuv, lr * lb)
    Zuv = np.reshape(Zuv, lr * lb)
    Xvv = np.reshape(Xvv, lr * lb)
    Yvv = np.reshape(Yvv, lr * lb)
    Zvv = np.reshape(Zvv, lr * lb)

    Xu = np.c_[Xu, Yu, Zu]
    Xv = np.c_[Xv, Yv, Zv]
    Xuu = np.c_[Xuu, Yuu, Zuu]
    Xuv = np.c_[Xuv, Yuv, Zuv]
    Xvv = np.c_[Xvv, Yvv, Zvv]

    # % First fundamental Coeffecients of the surface (E,F,G)

    E = np.einsum('ij,ij->i', Xu, Xu)
    F = np.einsum('ij,ij->i', Xu, Xv)
    G = np.einsum('ij,ij->i', Xv, Xv)

    m = np.cross(Xu, Xv, axisa=1, axisb=1)
    p = np.sqrt(np.einsum('ij,ij->i', m, m))
    n = m / np.c_[p, p, p]
    # n is the normal
    # % Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
    L = np.einsum('ij,ij->i', Xuu, n)  # e
    M = np.einsum('ij,ij->i', Xuv, n)  # f
    N = np.einsum('ij,ij->i', Xvv, n)  # g

    # Alternative formula for gaussian curvature in wiki
    # K = det(second fundamental) / det(first fundamental)
    # % Gaussian Curvature
    K = (L * N - M ** 2) / (E * G - F ** 2)
    K = np.reshape(K, lr * lb)
    print(K.size)
    # wiki trace of (second fundamental)(first fundamental inverse)
    # % Mean Curvature
    H = ((E * N + G * L - 2 * F * M) / ((E * G - F ** 2))) / 2
    print(H.shape)
    H = np.reshape(H, lr * lb)
    print(H.size)

    # % Principle Curvatures
    Pmax = H + np.sqrt(H ** 2 - K)
    Pmin = H - np.sqrt(H ** 2 - K)
    # [Pmax, Pmin]
    Principle = [Pmax, Pmin]
    return Principle


def fun(x, y):
    return x ** 2 + y ** 2


def main():
    file = '10000004_FF_A38900064_00_DFPD_120_Deckel_Fv_Max.odb__P2325PF21_01_DFPD_120_DECKEL.vtu.vtp'
    mesh_connect, data = readVTP(file)
    print(mesh_connect[:5])
    print(data[:5])
    N = np.max(mesh_connect) + 1
    mesh = pymesh.form_mesh(data[:, 1:], mesh_connect)
    print(mesh.num_vertices, mesh.num_faces, mesh.num_voxels)

    # ============get the connected vertices================
    # adjacencyM = {}
    # for (a, b, c) in mesh:
    #     if a in adjacencyM:
    #         adjacencyM[a].add(b)
    #         adjacencyM[a].add(c)
    #     else:
    #         adjacencyM[a] = {b, c}
    #
    #     if b in adjacencyM:
    #         adjacencyM[b].add(a)
    #         adjacencyM[b].add(c)
    #     else:
    #         adjacencyM[b] = {a, c}
    #
    #     if c in adjacencyM:
    #         adjacencyM[c].add(a)
    #         adjacencyM[c].add(b)
    #     else:
    #         adjacencyM[c] = {a, b}
    # for idx in indexes:
    #     adjacencyM[idx] = adjacencyM[idx].intersection(selected_points)
    #     adjacencyM[idx] = list(adjacencyM[idx])
    #     row = [idx]+adjacencyM[idx]
    #     neighbors_adjancency.append(row)

    # # get neighbors
    # minSF = getMinSF(data)
    # nNeighbors = 1024
    # indexes, neighbors = getNearpoints(data, minSF, nNeighbors)
    # selected_points = set(indexes)
    # points = dict(zip(indexes, neighbors))
    # neighbors_adjancency = np.array(neighbors_adjancency, dtype=object)
    # print(neighbors_adjancency)

    # ======== compute curvature based on mesh====================
    # x = np.linspace(-1, 1, 20)
    # y = np.linspace(-1, 1, 20)
    # [x, y] = np.meshgrid(x, y)
    #
    # z = (x ** 3 + y ** 2 + x * y)

    # # s = nd.gaussian_filter(z,10)
    # temp1 = surface_curvature(x, y, z)
    # print("maximum curvatures")
    # print(temp1[0].shape)
    # print("minimum curvatures")
    # print(temp1[1].shape)
    #
    # fig = pylab.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(x, y, z)
    # pylab.show()


if __name__ == '__main__':
    main()
