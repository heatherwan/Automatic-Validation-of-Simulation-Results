import os
import pickle
import sys

import vtk
from scipy import spatial
from vtk.util.numpy_support import vtk_to_numpy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

matplotlib.use('tkagg')
nNeighbor = 1024


def readVTP(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()
    mesh = vtk_to_numpy(poly.GetPolys().GetData())
    mesh = mesh.reshape(mesh.shape[0] // 4, 4)[:, 1:]

    Points = vtk_to_numpy(poly.GetPoints().GetData())
    SF = vtk_to_numpy(poly.GetPointData().GetScalars()).reshape(-1, 1)
    data = np.concatenate((SF, Points), axis=1)
    return mesh, data


def get_MinSF(data):
    idx = np.argmin(data[:, 0])
    return data[idx, :]


def get_Nearpoints(data, MinSF, NNeighbors):
    coord = data[:, 1:]
    dd, indexes = spatial.cKDTree(coord).query(MinSF[1:], NNeighbors)
    nearestpoints = data[indexes, :]
    # print(nearestpoints.shape)
    return indexes, nearestpoints


def main():
    filename = 'wanting_split/test/Good/10000008_FF_ds200_v5_Fv_Min_BREAK.odb__AA_DECKEL.vtu.vtp'
    poly, data = readVTP(filename)
    MinSF = get_MinSF(data)
    points = data[:, 1:]

    indexes, nearpoints1000 = get_Nearpoints(data, MinSF, nNeighbor)
    n = 7
    pcCoordinates = nearpoints1000[:, 1:]
    tree = spatial.cKDTree(pcCoordinates)
    dd, ii = tree.query(pcCoordinates, k=n)

    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection="3d")
    ax.view_init(azim=170, elev=-120)

    # Data for a three-dimensional line
    sc = ax.scatter3D(nearpoints1000[:, 1], nearpoints1000[:, 2], nearpoints1000[:, 3], s=1,
                      c=nearpoints1000[:, 0], marker='o', cmap="bwr")
    ax.scatter3D(MinSF[1], MinSF[2], MinSF[3], c=MinSF[0], s=10)
    for row in ii:
        a = np.repeat(row[0], n-1)
        b = np.asarray(row[1:])
        a = pcCoordinates[a]
        b = pcCoordinates[b]
        for i in range(n-1):
            ax.plot3D([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], zs=[a[i, 2], b[i, 2]], color='grey')

    # plt.colorbar(sc)
    plt.show()


if __name__ == '__main__':
    main()
