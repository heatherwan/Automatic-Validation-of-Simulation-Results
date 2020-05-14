#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:39:59 2020
@author: wantinglin
"""
import math
import sys
import time

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import pandas as pd
import pyvista as pv
import trimesh
import vtk
from scipy import spatial
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import CalCurvature as CC

# from compare_result import compare, output_wrong
matplotlib.use('tkagg')
matplotlib.matplotlib_fname()

np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

resultDir = 'result_folder'
classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}


# classes = {1: 'EM13_contactradius', 2: 'EM4_hole', 0: 'Good'}


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


def writeVTP(data, filename, poly):
    VTK_data_point = numpy_support.numpy_to_vtk(num_array=data[:, 1:].ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    VTK_data_SF = numpy_support.numpy_to_vtk(num_array=data[:, 0].ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    # Add data set and write VTK file
    polyNew = poly
    polyNew.SetPoints = VTK_data_point
    polyNew.GetPointData().SetScalars(VTK_data_SF)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.Update()
    writer.SetInputData(polyNew.VTKObject)
    writer.Update()
    writer.Write()


def get_MinSF(data):
    idx = np.argmin(data[:, 0])
    return data[idx, :]


def get_Nearpoints(data, MinSF, NNeighbors):
    coord = data[:, 1:]
    dd, indexes = spatial.cKDTree(coord).query(MinSF[1:], NNeighbors)
    nearestpoints = data[indexes, :]
    # print(nearestpoints.shape)

    return dd, indexes, nearestpoints


def get_neighborpolys(indexes, polys):
    neighborpoly = []
    index = dict((y, x) for x, y in enumerate(indexes))

    # get the extracted faces
    for p in np.asarray(polys).flatten():
        try:
            neighborpoly.append(index[p])
        except KeyError:
            neighborpoly.append(np.nan)

    neighborpoly = np.asarray(neighborpoly)
    neighborpoly = neighborpoly.reshape(neighborpoly.shape[0] // 3, 3)
    mask = np.any(np.isnan(neighborpoly), axis=1)
    neighborpoly = neighborpoly[~mask]
    connected_points = set(neighborpoly.flatten())
    return neighborpoly, connected_points


def get_curvatures(mesh):
    # Estimate curvatures by Rusinkiewicz method
    # "Estimating Curvatures and Their Derivatives on Triangle Meshes."
    # Symposium on 3D Data Processing, Visualization, and Transmission, September 2004.

    PrincipalCurvatures = CC.GetCurvaturesAndDerivatives(mesh)
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    return gaussian_curv, mean_curv


def get_files(folder, category):
    datafold = os.path.join(folder, category)
    files = []
    for f in os.listdir(datafold):
        files.append(os.path.join(datafold, f))
    return files


def get_normals(data, visual=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, 1:])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=4))
    if visual:
        o3d.visualization.draw_geometries([pcd])
    return np.asarray(pcd.normals)


def readfilestoh5(export_filename, train_test_folder, NNeighbors=1024, dim=5, minSF_importance=10):
    """

    :param NNeighbors:
    :param export_filename: .hdf5 filename
    :param train_test_folder: .vtp files to read
    :param dim: features output: 5-distance, 6-normal, 7-curvature, 10 normal+curvature
    :return:
    """

    if dim == 6:
        export_filename = f'{export_filename}_{NNeighbors}_dim{dim}_SF{minSF_importance}.hdf5'
    else:
        export_filename = f'{export_filename}_{NNeighbors}_dim{dim}.hdf5'

    with h5py.File(export_filename, "w") as f:
        all_data = []
        all_label = []
        for num, cat in classes.items():
            print(cat)
            now = time.time()
            files = get_files(train_test_folder, cat)
            all_label.extend([num] * len(files))
            print(len(files))
            for i, filename in enumerate(files):
                # get points near MinSF point
                poly, data = readVTP(filename)
                MinSF = get_MinSF(data)
                distance, indexes, nearpoints = get_Nearpoints(data, MinSF, NNeighbors)

                # set minSF
                minP = np.zeros(NNeighbors)
                minP[0] = 10

                # get normals
                if dim == 8 or dim == 10:
                    normals = get_normals(data)[indexes]

                # get gaussian and mean curvature
                if dim == 7 or dim == 10:
                    neighborpolys, connected_points = get_neighborpolys(indexes, poly)
                    mesh = trimesh.Trimesh(vertices=nearpoints[:, 1:], faces=neighborpolys)
                    k_curvature, m_curvature = get_curvatures(mesh)
                    # sometimes the selected point is not connected to mesh
                    if NNeighbors > len(connected_points):
                        print(f'not connected {NNeighbors - len(connected_points)}')
                        no_curvature = set(range(1024)) - connected_points
                        for idx in no_curvature:
                            k_curvature = np.insert(k_curvature, idx, 0)
                            m_curvature = np.insert(m_curvature, idx, 0)

                # gather data into hdf5 format
                if dim == 5:
                    other = np.concatenate((nearpoints.reshape(NNeighbors, 4),
                                            distance.reshape(-1, 1)), axis=1)
                elif dim == 6:
                    other = np.concatenate((nearpoints.reshape(NNeighbors, 4),
                                            distance.reshape(-1, 1),
                                            minP.reshape(-1, 1)), axis=1)
                elif dim == 8:
                    other = np.concatenate((nearpoints.reshape(NNeighbors, 4),
                                            distance.reshape(-1, 1),
                                            normals), axis=1)
                elif dim == 7:
                    other = np.concatenate((nearpoints.reshape(NNeighbors, 4),
                                            distance.reshape(-1, 1),
                                            k_curvature.reshape(-1, 1),
                                            m_curvature.reshape(-1, 1)), axis=1)
                elif dim == 10:
                    other = np.concatenate((nearpoints.reshape(NNeighbors, 4),
                                            distance.reshape(-1, 1),
                                            normals,
                                            k_curvature.reshape(-1, 1),
                                            m_curvature.reshape(-1, 1)), axis=1)
                all_data.append(other)
            print(f'total find time = {time.time() - now}')
        data = f.create_dataset("data", data=all_data)
        label = f.create_dataset("label", data=all_label)


def visualize_selected_points(filename, NNeighbors=1024, normal=False, curvature=False):
    poly, data = readVTP(filename)
    MinSF = get_MinSF(data)

    _, _, nearpoints2000 = get_Nearpoints(data, MinSF, NNeighbors * 2)
    nearpoints2000 = nearpoints2000[:, 1:]

    _, indexes, nearpoints1000 = get_Nearpoints(data, MinSF, NNeighbors)

    if curvature:
        neighborpolys, connected_points = get_neighborpolys(indexes, poly)
        mesh = trimesh.Trimesh(vertices=nearpoints1000[:, 1:], faces=neighborpolys)
        k_curvature, m_curvature = get_curvatures(mesh)

        # sometimes the selected point is not connected to mesh
        if NNeighbors > len(connected_points):
            print(f'not connected {NNeighbors - len(connected_points)}')
            no_curvature = set(range(1024)) - connected_points
            for idx in no_curvature:
                k_curvature = np.insert(k_curvature, idx, 0)
                m_curvature = np.insert(m_curvature, idx, 0)

    # Make PolyData
    minpoint = pv.PolyData(MinSF[1:])
    point_cloud1 = pv.PolyData(nearpoints1000[:, 1:])
    point_cloud2 = pv.PolyData(nearpoints2000)
    plotter = pv.Plotter()
    # plotter.add_mesh(points, color='white', point_size=2., render_points_as_spheres=True)
    plotter.add_mesh(point_cloud2, color='white',
                     point_size=2., render_points_as_spheres=True)
    plotter.add_mesh(point_cloud1, scalars=nearpoints1000[:, 0], stitle='safety factor',
                     point_size=5., render_points_as_spheres=True)
    plotter.add_mesh(minpoint, color='white', point_size=8, render_points_as_spheres=True)
    plotter.show_grid()
    plotter.show()

    # visualize normal
    if normal:
        normals = get_normals(data, visual=True)[indexes]
    if curvature:
        visualize_color_by_array(m_curvature, mesh)
        visualize_color_by_array(k_curvature, mesh)


def visualize_color_by_array(curvature, mesh):
    # Plot mean curvature
    vect_col_map = \
        trimesh.visual.color.interpolate(curvature, color_map='bwr')
    if curvature.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif curvature.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
    mesh.show(background=[0, 0, 0, 255])


def get_wrong_filename(filenamepath, wrongpath):
    filename = pd.read_csv(filenamepath, sep='\t', index_col=0)
    wrong = pd.read_csv(wrongpath, sep='\t', index_col=0)

    wrong['filename'] = filename.loc[wrong.index, :]
    wrong.to_csv(f"{wrongpath.replace('.txt', '')}_result_all.txt", sep='\t', float_format='%1.3f')
    return wrong


def get_allfilename(datafolder, type='Train'):
    data = []
    for cls in os.listdir(f'{datafolder}/{type}'):
        for data in os.listdir(os.path.join(f'{datafolder}/{type}', cls)):
            data.append(os.path.join(f'{datafolder}/{type}', cls, data))

    fout = open(os.path.join(resultDir, f'{datafolder}_{type}.txt'), 'w')
    fout.write(f'no\tfilepath\n')
    for i, train in enumerate(data):
        fout.write(f'{i}\t{train}\n')


def visualize_graph(filename, NNeighbors=1024, predict=None):
    poly, data = readVTP(filename)
    MinSF = get_MinSF(data)

    _, indexes, nearpoints1000 = get_Nearpoints(data, MinSF, NNeighbors)
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
    ax.scatter3D(MinSF[1], MinSF[2], MinSF[3], c='yellow', s=10)
    for row in ii:
        a = np.repeat(row[0], n - 1)
        b = np.asarray(row[1:])
        a = pcCoordinates[a]
        b = pcCoordinates[b]
        for i in range(n - 1):
            ax.plot3D([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], zs=[a[i, 2], b[i, 2]], color='grey')
    # ax.text(0, 0, 0, f'{filename} predict:{predict}', color='red')
    ax.text2D(0, 1, f'{filename}\npredict:{predict}', transform=ax.transAxes)

    plt.colorbar(sc)
    plt.show()


def get_min_SF_graph(knn, ls, NNeighbors=1024):
    filename = pd.read_csv('result_folder/testdatafile_0705.txt', sep='\t')
    # print(filename)
    for i, file in enumerate(filename.filepath):
        if i in ls:
            poly, data = readVTP(file)
            MinSF = get_MinSF(data)

            _, _, nearpoints2000 = get_Nearpoints(data, MinSF, NNeighbors * 2)
            nearpoints2000 = nearpoints2000[:, 1:]

            distance, indexes, nearpoints1000 = get_Nearpoints(data, MinSF, NNeighbors)

            plotter = pv.Plotter(shape=(2, 3))

            for j in range(len(knn) + 2):
                plotter.subplot(j // 3, j % 3)

                point_cloudall = pv.PolyData(nearpoints2000)
                plotter.add_mesh(point_cloudall, color='white',
                                 point_size=2., render_points_as_spheres=True)
                if j == 0:
                    point_cloud = pv.PolyData(nearpoints1000[:, 1:])
                    plotter.add_mesh(point_cloud, scalars=nearpoints1000[:, 0], stitle='safety factor',
                                     point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
                elif j == 1:
                    point_cloud = pv.PolyData(nearpoints1000[:, 1:])
                    plotter.add_mesh(point_cloud, scalars=distance, stitle='distance',
                                     point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
                else:
                    point_cloud = pv.PolyData(nearpoints1000[:, 1:])
                    plotter.add_mesh(point_cloud, scalars=knn[j - 2][i], stitle=f'knn{j-1}',
                                     point_size=5., render_points_as_spheres=True, interpolate_before_map=True)

                minpoint = pv.PolyData(MinSF[1:])
                plotter.add_mesh(minpoint, scalars=MinSF[0], point_size=8, render_points_as_spheres=True)
                # plotter.show_grid()

            plotter.link_views()
            plotter.show(title=f'{i}_{file[19:]}')


def main():
    # =============generate h5df dataset=========================================
    export_filename = f"outputdataset/traindataset_651"
    readfilestoh5(export_filename, 'Fourth_new_data/Train', NNeighbors=1024, dim=6)

    export_filename = f"outputdataset/testdataset_163"
    readfilestoh5(export_filename, 'Fourth_new_data/Test', NNeighbors=1024, dim=6)
    # ===========================================================================

    # ============get visulaization with files in test_same folder===========================
    # filenamepath = 'result_folder/testdatafile_0705.txt'
    # wrongfilepath = 'result_folder/exp423_wrong_pred_prob.txt'
    # wrongfile_df = get_wrong_filename(filenamepath, wrongfilepath)
    #
    # for pred, file in zip(wrongfile_df.pred, wrongfile_df.filename):
    #     visualize_graph(file, predict=pred)
    # visualize_selected_points(file)

    # =======================================================================================

    # ============get visulaization in dgcnn graph============================
    # sample_num = 163
    # expname = '428'
    # knn1 = np.fromfile(f'result_folder/exp{expname}_knn1.txt', sep=" ").reshape(sample_num, 1024)
    # knn2 = np.fromfile(f'result_folder/exp{expname}_knn2.txt', sep=" ").reshape(sample_num, 1024)
    # knn3 = np.fromfile(f'result_folder/exp{expname}_knn3.txt', sep=" ").reshape(sample_num, 1024)
    # knn4 = np.fromfile(f'result_folder/exp{expname}_knn4.txt', sep=" ").reshape(sample_num, 1024)
    # # knn5 = np.fromfile(f'result_folder/exp{expname}_knn5.txt', sep=" ").reshape(sample_num, 1024)
    # list = [49, 50, 51, 96, 108, 111, 121]  # 1, 3, 4, 13, 18, 22, 31, 17,30,35,
    # list422 = [17, 28, 29, 44, 45, 48, 49, 55, 61, 97, 98, 131, 158, 159, 160]
    # list423 = [21, 44, 55, 57, 97, 98, 112]
    # list424 = [4, 17, 36, 43, 55, 57, 153, 158]
    # list425 = [21, 28, 45, 48, 61, 97, 112, 119, 120, 141]
    # list428 = [17, 55, 97, 98, 158, 159]
    # get_min_SF_graph([knn1, knn2, knn3, knn4], list428)

    # =======================================================================================
    datafolder = 'Fourth_new_data'
    get_allfilename(datafolder, 'Train')
    get_allfilename(datafolder, 'Test')


if __name__ == '__main__':
    main()
