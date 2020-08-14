#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:39:59 2020
@author: wantinglin
"""
import os
import sys
import time

import h5py
import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd
import pyvista as pv
import trimesh
import vtk
from scipy import spatial
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy

import CalCurvature as CC

# from compare_result import compare, output_wrong
matplotlib.use('tkagg')
matplotlib.matplotlib_fname()

np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

resultDir = 'result_folder'
classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}
# classes = {1: 'EM1_contact', 2: 'EM2_inhole', 3: 'EM3_radius', 4: 'EM4_hole', 0: 'Good'}


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


def normalize_coordinate_pos(coordinate, distance, normals):
    mean_coor = np.mean(coordinate[:, 1:], axis=0)
    translate_coor = coordinate[:, 1:] - mean_coor
    max_dist = np.max(np.linalg.norm(translate_coor))
    normal_coor = translate_coor / max_dist

    max_SF = np.max(coordinate[:, 0])
    min_SF = np.min(coordinate[:, 0])
    normal_SF = (coordinate[:, 0] - min_SF) / (max_SF - min_SF)
    # rescale SF
    normal_SF = np.power(normal_SF, 1 / 3)

    normal_nearpoint = np.concatenate((normal_SF.reshape(-1, 1), normal_coor), axis=1)

    max_distance = np.max(distance)
    min_distance = np.min(distance)
    normal_distance = (distance - min_distance) / (max_distance - min_distance)

    if normals is not None:
        max_normals = np.max(normals)
        min_normals = np.min(normals)
        normals = (normals - min_normals) / (max_normals - min_normals)

    return normal_nearpoint, normal_distance, normals


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    # for fix 2 directions
    nodes = np.absolute(nodes)
    node = np.absolute(node)
    deltas = nodes - node
    # print(nodes[:5])
    # print(deltas[:5])
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    # print(dist_2[:5])
    return dist_2


def readfilestoh5_normal(export_filename, train_test_folder, NNeighbors=1024, dim=5):
    export_filename = f'{export_filename}_{NNeighbors}.hdf5'

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

                # get normal_difference
                if dim == 6:
                    normal_difference = get_normals(data)[indexes]
                    normals_distance = closest_node(normal_difference[0], normal_difference)
                else:
                    normals_distance = None

                # min-max normalize
                normalized_nearpoints, normalized_distance, normal_difference = normalize_coordinate_pos(nearpoints,
                                                                                                         distance,
                                                                                                         normals_distance)
                # # check in plot
                # plotter = pv.Plotter(shape=(2, 2))
                # plotter.subplot(0, 0)
                # point_cloudall = pv.PolyData(nearpoints[:, 1:])
                # plotter.add_mesh(point_cloudall, stitle='selected neighborhood', point_size=5., render_points_as_spheres=True)
                # minpoint = pv.PolyData(MinSF[1:])
                # plotter.add_mesh(minpoint, point_size=8, render_points_as_spheres=True)
                #
                # plotter.subplot(0, 1)
                # point_cloud = pv.PolyData(nearpoints[:, 1:])
                # plotter.add_mesh(point_cloud, scalars=nearpoints[:, 0], stitle='safety factor',
                #                  point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
                # minpoint = pv.PolyData(MinSF[1:])
                # plotter.add_mesh(minpoint, scalars=MinSF[0], point_size=8, render_points_as_spheres=True)
                #
                # plotter.subplot(1, 0)
                # point_cloud = pv.PolyData(nearpoints[:, 1:])
                # plotter.add_mesh(point_cloud, scalars=normalized_distance, stitle='distance',
                #                  point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
                # minpoint = pv.PolyData(MinSF[1:])
                # plotter.add_mesh(minpoint, scalars=MinSF[0], point_size=8, render_points_as_spheres=True)
                #
                # plotter.subplot(1, 1)
                # point_cloud = pv.PolyData(nearpoints[:, 1:])
                # plotter.add_mesh(point_cloud, scalars=normal_difference, stitle='normal_difference distance',
                #                  point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
                # minpoint = pv.PolyData(MinSF[1:])
                # plotter.add_mesh(minpoint, scalars=normal_difference[0], point_size=8, render_points_as_spheres=True)
                #
                # plotter.link_views()
                # plotter.show(title=f'{i}_{filename[19:]}')

                # get gaussian and mean curvature
                if dim == 7:
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
                if dim == 4:
                    other = normalized_nearpoints
                elif dim == 5:
                    other = np.concatenate((normalized_nearpoints,
                                            normalized_distance.reshape(-1, 1)), axis=1)
                elif dim == 6:
                    other = np.concatenate((normalized_nearpoints,
                                            normalized_distance.reshape(-1, 1),
                                            normal_difference.reshape(-1, 1)), axis=1)
                elif dim == 7:
                    other = np.concatenate((normalized_nearpoints,
                                            normalized_distance.reshape(-1, 1),
                                            k_curvature.reshape(-1, 1),
                                            m_curvature.reshape(-1, 1)), axis=1)
                all_data.append(other)
            print(f'total find time = {time.time() - now}')

        data = f.create_dataset("data", data=all_data)
        label = f.create_dataset("label", data=all_label)


def get_wrong_filename(filenamepath, wrongpath):
    filename = pd.read_csv(filenamepath, sep='\t', index_col=0)
    wrong = pd.read_csv(wrongpath, sep='\t', index_col=0)

    wrong['filename'] = filename.loc[wrong.index, :]
    wrong.to_csv(f"{wrongpath.replace('.txt', '')}_result_all.txt", sep='\t', float_format='%1.3f')
    return wrong


def get_allfilename(datafolder, type='Train'):
    alldata = []
    for cls in os.listdir(f'{datafolder}/{type}'):
        for data in os.listdir(os.path.join(f'{datafolder}/{type}', cls)):
            alldata.append(os.path.join(f'{datafolder}/{type}', cls, data))

    fout = open(os.path.join(resultDir, f'{datafolder}_{type}.txt'), 'w')
    fout.write(f'no\tfilepath\n')
    for i, train in enumerate(alldata):
        fout.write(f'{i}\t{train}\n')


def get_min_SF_graph(knn, ls, filename, NNeighbors=1024, last=False):
    for i, file in enumerate(filename.filepath):
        if i in ls:
            poly, data = readVTP(file)
            MinSF = get_MinSF(data)

            _, _, nearpoints2000 = get_Nearpoints(data, MinSF, NNeighbors * 2)
            nearpoints2000 = nearpoints2000[:, 1:]

            distance, indexes, nearpoints1000 = get_Nearpoints(data, MinSF, NNeighbors)

            normals = get_normals(data)[indexes]
            normals_distance = closest_node(normals[0], normals)
            # min-max normalize
            normal_nearpoints, normal_distance, normals = normalize_coordinate_pos(nearpoints1000, distance,
                                                                                   normals_distance)

            plotter = pv.BackgroundPlotter(shape=(2, 3))

            for j in range(len(knn) + 2):
                plotter.subplot(j // 3, j % 3)

                point_cloudall = pv.PolyData(nearpoints2000)
                plotter.add_mesh(point_cloudall, color='white',
                                 point_size=2., render_points_as_spheres=True)
                if j == 0:
                    point_cloud = pv.PolyData(nearpoints1000[:, 1:])
                    plotter.add_mesh(point_cloud, scalars=normal_nearpoints[:, 0], stitle='safety factor',
                                     point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
                elif j == 1:
                    point_cloud = pv.PolyData(nearpoints1000[:, 1:])
                    plotter.add_mesh(point_cloud, scalars=normal_distance, stitle='distance',
                                     point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
                else:
                    point_cloud = pv.PolyData(nearpoints1000[:, 1:])
                    plotter.add_mesh(point_cloud, scalars=knn[j - 2][i], stitle=f'knn{j - 1}',
                                     point_size=5., render_points_as_spheres=True, interpolate_before_map=True)

                minpoint = pv.PolyData(MinSF[1:])
                plotter.add_mesh(minpoint, scalars=MinSF[0], point_size=8, render_points_as_spheres=True)
                # plotter.show_grid()

            plotter.link_views()
            plotter.show()
    if last:
        plotter.app.exec_()


def get_min_SF_graph_compare(ls, knn1, filename, NNeighbors=1024, type='distance', last=False):
    # print(filename)

    plotter = pv.BackgroundPlotter(shape=(2, 4))

    for idx, file_id in enumerate(ls):
        if file_id > 0:
            file = filename.filepath[file_id]
            poly, data = readVTP(file)
            MinSF = get_MinSF(data)
            _, _, nearpoints2000 = get_Nearpoints(data, MinSF, NNeighbors * 2)
            nearpoints2000 = nearpoints2000[:, 1:]

            distance, indexes, nearpoints1000 = get_Nearpoints(data, MinSF, NNeighbors)

            plotter.subplot(idx // 4, idx % 4)
            plotter.add_point_labels(MinSF[1:], ['minSF'], font_size=12, point_color="red", text_color="red")

            plotter.add_text(str(file_id), font_size=12)

            point_cloudall = pv.PolyData(nearpoints2000)
            plotter.add_mesh(point_cloudall, color='white',
                             point_size=2., render_points_as_spheres=True)
            if type == 'SF':
                point_cloud = pv.PolyData(nearpoints1000[:, 1:4])
                plotter.add_mesh(point_cloud, scalars=nearpoints1000[:, 0], stitle='safety factor',
                                 point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
            elif type == 'distance':
                point_cloud = pv.PolyData(nearpoints1000[:, 1:4])
                plotter.add_mesh(point_cloud, scalars=distance, stitle='distance',
                                 point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
            elif type == 'knn1':
                point_cloud = pv.PolyData(nearpoints1000[:, 1:4])
                plotter.add_mesh(point_cloud, scalars=knn1[file_id], stitle=f'knn1',
                                 point_size=5., render_points_as_spheres=True, interpolate_before_map=True)
    # plotter.link_views()
    plotter.show()
    if last:
        plotter.app.exec_()


def visualize_color_by_array(curvature, mesh):
    # Plot mean curvature
    vect_col_map = \
        trimesh.visual.color.interpolate(curvature, color_map='bwr')
    if curvature.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif curvature.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
    mesh.show(background=[0, 0, 0, 255])


def main():
    # ============== generate figure for thesis =============================================================
    # filename = 'Sixth_new_data/EM1_contact/' \
    #            '10000154_FF_P8120737_Rev5_DFPD10_Deckel_Fv_Max.odb__P8120737_REV5_DFPD10_DECKEL-1.vtu.vtp'
    # NNeighbors = 1024
    # now = time.time()
    # poly, data = readVTP(filename)
    # time1 = time.time()
    # MinSF = get_MinSF(data)
    # time2 = time.time()
    # distance, indexes, nearpoints = get_Nearpoints(data, MinSF, NNeighbors)
    # time3 = time.time()
    # normals = get_normals(data)[indexes]
    # time4 = time.time()
    # normals_distance = closest_node(normals[0], normals)
    # time5 = time.time()
    #
    # # min-max normalize
    # normalized_nearpoints, normalized_distance, normal_difference = normalize_coordinate_pos(nearpoints,
    #                                                                                          distance,
    #                                                                                          normals_distance)
    # time6 = time.time()
    # neighborpolys, connected_points = get_neighborpolys(indexes, poly)
    # mesh = trimesh.Trimesh(vertices=nearpoints[:, 1:], faces=neighborpolys)
    # k_curvature, m_curvature = get_curvatures(mesh)
    # # sometimes the selected point is not connected to mesh
    # if NNeighbors > len(connected_points):
    #     print(f'not connected {NNeighbors - len(connected_points)}')
    #     no_curvature = set(range(1024)) - connected_points
    #     for idx in no_curvature:
    #         k_curvature = np.insert(k_curvature, idx, 0)
    #         m_curvature = np.insert(m_curvature, idx, 0)
    # # visualize_color_by_array(m_curvature, mesh)
    # # visualize_color_by_array(k_curvature, mesh)
    # time7 = time.time()
    #
    # print(f"time read: {time1-now}")
    # print(f"time MinSF: {time2 - time1}")
    # print(f"time subsampling: {time3 - time2}")
    # print(f"time normal: {time4 - time3}")
    # print(f"time normdiff: {time5 - time4}")
    # print(f"time normalize: {time6 - time5}")
    # print(f"time curvatures: {time7 - time6}")

    # # check in plot
    # plotter = pv.Plotter()

    # point_cloudall = pv.PolyData(data[:, 1:])
    # plotter.add_mesh(point_cloudall, color='white', point_size=3., render_points_as_spheres=True)
    # point_cloudall = pv.PolyData(nearpoints[:, 1:])
    # plotter.add_mesh(point_cloudall, color='blue', point_size=5., render_points_as_spheres=True)

    # point_cloud = pv.PolyData(nearpoints[:, 1:])
    # plotter.add_mesh(point_cloud, scalars=normal_difference, stitle='Normal to MinSF',
    #                  point_size=5., render_points_as_spheres=True, interpolate_before_map=True)

    # minpoint = pv.PolyData(MinSF[1:])
    # plotter.add_mesh(minpoint, point_size=8, color='red', render_points_as_spheres=True)

    # plotter.show(title=f'{filename[19:]}')
    # =======================================================================================

    # =============generate h5df dataset=========================================
    # export_filename = f"outputdataset/traindataset_651_SF"
    # readfilestoh5_normal(export_filename, 'Fourth_new_data/Train', NNeighbors=2048, dim=8)
    #
    # export_filename = f"outputdataset/testdataset_163_SF"
    # readfilestoh5_normal(export_filename, 'Fourth_new_data/Test', NNeighbors=2048, dim=8)
    #
    # export_filename = f"outputdataset/traindataset_814_SF"
    # readfilestoh5_normal(export_filename, 'Fifth_new_data/Trainall', NNeighbors=2048, dim=8)
    #
    # export_filename = f"outputdataset/testdataset_71_SF"
    # readfilestoh5_normal(export_filename, 'Fifth_new_data/Test_onlynew', NNeighbors=2048, dim=8)
    #
    # export_filename = f"outputdataset/testdataset_88_SF"
    # readfilestoh5_normal(export_filename, 'Sixth_new_data/UNKNOWN', NNeighbors=2048, dim=8)

    # export_filename = f"outputdataset/testdataset_geo_c4"
    # readfilestoh5_normal(export_filename, 'Final_data/Test', NNeighbors=1024, dim=7)
    # export_filename = f"outputdataset/traindataset_SF_c4"
    # readfilestoh5_normal(export_filename, 'Final_data/Train', NNeighbors=512, dim=6)
    # export_filename = f"outputdataset/validdataset_SF_c4"
    # readfilestoh5_normal(export_filename, 'Final_data/Validation', NNeighbors=512, dim=6)

    # export_filename = f"outputdataset/testdataset_SF_c5"
    # readfilestoh5_normal(export_filename, 'Final_data/Test', NNeighbors=2048, dim=6)
    # export_filename = f"outputdataset/traindataset_SF_c5"
    # readfilestoh5_normal(export_filename, 'Final_data/Train', NNeighbors=2048, dim=6)
    # export_filename = f"outputdataset/validdataset_SF_c5"
    # readfilestoh5_normal(export_filename, 'Final_data/Validation', NNeighbors=2048, dim=6)
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
    sample_num = 149
    expname = '141'
    filename = pd.read_csv('result_folder/Final_data_validation.txt', sep='\t')
    knn1 = np.fromfile(f'result_folder/exp{expname}_knn1.txt', sep=" ").reshape(sample_num, 1024)
    knn2 = np.fromfile(f'result_folder/exp{expname}_knn2.txt', sep=" ").reshape(sample_num, 1024)
    knn3 = np.fromfile(f'result_folder/exp{expname}_knn3.txt', sep=" ").reshape(sample_num, 1024)
    knn4 = np.fromfile(f'result_folder/exp{expname}_knn4.txt', sep=" ").reshape(sample_num, 1024)
    listall = [33]
    # get_min_SF_graph([knn1, knn2, knn3, knn4], listall, filename, NNeighbors=1024)

    expname = '142'
    filename = pd.read_csv('result_folder/Final_data_validation.txt', sep='\t')
    knn1 = np.fromfile(f'result_folder/exp{expname}_knn1.txt', sep=" ").reshape(sample_num, 1024)
    knn2 = np.fromfile(f'result_folder/exp{expname}_knn2.txt', sep=" ").reshape(sample_num, 1024)
    knn3 = np.fromfile(f'result_folder/exp{expname}_knn3.txt', sep=" ").reshape(sample_num, 1024)
    knn4 = np.fromfile(f'result_folder/exp{expname}_knn4.txt', sep=" ").reshape(sample_num, 1024)
    get_min_SF_graph([knn1, knn2, knn3, knn4], listall, filename, NNeighbors=1024)

    expname = '143'
    filename = pd.read_csv('result_folder/Final_data_validation.txt', sep='\t')
    knn1 = np.fromfile(f'result_folder/exp{expname}_knn1.txt', sep=" ").reshape(sample_num, 1024)
    knn2 = np.fromfile(f'result_folder/exp{expname}_knn2.txt', sep=" ").reshape(sample_num, 1024)
    knn3 = np.fromfile(f'result_folder/exp{expname}_knn3.txt', sep=" ").reshape(sample_num, 1024)
    knn4 = np.fromfile(f'result_folder/exp{expname}_knn4.txt', sep=" ").reshape(sample_num, 1024)
    get_min_SF_graph([knn1, knn2, knn3, knn4], listall, filename, NNeighbors=1024, last=True)

    # ==========get visualization of models graph comparison========================================
    # list_low1 = [138, 139, 140, 141, 142, 143, 144, 145]
    # list_low3 = [530, 531, 532, 533, 534, 535, 536, 537]
    # list_lowg = [846, 847, 848, 849, 850, 851, 852, 853]
    # get_min_SF_graph_compare(list_low1, knn4, filename, type='knn1', NNeighbors=2048)
    # get_min_SF_graph_compare(list_low3, knn4, filename, type='knn1', NNeighbors=2048)
    # get_min_SF_graph_compare(list_lowg, knn4, filename, type='knn1', last=True, NNeighbors=2048)

    # =======================================================================================
    # datafolder = 'Final_data'
    # get_allfilename(datafolder, 'validation')
    # get_allfilename(datafolder, 'Test_all')
    # =======================================================================================
    # h5_filename = 'outputdataset/testdataset_163_1024_dim6_normal.hdf5'
    # f = h5py.File(h5_filename)
    # data = f["data"][:]
    # label = f["label"][:]
    #
    # for i, l in enumerate(label):
    #     print(f'data {i} label: {l}')
    #     print(f'SF: {data[i,0:2,0]}')
    #     print(f'xyz: {data[i,0:2, 1:4]}')
    #     print(f'dist: {data[i, 0:2,4]}')
    #     print(f'MinSF: {data[i,0:2,5]}')


if __name__ == '__main__':
    main()
