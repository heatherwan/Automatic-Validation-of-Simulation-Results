import math
import os
import sys
import time

import h5py
import numpy as np
import open3d as o3d
import pyvista as pv
import vtk
from scipy import spatial
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
import pymesh
# from compare_result import compare, output_wrong

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

resultDir = 'result_folder'
NNeighbors = 1024
classes = {1: 'EM1_contact', 2: 'EM3_radius', 3: 'EM4_hole', 0: 'Good'}
# classes = {1: 'EM13_contactradius', 2: 'EM4_hole', 0: 'Good'}


def readVTP(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()
    mesh = vtk_to_numpy(poly.GetPolys().GetData())
    mesh = mesh.reshape(mesh.shape[0]//4, 4)[:, 1:]

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
    print(polyNew)
    polyNew.GetPointData().SetScalars(VTK_data_SF)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.Update()
    writer.SetInputData(polyNew.VTKObject)
    writer.Update()
    writer.Write()


def getMinSF(data):
    idx = np.argmin(data[:, 0])
    return data[idx, :]


def getNearpoints(data, MinSF, NNeighbors):
    coord = data[:, 1:]
    dd, indexes = spatial.cKDTree(coord).query(MinSF[1:], NNeighbors)
    nearestpoints = data[indexes, :]
    # print(nearestpoints.shape)
    return indexes, nearestpoints


def getfiles(folder, category):
    datafold = os.path.join(folder, category)
    files = []
    for f in os.listdir(datafold):
        files.append(os.path.join(datafold, f))
    # print(files)
    return files


def readfilestoh5(export_filename, train_test_folder):
    with h5py.File(export_filename, "w") as f:
        all_data = []
        all_label = []
        all_other = []
        for num, cat in classes.items():
            print(cat)
            now = time.time()
            files = getfiles(train_test_folder, cat)
            all_label.extend([num] * len(files))
            print(len(files))
            for i, filename in enumerate(files):
                poly, data = readVTP(filename)
                MinSF = getMinSF(data)
                indexes, nearpoints = getNearpoints(data, MinSF, NNeighbors)
                # # get distance
                distance = get_dist(MinSF, nearpoints)
                # # get normals
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(data[:, 1:])
                # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=4))
                # o3d.visualization.draw_geometries([pcd])
                # normals = np.asarray(pcd.normals)
                # get normals and gaussain curvature
                gaussian, normals = get_gaussian_curvature_normal(poly, data, indexes)
                all_data.append(nearpoints[:, 1:])
                # # concat all other features
                other = np.concatenate((nearpoints[:, 0].reshape(NNeighbors, 1),
                                        distance.reshape(NNeighbors, 1),
                                        normals[indexes, :],
                                        gaussian), axis=1)
                all_other.append(other)

            print(f'total find time = {time.time() - now}')
        data = f.create_dataset("data", data=all_data)
        other = f.create_dataset("other", data=all_other)
        label = f.create_dataset("label", data=all_label)


def compare_plot(filename, df, single=True, name_a=None, name_b=None):
    poly, data = readVTP(filename)
    MinSF = getMinSF(data)
    points = data[:, 1:]
    _, nearpoints500 = getNearpoints(data, MinSF, 512)
    nearpoints500 = nearpoints500[:, 1:]
    _, nearpoints1000 = getNearpoints(data, MinSF, 1024)
    nearpoints1000 = nearpoints1000[:, 1:]
    _, nearpoints2000 = getNearpoints(data, MinSF, 2048)
    nearpoints2000 = nearpoints2000[:, 1:]

    # Make PolyData
    points = pv.PolyData(nearpoints2000)
    minpoint = pv.PolyData(MinSF[1:])
    point_cloud1 = pv.PolyData(nearpoints1000)
    point_cloud2 = pv.PolyData(nearpoints500)
    plotter = pv.Plotter()

    plotter.add_mesh(points, color='white', point_size=2., render_points_as_spheres=True)
    plotter.add_mesh(point_cloud1, color='blue', point_size=5., render_points_as_spheres=True)
    plotter.add_mesh(point_cloud2, color='maroon', point_size=5., render_points_as_spheres=True)
    plotter.add_mesh(minpoint, color='yellow', point_size=8, render_points_as_spheres=True)
    if single:
        plotter.add_text(f"no{df['  no']}, predict:{classes[df.pred]}, "
                         f"label:{classes[df.real]}")
    else:
        plotter.add_text(f"no{df.iloc[0]}, {name_a} predict:{classes[df.pred]}, "
                         f"{name_b} predict:{classes[df.pred2]}, "
                         f"label:{classes[df.real]}")
    plotter.show_grid()
    plotter.show(auto_close=True)


def dist(p1, p2):
    distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return distance


def get_dist(MinSF, nearpoints):
    distance = []
    for point in nearpoints[:, 1:]:
        distance.append(dist(point, MinSF[1:]))
    return np.array(distance)


def get_gaussian_curvature_normal(mesh_connect, data, indexes):
    mesh = pymesh.form_mesh(data[:, 1:], mesh_connect)
    # print(mesh.num_vertices, mesh.num_faces, mesh.num_voxels)
    mesh.add_attribute("vertex_normal")
    normal = mesh.get_attribute("vertex_normal")
    normal = normal.reshape(normal.shape[0] // 3, 3)
    # print(normal.shape)
    mesh.add_attribute("vertex_gaussian_curvature")
    gaussian = mesh.get_attribute("vertex_gaussian_curvature")
    mesh.add_attribute("vertex_index")
    index = mesh.get_attribute("vertex_index")

    # print(f'index :\n {index[indexes]}')
    # print(f'gaussian :\n {gaussian[indexes]}')
    # print(f'normals :\n {normal[indexes, :]}')

    return gaussian[indexes], normal[indexes, :]


def main():
    # =============generate h5df dataset=============================
    export_filename = f"outputdataset/traindataset_dim9_480_{NNeighbors}.hdf5"
    readfilestoh5(export_filename, 'train')

    export_filename = f"outputdataset/testdataset_dim9_160_{NNeighbors}.hdf5"
    readfilestoh5(export_filename, 'test')

    # =============visualize result with prediction=============================
    # 1. single
    # file = 'pred_label_dgcnn.txt'
    # data = output_wrong(file)
    #
    # for index, row in data.iterrows():
    #     compare_plot(row.file, row, single=True)

    # 2. comparison
    # file1 = 'pred_label_1024.txt'
    # file2 = 'pred_label_GCN1024.txt'
    # outname = 'Pointnet_GCN'
    # data = compare(file1, file2, outname)
    # for index, row in data.iterrows():
    #     compare_plot(row.file, row, single=False, name_a='PointNet', name_b='PointGCN')


if __name__ == '__main__':
    main()
