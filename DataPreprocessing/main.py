import vtk
from vtk.util.numpy_support import vtk_to_numpy
from outputselected import VtkPointCloud
from vtk.util import numpy_support
import numpy as np
from scipy import spatial
import time
import os
import h5py
import pyvista as pv
import matplotlib.pyplot as plt

NNeighbors = 1024
classes = {1: 'EM1', 2: 'EM3', 3: 'EM4', 0: 'Good'}


def readVTP(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()
    ID = vtk_to_numpy(poly.GetPolys().GetData())
    # print(ID)  # not useful include inside point
    Points = vtk_to_numpy(poly.GetPoints().GetData())
    SF = vtk_to_numpy(poly.GetPointData().GetScalars()).reshape(-1, 1)
    # print(f'Dimension points: {Points.shape}')
    # print(Points)
    # print(f'Dimension SF: {SF.shape}')
    # print(SF)
    data = np.concatenate((SF, Points), axis=1)
    return poly, data


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
    return nearestpoints


def visualiza_pointcloud(filename):
    poly, data = readVTP(filename)
    MinSF = getMinSF(data)
    nearpoints = getNearpoints(data, MinSF, NNeighbors)[:, 1:]
    pointCloud = VtkPointCloud()
    # data = np.genfromtxt(filename, dtype=float, usecols=[1, 2, 3])

    for k in range(np.size(nearpoints, 0)):
        point = nearpoints[k]
        pointCloud.addPoint(point)

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)

    renderer.SetBackground(.2, .3, .4)
    # renderer.SetBackground(200.0, 200.0, 200.0)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("XYZ Data Viewer" + filename)
    renderWindowInteractor.Start()


def getfiles(folder, category):
    current_fold = os.getcwd()
    datafold = os.path.join(current_fold, f'{folder}/{category}')
    files = []
    for f in os.listdir(datafold):
        files.append(os.path.join(datafold, f))
    # print(files)
    return files


def readfilestoh5(export_filename, train_test_folder):
    with h5py.File(export_filename, "w") as f:
        all_data = []
        all_label = []
        all_sf = []

        for num, cat in classes.items():
            print(cat)
            now = time.time()
            files = getfiles(train_test_folder, cat)
            all_label.extend([num] * len(files))
            print(len(files))
            for i, filename in enumerate(files):
                poly, data = readVTP(filename)
                MinSF = getMinSF(data)
                nearpoints = getNearpoints(data, MinSF, NNeighbors)
                # np.savetxt(f"extractfile/{os.path.basename(filename).replace('.vtu.vtp','.txt')}", nearpoints)
                all_data.append(nearpoints[:, 1:])
                all_sf.append(nearpoints[:, 0])
                # print(all_sf)
            print(f'total find time = {time.time() - now}')
        # print(all_label)
        data = f.create_dataset("data", data=all_data)
        sf = f.create_dataset("sf", data=all_sf)
        label = f.create_dataset("label", data=all_label)


def compare_plot(filename):
    poly, data = readVTP(filename)
    MinSF = getMinSF(data)
    points = data[:, 1:]
    nearpoints1000 = getNearpoints(data, MinSF, 1024)[:, 1:]
    nearpoints2000 = getNearpoints(data, MinSF, 2048)[:, 1:]

    # Make PolyData
    points = pv.PolyData(points)
    minpoint = pv.PolyData(MinSF[1:])
    point_cloud1 = pv.PolyData(nearpoints1000)
    point_cloud2 = pv.PolyData(nearpoints2000)
    plotter = pv.Plotter()

    plotter.add_mesh(points, color='white', point_size=2., render_points_as_spheres=True)
    plotter.add_mesh(point_cloud2, color='blue', point_size=5., render_points_as_spheres=True)
    plotter.add_mesh(point_cloud1, color='maroon', point_size=5., render_points_as_spheres=True)
    plotter.add_mesh(minpoint, color='yellow', point_size=8, render_points_as_spheres=True)

    plotter.show_grid()
    plotter.show(auto_close=True)


def main():

    export_filename = "traindataset_dim4_480.hdf5"
    readfilestoh5(export_filename, 'train')

    export_filename = "testdataset_dim4_160.hdf5"
    readfilestoh5(export_filename, 'test')

    # folder = os.path.join(os.getcwd(), 'visualize/good')
    # for file in os.listdir(folder):
    #     compare_plot(os.path.join(folder, file))


if __name__ == '__main__':
    main()
