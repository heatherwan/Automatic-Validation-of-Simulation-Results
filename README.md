# Automatic-Validation-of-Simulation-Results

DataPreprocessing : 
- Import vtp file, find minSF point and sample n points from each object 
- create new features and export as hdf5 file (data, label) 
 
PointNet : Use x,y,z with additional features added
Additional features: Safety Factor, distance to minSF, (normals, curvatures)
Model:
- PointNet
- POintNet++
- DGCNN
- LDGCNN 

using Parameter.py to change models and parameters

PointGCN : Use PointGCN (Global pooling) with Safety Factor added  (remove)

