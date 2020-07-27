# Automatic-Validation-of-Simulation-Results
1. DataPreprocessing_Visualize:
Tools for preprocess Point cloud data from vtp file and visualization  
- Import vtp file, find minSF point and sample n points from each object 
- create new features and export as hdf5 file (data, label) 
- visualize generated feature graph with 3D point cloud in interactive mode

2. AutoValidation
Contain different deep learning neural networks eg. PointNet, PointNet++, DGCNN/LDGCNN, DensePoint
tensorflow implementation with additional features added eg. Safety Factor, distance to minSF, (normals, curvatures)

3. Confidence_Calibration
post-processing calibration of neural networks output logits (confidence) 
- Temperature Scaling
- Matrix-Scaling
- Dirichlet Calibration
