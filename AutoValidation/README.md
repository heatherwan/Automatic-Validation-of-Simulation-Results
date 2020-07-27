## Automatic Validation of Simulation Results using Neural Network
This is the code repository accompanying the master thesis that was carried out with support from Festo AG & Co. KG.

### Introduction
This is a neural network framework for classification trustworthiness of point clouds acquired by FEM simulation from pneumatic cylinder cap.
More details can be found in the `document/presentation`.
   
### Installation 
dockerfile in `document/AutoValid_dockerfile` for tensorflow environment with required point cloud processing libraries.

### Usage
This sections shows how to use in training and in inference. 

Using `Parameters.py` to set up the hyperparameters and input dataset.

For example, 

-model  eg. pointnet_cls, pointnet2_cls dgcnn, ldgcnn, densepoint_tf)

-TRAIN_FILES to indicate training data

-TEST_FILES to indicate testing data

#### Training
train model  

    train_sf.py 
    
The trained log and model will save in `log` and `logmodel` with name of experiment accordingly.
    
#### Inferece
Then get prediction with pre-traind model
    
    evaluate.py  
or get average prediction with multiple angle rotation
     
    evaluate_multiangle.py 
    
The prediction result(evaluation score, logits and probability) will save in `evallog` with name of experiment accordingly.

### License
Our code is released under MIT License (see LICENSE file for details).

### Related Projects and Publications
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). 
* <a href="http://stanford.edu/~rqi/pointnet2/" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017) 
* <a href="https://arxiv.org/abs/1801.07829" target="_blank">Dynamic Graph CNN for Learning on Point Clouds</a> by Wang et al. (arXiv). 
* <a href="https://arxiv.org/abs/1904.10014" target="_blank">Linked Dynamic Graph CNN: Learning through Point Cloud by Linking Hierarchical Features</a> by Zhang et al. 
* <a href="https://arxiv.org/abs/1909.03669" target="_blank">DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing</a> by Liu et al. (ICCV 2019)