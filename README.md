[//]: # (Image References)

[img1]: ./output_images/example_images.png "example_images.png"
[img2]: ./output_images/example_images2.png "example_images2.png"
[img3]: ./output_images/hog_car1.png "hog_car1.png"
[img4]: ./output_images/hog_car2.png "hog_car2.png"
[img4a]: ./output_images/hog_feat_car.png "hog_feat_car.png"
[img5]: ./output_images/hog_notcar1.png "hog_notcar1.png"
[img6]: ./output_images/hog_notcar2.png "hog_notcar2.png"
[img6a]: ./output_images/hog_feat_notcar.png "hog_feat_notcar.png"
[img7]: ./output_images/spatial_car.png "spatial_car.png"
[img8]: ./output_images/spatial_notcar.png "spatial_notcar.png"
[img9]: ./output_images/colorbins_car.png "colorbins_car.png"
[img10]: ./output_images/colorbins_notcar.png "colorbins_notcar.png"
[img11]: ./output_images/feats_car.png "feats_car.png"
[img12]: ./output_images/feats_notcar.png "feats_notcar.png"
[img13]: ./output_images/scaled_feats.png "scaled_feats"
[img14]: ./output_images/PCA_reduced.png "PCA_reduced.png"
[img14b]: ./output_images/convergence.png "convergence.png"
[img15]: ./output_images/multiscale_search.png "multiscale_search.png.png"
[img16]: ./output_images/pipeline_vehicle_detection.png "PCA_reduced.png"

[img17]: ./output_images/Yolo0.png "Yolo0"
[img18]: ./output_images/yolo.gif "Yolo gif"

<!-- [img18]: ./output_images/Yolo1.png "Yolo1"
[img19]: ./output_images/Yolo2.png "Yolo2"
[img20]: ./output_images/Yolo3.png "Yolo3"
[img21]: ./output_images/Yolo4.png "Yolo4"
[img22]: ./output_images/Yolo5.png "Yolo5" -->

---
# SDCND Term 1 Project 5: Vehicle Detection
## With 2 different methods: Classing computer vision using HOG-features & SVM and Using Deep Neural Networks (YOLOv2) enabling real time
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Goal:**
Detect vehicles in a video recorded on a highway.

**Chosen methods:**  
  1. Classic approach with tools from classic computer vision and machine learning (mainly [HOG](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) & [Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html))
  2. State of the art Deep Convolution Neural Network [YOLOv2](https://pjreddie.com/darknet/yolo/)

**Results (Youtube Links):**
classic approach:

[![Results using classic approach](https://img.youtube.com/vi/yzNtLwBol6k/0.jpg)](https://www.youtube.com/watch?v=yzNtLwBol6k)

YOLOv2:

[![Results using YOLOv2](https://img.youtube.com/vi/p1IDWgE1RhI/0.jpg)](https://www.youtube.com/watch?v=p1IDWgE1RhI)

and

[![Results using YOLOv2](https://img.youtube.com/vi/6v--_ULavII/0.jpg)](https://www.youtube.com/watch?v=6v--_ULavII)

In the end both methods were able to detect the vehicles in front of the car, especially the Deep Learning method produec realiably results. The **Deep Learning** approach enables almost **real time** detection and tracking with around **15-20 fps** on the used machine (1060 6GB GPU). This document describes my approach and implemented pipeline. The code can be found in the corresponding Jupyter Notebooks for the [classic approach](SDCND-P5_Vehicle-Detection.ipynb) and [YOLOv2](SDCND-P5_Vehicle-Detection_YOLO.ipynb).

---
## Detailed explanation of the pipeline of the classic approach
The steps to achieve these results are the following (you can look up the code from each step under the same chapter number in the [notebook](SDCND-P5_Vehicle-Detection.ipynb):

1. **Dataset Summary & Exploration**
2. **Feature Extraction**
    2.1 HOG features
    2.2 Spatial binned color features
    2.3 Color histogram features
    2.4 Complete Feature Extraction pipeline
    2.5 Pipeline on dataset
    2.6 Normalize features
    2.7 Linear dimensionality reduction with PCA
3. **Training**
    3.1 Definition of SVM-classifier
    3.2 Parameter search
    3.3 Final Parameter&Training
4. **Vehicle detection pipeline**
    4.1 Multi scale sliding window search for cars
    4.2 Heatmap
5. **Video processing**


#### 1. Dataset Summary & Exploration
For the training of the classifier a dataset of images is provided of both cars and "not-cars".

| Class      | Count  | Shape   |
|------------|--------|---------|
| Car        | 8792   | 64x64x3 |
| Not-car    | 8968   | 64x64x3 |

**Info images:**
* each image is 12288 pixel (64x64x3)
* every pixel could used as a feature, but training the classifier with that would be higly expensive
* Instead: HOG, color and spatial features


Example image (making sure **png and jpg** are loaded):
![img1]


#### 2 Feature Extraction
##### 2.1 HOG features
**Histogram of Oriented Gradients (HOG):** HOG subsamples the image to a grid of e.g. 8x8 or 4x4 pixel grids, with each only containing the most prominent gradient in that part of the image. It therefore extracts gradients in the image while dropping other information like color.

With a color-image you can use all color channels for the HOG-feature. As described later, the LUV-color space was performing best. So below you see the HOG of the 1., 2. and 3. color channel of one car and one not car LUV image:

RGB:
![img2]

LUV & 3-channel-HOG:
![img3]
![img4]

![img5]
![img6]

In vector-form this looks something like this:
![img4a]
![img6a]

##### 2.2 Spatial binned color features
Spatial binned color features are extracted by downscaling the image and taking the pixel as feature vector. In the downscaling process we reduce the number of pixel significantly but maintain the rough color information of the image. This process is defined in ``get_hog_features``. Example feature vector:

**Car**
![img7]
**Not Car**
![img8]

##### 2.3 Color histogram features
Cars often a dominant color in an image, to capture that ``numpy.histogram`` is used to get this feature. Example feature vector

**Car**
![img9]
**Not Car**
![img10]

##### 2.4 Complete Feature Extraction pipeline
Now we can combine all features to one big feature vector. Its length depends on the parameter we chose for the feature extraction methods above. Example:

**Car**
![img11]
**Not Car**
![img12]

##### 2.5 Pipeline on dataset
This process can be used to vectorize the whole data set as defined in ``pipeline_dataset``.

##### 2.6 Normalize features
We have to scale/normalize the feature vector to zero mean and unit variance before training the classifier, because numerically the feature vectors are very different from each other. This was done using ``sklearn.preprocessing.StandardScaler``. After scaling the feature vector looks something like this:

![img13]

After normalization the dataset is split into train and test set with a ratio 0.8/0.2 using ``sklearn.model_selection.train_test_split``.

After this step the training and test data were saved as a pickle file for later use.

##### 2.7 Linear dimensionality reduction with PCA

To reduce the number of dimension ``sklearn.decomposition.PCA`` was used.

PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance. In scikit-learn, PCA is implemented as a transformer object that learns n components in its fit method, and can be used on new data to project it on these components.

Simply put, it allowed to shrink the feature vector by a factor of 10-20 with only a minor loss in accuracy, see example below. Because the prediction time is highly dependent on the number of features, now the more expensive rbf kernel for the Support Vector Machine kernel can be used.

![img14]

#### 3. **Training**

##### 3.1 Definition of SVM-classifier
The extracted features where fed to SVM with a radial basis function kernel (rbf) of sklearn with default setting of square-hinged loss function and l2 normalization.

The trained model along with the parameters used for training were written to a pickle file to be further used by vehicle detection pipeline.

##### 3.2 Parameter search
Since this project involves a lot of parameters, I decided to implement an automatic parameter search. These parameters could be tuned:

* ``color_space`` - used color space
* ``orient`` - no of possible HOG orientations
* ``pix_per_cell`` - HOG feature extraction parameter
* ``spatial_size`` - feature extraction parameter
* ``hist_bins`` - feature extraction parameter
* ``reduc_factor`` - Reduction for PCA
* ``C`` - SVM parameter
* ``gamma`` - SVM parameter
* ``kernel`` - SVM parameter

These parameters were manually set as the showed the best performance from the beginning:
* ``cell_per_block = 2``- HOG feature extraction parameter
* ``hog_channel = "ALL"``- HOG feature extraction parameter

2 methods were implemented for the automatic parameter search:

* **1.** Creating and using a csv-file for a grid-search which can be created by [Create_search_space.ipynb](Create_search_space.ipynb). Results with accuracy and speed

* **2.** Scikit-Optimize: Bayesian optimization using Gaussian Processes [**``skopt.gp_minimize``**](https://scikit-optimize.github.io/)
with a customized cost function, to get a good compromise between accuracy and speed:

```python
cost =  25*(1 - accuracy) + time_getting_features/100 + time_prediction_100_samples
```

![img14b]

##### 3.3 Final Parameter & Training
Finally these following parameters were chosen, because they performed best:

| Color Space | HOG (orient, pix per cell, cell per block, hog channel)         | Color   | Histogram | Classifier (kernel, C, gamma) | reduction factor PCA  | Accuracy | Prediction time for 100 samples |
|-------------|---------------|---------|-----------|------------|-------|----------|---------------------------------|
| LUV         | 19, 16, 2, 'ALL' | (16,16) | 16       | rbf, C=10, 0.005 | n=20 | 0.99718  | 0.051499s                       |


#### 4. Vehicle detection pipeline
The detection pipeline scans over an image and uses the trained classifier, normalizer and PCA  to detect vehicle in a frame of video.

The pipeline consists of the following steps:

* Calculate HOG feature over entire image and subsample with three different window sizes
* Extract color spatial and histogram features as well
* Normalize and apply PCA to combined feature vector
* Use classifier to predict if there is a car in the subsample
* False positive rejection via heatmap and history
* Annotate found vehicle in input image by drawing a bounding box


##### 4.1 Sliding window search for cars
The function ``find_cars`` from the Udacity lesson material was adopted to allow a multi-scale-search functionality. It applies a trick to the extraction of HOG features. Instead of adjusting the size of the sliding window and then rescaling the subsample back to 64x64 pixel to calculate HOG features, the function is resizing the entire image and calculating HOG features over the entire image. So instead of a 96x96 sliding window that we would have to rescale to 64x64 to calculate HOG features, we scale the entire image down by a factor of 1.5 and subsample with a window size of 64x64.

The image below shows the used region of interest and 3 different scales, which were the input to the function  ``multi_scale_search``, which uses the above mentioned function ``find_cars`` .

| scale      | ystart | ystop   |
|------------|--------|---------|
| 1       | 390   | 540 |
| 1.5   | 390   | 615 |
| 2.5   | 390   | 666 |

![img15]

##### 4.2 Heatmap (False positive rejection)
Next a algorithm/class was implemented to keep a history of a variable amount of frames to only draw bounding boxes when the detection was validated over a few frames. This was done to reject false positives in the video stream. Therefore the concept of heatmap and thresholding was chosen. Also a blur filter was applied to smoothen the results. See class ``Heatmap`` for details.

A result of this algorithm is shown below:

![img16]
Bottom left: red is history, yellow is validated
#### 5. Video processing
For the video each frame was processed according to the above mentioned pipeline.

---

## Detailed explanation of the pipeline of the Deep Learning approach (YOLOv2)
In this approach [*darkflow*](https://github.com/thtrieu/darkflow) (an tensorflow implementation of [darknet](https://github.com/pjreddie/darknet) & YOLOv2) was used. Setting up the YOLO network was fairly easy, the chosen network configuration was pretrained on the VOC-dataset(which contains a car-class) so a weights file could be loaded. This means that the network didn't need any training at all. These options were chosen:

```python
# define the model options and run
options = {
    'model': 'cfg/yolo-voc.cfg',
    'load': 'bin/yolo-voc.weights',
    'threshold': 0.40,
    'gpu': 0.7,
}
```

Which gives this network:

```python
Parsing ./cfg/yolo-voc.cfg
Parsing cfg/yolo-voc.cfg
Loading bin/yolo-voc.weights ...
Successfully identified 202704260 bytes
Finished in 0.03650522232055664s
Model has a VOC model name, loading VOC labels.

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 416, 416, 3)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 416, 416, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 208, 208, 32)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 208, 208, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 104, 104, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 128)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 104, 104, 64)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 52, 52, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 256)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 52, 52, 128)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | concat [16]                      | (?, 26, 26, 512)
 Load  |  Yep!  | conv 1x1p0_1  +bnorm  leaky      | (?, 26, 26, 64)
 Load  |  Yep!  | local flatten 2x2                | (?, 13, 13, 256)
 Load  |  Yep!  | concat [27, 24]                  | (?, 13, 13, 1280)
 Load  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Load  |  Yep!  | conv 1x1p0_1    linear           | (?, 13, 13, 125)
-------+--------+----------------------------------+---------------
GPU mode with 0.7 usage
Finished in 32.91464304924011s
```
The most complicated part was to implement a method to smoothen the detections and delete false positives. Therefore the class ``car_boxes`` was implemented in the [notebook](). This class takes in the box-list as a result from YOLO, compares these box coordinates to all box coordinates in the recorded history (with variable ``history_depth``) and if the current box coordinates are in a relative tolerance (``rtol``) of one of the history boxes it is validated. These technique allows the *track* an object. After a variable amount of validated history records(``threshold``) the box coordinates are considered as a car-box and a moving averaring algorithm is used to smoothen these coordinates.

The algorithm can follow a variable amount of objects (``max_no_boxes = 5``)

A region of interest (ROI) was also implemented to focus YOLO on the important parts of the video.
![img17]

Below you see the pipeline in action with a threshold of 3 (after frame 3, the results are getting validated as car-boxes):

**Left is the direct output from yolo  |  Right is the validated and smoothed result**
![img18]


---
## Discussion
As one can see the Deep Learning approach was a lot easier to set up, less time-consuming and performed better & faster. Of course it is possible to improve the classic classifier approach by additional data augmentation, negative mining, classifier parameters tuning etc., but it will may fail in case of difficult light conditions and still does not perform in real-time.

The classic approach also has some problems in case of car overlaps another. To resolve this problem one may introduce long term memory of car position and a kind of predictive algorithm which can predict where occluded car can be and where it is worth to look for it.

To even optimize the Deep Learning approach even further, one could train the network on the [Udacity Annotated Driving Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations), but the performance is quite astonishing as it is.

It is a bit unfair to compare the both approaches regarding the real-time capability, because YOLOv2 uses GPU, which has more computational power than the classic approach & CPU. But I would still prefer the Deep Learning approach, because there is no manual fine-tuning of every feature extraction method and training parameter. It just works!

This was a very tedious project which involved the tuning of several parameters by hand. With the traditional Computer Vision approach I was able to develop a strong intuition for what worked and why. I also learnt that the solutions developed through such approaches aren't very optimized and can be sensitive to the chosen parameters. As a result, I developed a strong appreciation for Deep Learning based approaches to Computer Vision. Although, they can appear as a black box at times Deep learning approaches avoid the need for fine-tuning these parameters, and are inherently more robust.
