# **Vehicle Detection Project**


Here I will explain pipeline for SDC project "Vehicle Detection and Tracking". Code consists of two main objects `CarClassifier` and `CarSearcher` related to classification and search tasks. I'll explain in details what each class is responsible for.

## Classifier

Class `CarClassifier` is responsible for image part classification as car or not as car. It has feature extraction methods: `bin_spatial`, `color_hist`, `get_hog_features`, `extract_features`. Also, it has `fit` and `predict` methods for classifier training and utilization.

### Colorspace

I've experimented with few color-spaces for classifier training. Here is example of car detection on test images for different color-spaces:

* RGB

![RGB example][image_RGB_example]

* HSV

![HSV example][image_HSV_example]

* LUV

![LUV example][image_LUV_example]

* HLS

![HLS example][image_HLS_example]

* YUV

![YUV example][image_YUV_example]

* YCrCb

![YCrCb example][image_YCrCb_example]

This images show that `HSV` `LUV` `YCrCb` works batter than others. Later experiments on video showed `LUV` to be the best candidate.

### Classifier type

I've experimented with different type of classifiers but settled on LinearSVC as prediction time of other classifiers was dramatically bigger, even though prediction results was better.

### Augmentation

For better classification I've augmented data with flipping all images over vertical axis.


## Car search

This class is responsible for sliding window technique and applying classifier prediction method. Class creates three search region for different search window size and iterates through the region to predict car possibility in each window. Here I chose `0.7` windows overlap value.

![All windows][image_all_windows]


### Search over region

To decrease prediction time histogram of oriented gradients is calculated once for each region. Last image is stored in `CarClassifier` class and if next `predict` method will be called with same image but different region, HOG will not be calculated again.

Search results:

![Found windows][image_windows]

### Heatmap

To improve car tracking on video I use heatmap. All possible car locations on each frame gives one point to each pixel. Prediction heatmap is calculated as sum of 5 last heatmaps. Next, threshold of minimum value 3 is applied.

![Heatmap][image_heatmap]

It is clear, that areas related to cars has higher maximum heatmap value. But won't be good idea to increase threshold value as it will highlight only center of each car. This way for each label on the image I look at maximum heatmap value and label counts as car if only maximum heatmap value is higher than 20.

![Result][image_result]


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.

All code references in this document are for `pipeline.py` file.

### Writeup / README

#### The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

If your are reading this documents, that means I haven't forgotten to submit it.

### Histogram of Oriented Gradients (HOG)

#### Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why.

I have chosen values `orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)` as they worked well enough.

Color space selection is described above.

#### The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier.

I have chosen LinearSVC by the reasons described above. Features are scaled to zero mean and unit variance (line code 173-174).

### Sliding Window Search

#### A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen.

Sliding window approach is described above (line code 237-238).

#### Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)

Improved reliability is described above in `heatmap` section.

### Video Implementation

#### The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video.

Sliding window search is described above.

Here's a [link to my video result](./project_video_res.mp4)

#### A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur.

Heatmap method is described above.

### Discussion

#### Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.

Here is the list of possible issues of current algorithm implementation:

1. Algorithm is not real-time

1. Low distance of vehicle detection

1. Probably, it will work worse in night

1. Not possible to separate to close cars

Here are some ideas how this algorithm can be improved:

1. Use color filters to separate close cars

2. Use faster classifier, for example neural network



[//]: # (Image References)

[image_RGB_example]: ./writeup_img/exRGB.png "RGB example"
[image_HSV_example]: ./writeup_img/exHSV.png "HSV example"
[image_LUV_example]: ./writeup_img/exLUV.png "LUV example"
[image_HLS_example]: ./writeup_img/exHLS.png "HLS example"
[image_YUV_example]: ./writeup_img/exYUV.png "YUV example"
[image_YCrCb_example]: ./writeup_img/exYCrCb.png "YCrCb example"
[image_all_windows]: ./writeup_img/windows.png "All windows"
[image_windows]: ./writeup_img/windows1.png "Found windows"
[image_heatmap]: ./writeup_img/heatmap.png "Heatmap"
[image_result]: ./writeup_img/result.png "Result"
