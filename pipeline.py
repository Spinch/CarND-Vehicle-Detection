
import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.preprocessing
import skimage.feature

import sklearn.svm
import sklearn.externals
import scipy.ndimage.measurements
from moviepy.editor import VideoFileClip

import cProfile, pstats, io


class CarClassifier(object):

    def __init__(self):
        self.dataShape = None

        self.bin_spatial_size = (32,32)

        self.color_hist_nbins = 32
        self.color_hist_binsRange = (0,256)

        self.hog_orient = 9
        self.hog_pix_per_cell = (8, 8)
        self.hog_cell_per_block = (2, 2)
        self.hog_feature_vec = False

        self.cspace = 'RGB'

        self.prevImg = None
        self.features = None

        pass

    def bin_spatial(self, img):
        # create the feature vector
        features = cv2.resize(img, self.bin_spatial_size).ravel()
        return features

    def color_hist(self, img):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.color_hist_nbins, range=self.color_hist_binsRange)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.color_hist_nbins, range=self.color_hist_binsRange)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.color_hist_nbins, range=self.color_hist_binsRange)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features

    def get_hog_features(self, img, region=None, vis=False):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = skimage.feature.hog(img, orientations=self.hog_orient, pixels_per_cell=self.hog_pix_per_cell,
                                      cells_per_block=self.hog_cell_per_block, block_norm='L2-Hys',
                                      transform_sqrt=True, visualise=vis, feature_vector=self.hog_feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            if img is not self.prevImg:
                self.prevImg = img
                self.features = []
                # for channel in range(img.shape[2]):
                for channel in [0,1,2]:
                    ch_features = skimage.feature.hog(img[:,:,channel], orientations=self.hog_orient, pixels_per_cell=self.hog_pix_per_cell,
                           cells_per_block=self.hog_cell_per_block, block_norm='L2-Hys',
                           transform_sqrt=True, visualise=vis, feature_vector=self.hog_feature_vec)
                    self.features.append(ch_features)
                self.features = np.array(self.features)
                if region is not None:
                    # print(region)
                    region[0] = int(region[0] / self.hog_pix_per_cell[0])
                    region[2] = int(region[2] / self.hog_pix_per_cell[1])
                    region[1] = region[0] + (64 // self.hog_pix_per_cell[0]) - self.hog_cell_per_block[0] + 1
                    region[3] = region[2] + (64 // self.hog_pix_per_cell[1]) - self.hog_cell_per_block[1] + 1
                    features = self.features[:,region[0]:region[1],region[2]:region[3]]
                    # features = np.hstack((features[0].ravel(), features[1].ravel(), features[2].ravel()))
                    features = np.hstack([features[i].ravel() for i in range(len(features))])
                else:
                    # features = np.hstack((self.features[0].ravel(), self.features[1].ravel(), self.features[2].ravel()))
                    features = np.hstack([self.features[i].ravel() for i in range(len(self.features))])
            else:
                region[0] = int(region[0] / self.hog_pix_per_cell[0])
                region[2] = int(region[2] / self.hog_pix_per_cell[1])
                region[1] = region[0] + (64 // self.hog_pix_per_cell[0]) - self.hog_cell_per_block[0] + 1
                region[3] = region[2] + (64 // self.hog_pix_per_cell[1]) - self.hog_cell_per_block[1] + 1
                features = self.features[:, region[0]:region[1], region[2]:region[3]]
                # features = np.hstack((features[0].ravel(), features[1].ravel(), features[2].ravel()))
                features = np.hstack([features[i].ravel() for i in range(len(features))])
            return features

    def extract_features(self, data, region=None):
        if not isinstance(data, list):
            data = [data]
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for feature_image in data:

            if self.dataShape is None:
                self.dataShape = feature_image.shape[:2]

            if region is None:
                feature_image_r = feature_image
            else:
                feature_image_r = feature_image[region[0]:region[1], region[2]:region[3]]

            # Apply bin_spatial() to get spatial color features
            spatial_features = self.bin_spatial(feature_image_r)

            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist(feature_image_r)

            # if hog_channel == 'ALL':
            # hog_features = []
            # for channel in range(feature_image.shape[2]):
            #     hog_features.append(self.get_hog_features(feature_image[:, :, channel], vis=False))
            # hog_features = np.ravel(hog_features)
            # if region is not None:
            #     region = [x/64 for x in region]
            hog_features = self.get_hog_features(feature_image, region, vis=False)
            # else:
            # hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
            #                                     pix_per_cell, cell_per_block, vis=False, feature_vec=True)

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))
            # features.append(np.concatenate((hist_features, spatial_features)))

        return features

    def fit(self, cars, notcars, Verbose=False):

        car_imgs = []
        for c in cars:
            image = cv2.imread(c)
            if self.cspace != 'BGR':
                conv_image = cv2.cvtColor(image, eval('cv2.COLOR_BGR2' + self.cspace))
            else:
                conv_image = image
            car_imgs.append(conv_image)
            car_imgs.append(cv2.flip(conv_image, 1))

        noncar_imgs = []
        for nc in notcars:
            image = cv2.imread(nc)
            if self.cspace != 'BGR':
                conv_image = cv2.cvtColor(image, eval('cv2.COLOR_BGR2' + self.cspace))
            else:
                conv_image = image
            noncar_imgs.append(conv_image)
            noncar_imgs.append(cv2.flip(conv_image, 1))

        car_features = self.extract_features(car_imgs)
        notcar_features = self.extract_features(noncar_imgs)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

        # Fit a per-column scaler only on the training data
        self.X_scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        # Apply the scaler to X_train and X_test
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        if Verbose:
            print('Feature vector length:', len(X_train[0]))

        # Use SVC
        # self.svc = sklearn.svm.SVC()
        self.svc = sklearn.svm.LinearSVC()

        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()

        if Verbose:
            print(round(t2 - t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t = time.time()
            n_predict = 1000
            self.svc.predict(X_test[0:n_predict])
            t2 = time.time()
            print(t2 - t, 'Seconds to predict', n_predict, 'labels with SVC')


    def predict(self, img, region=None):
        img_features = self.extract_features(img, region)
        res = self.svc.predict(self.X_scaler.transform(img_features))
        return res


class CarSearcher(object):

    def __init__(self):
        self.heatMap = None
        self.heatMaps = []
        pass

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def searchRegion(self, img, classifier, scale, xy_overlap, y_start_stop, x_start_stop=[None, None]):
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]

        img_tosearch = img[y_start_stop[0]:y_start_stop[1],x_start_stop[0]:x_start_stop[1],:]

        windows = []

        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        for yi in range(0, img_tosearch.shape[0]-64, int(64*(1-xy_overlap[1]))):
            for xi in range(0, img_tosearch.shape[1]-64, int(64 * (1-xy_overlap[0]))):
                if classifier.predict(img_tosearch, [yi,yi+64, xi, xi+64]):
                # if True:
                    tl = (int(x_start_stop[0]+xi*scale), int(y_start_stop[0]+yi*scale))
                    br = (int(x_start_stop[0]+(xi+64)*scale), int(y_start_stop[0]+(yi+64)*scale))
                    windows.append((tl, br))
        return windows


    def readData(self, path):
        data = []
        data_dir_tree = os.walk(path)
        for d in data_dir_tree:
            for f in d[2]:
                if f[-4:] == '.png':
                    path = os.path.join(d[0], f)
                    data.append(path)
        return data

    def drawLabeles(self, img, threshold, color):
        heatMapAv = np.sum(self.heatMaps, axis=0)
        heatmap = np.copy(heatMapAv)
        heatmap[heatMapAv <= threshold] = 0
        labels = scipy.ndimage.measurements.label(heatmap)

        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bbox_max = np.max(heatMapAv[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]])
            # print(bbox_max)
            # hs = np.histogram(img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0], 1], bins=10, range=(0,255))
            # print(hs)
            # Draw the box on the image
            # if (bbox_max > 20):
            # if (bbox_max > 200):
            #     cv2.rectangle(img, bbox[0], bbox[1], color, 6)
            cv2.rectangle(img, bbox[0], bbox[1], color, 6)
            # Return the image
        return img


    def do(self, img):

        # if self.heatMap is None:
        #     self.heatMap = np.zeros_like(img[:,:,0], dtype='int')

        ol = 0.7
        saveModeFile = 'trainedModel.sav'
        carClassifier = sklearn.externals.joblib.load(saveModeFile)

        if carClassifier.cspace != 'RGB':
            img0 = cv2.cvtColor(img, eval('cv2.COLOR_RGB2'+carClassifier.cspace))
        else:
            img0 = img

        window_list1 = self.searchRegion(img0, carClassifier, 1, xy_overlap=(ol, ol), y_start_stop=[350, 500])
        window_list2 = self.searchRegion(img0, carClassifier, 1.5, xy_overlap=(ol, ol), y_start_stop=[400, 550])
        window_list3 = self.searchRegion(img0, carClassifier, 2, xy_overlap=(ol, ol), y_start_stop=[400, 600])

        img1 = self.draw_boxes(img, window_list1, color=(0, 0, 255))
        img1 = self.draw_boxes(img1, window_list2, color=(0, 255, 0))
        img1 = self.draw_boxes(img1, window_list3, color=(255, 0, 0))
        #
        # window_list = window_list1+window_list2+window_list3
        # windows_wcars = search_windows(img0, window_list, carClassifier)
        # img2 = draw_boxes(img0, windows_wcars, color=(255, 0, 0))

        # self.heatMap = np.copy(img)
        # for w in window_list1:
        #     self.heatMap[w[0][1]:w[1][1],w[0][0]:w[1][0],:] = 255

        heatMap = np.zeros_like(img[:, :, 0], dtype='int')
        for w in window_list1:
            heatMap[w[0][1]:w[1][1],w[0][0]:w[1][0]] += 1
        heatMap[heatMap > 255] = 255
        self.heatMaps.append(np.copy(heatMap))
        if len(self.heatMaps) > 5:
            self.heatMaps.pop(0)

        img2 = np.copy(img)
        # self.drawLabeles(img2, 1, (0,0,255))
        self.drawLabeles(img2, 3, (0,255,0))
        # self.drawLabeles(img2, 5, (255,0,0))

        # return heatMap
        # return img1
        return img2

if __name__ == '__main__':

    cars_dir = './data/vehicles'
    notcars_dir = './data/non-vehicles'
    saveModeFile = 'trainedModel.sav'

    sc = CarSearcher()

    cars = sc.readData(cars_dir)
    notcars = sc.readData(notcars_dir)

    # carClassifier = CarClassifier()
    # carClassifier.cspace = 'LUV'
    # 'RGB' 'HSV' 'LUV' 'HLS' 'YUV' 'YCrCb':
    # 'HSV' 'LUV' 'YCrCb'

    # carClassifier.fit(cars, notcars, Verbose=True)
    # sklearn.externals.joblib.dump(carClassifier, saveModeFile)

    # imgPath = './test_images/test6.jpg'
    # img0 = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

    # pr = cProfile.Profile()
    # pr.enable()
    # img1 = sc.do(img0)
    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumtime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    # cProfile.run('img1 = sc.do(img0)')
    # img1 = do(img0)

    # plt.figure()
    # plt.imshow(img1)
    # plt.show()

    # imList = [os.path.join('./test_images', f) for f in os.listdir('./test_images')]
    # f, ax = plt.subplots(2, 3)
    # ax = np.hstack(ax)
    # for im,i in zip(imList, range(len(imList))):
    #     img0 = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
    #     img = sc.do(img0)
    #     sc.heatMaps = []
    #     ax[i].imshow(img)
    # plt.show()




    # video_in_name = './test_video.mp4'
    video_in_name = './project_video.mp4'
    sp = os.path.splitext(video_in_name)
    video_out_name = sp[0] + '_res' + sp[1]

    video_in = VideoFileClip(video_in_name)
    video_out = video_in.fl_image(sc.do)
    # video_out = video_in.subclip(0,10).fl_image(sc.do)
    video_out.write_videofile(video_out_name, audio=False)
