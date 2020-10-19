import os
import re
import zipfile
import numpy as np
import torch
import cv2
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
import utils.Solver as solver
from sklearn.model_selection import KFold



class Dataset(object):

    def __init__(self, normalize=True, normal=['normal', 'pneumonia'], n_splits=3, n_exp=1):

        self.data_path = os.path.join(os.getcwd(), 'resized_COVIDx')
        self.data_frame_path = os.path.join(self.data_path, 'no_split.txt')
        self.images_path = os.path.join(self.data_path, 'resized_COVIDx')

        print("\nInitializing Dataset...")

        solver.unzipdata()
        n_exp = n_exp - 1

        data_frame = pd.read_csv(self.data_frame_path, sep=" ", names=['code', 'filename', 'label', 'source'])

        self.normal_df = shuffle(data_frame.loc[data_frame['label'].isin(normal)], random_state=42)
        self.covid_df = shuffle(data_frame.loc[data_frame['label'].isin(['COVID-19'])], random_state=42)

        kf_normal = KFold(n_splits=n_splits)

        self.trains = []
        self.tests_normals = []
        self.tests_covid = []

        for train_index, test_index in kf_normal.split(self.normal_df):
            self.trains.append(train_index)
            self.tests_normals.append(test_index)

        for train_index, test_index in kf_normal.split(self.covid_df):
            self.tests_covid.append(test_index)

        normal_df_train, normal_df_test = self.normal_df.iloc[self.trains[n_exp]], self.normal_df.iloc[
            self.tests_normals[n_exp]]
        covid_df_test = self.covid_df.iloc[self.tests_covid[n_exp]]

        # train_df, test_df = train_test_split(normal_df, test_size=0.33)

        self.train_df = normal_df_train
        self.test_df = shuffle(pd.concat([covid_df_test, normal_df_test]), random_state=42)
        print("Running experiment number {} out of {} ----  Train Cases: {} / Test Cases: {} Out of which {} are COVID"
              .format(n_exp + 1, n_splits, self.train_df.shape[0], self.test_df.shape[0], covid_df_test.shape[0]))

        self.normalize = normalize

        self.num_train, self.num_test = self.train_df.shape[0], self.test_df.shape[0]
        self.idx_train, self.idx_test = 0, 0

        sample_image = cv2.imread(os.path.join(self.images_path, self.train_df.iloc[0]['filename']))

        self.height = sample_image.shape[0]
        self.width = sample_image.shape[1]

        self.channel = 1

        self.num_class = len(normal) + 1
        self.min_val, self.max_val = sample_image.min(), sample_image.max()

        print("Information of data")
        print("Number of Training Cases: {}".format(self.num_train))
        print("Number of Test Cases: {}, COVID-19 Cases: {}".format(self.num_test, covid_df_test.shape[0]))
        print("Shape  Height: %d, Width: %d, Channel: %d" % (self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" % (self.min_val, self.max_val))
        print("Class  %d" % (self.num_class))
        print("Normalization: %r" % (self.normalize))
        if (self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" % (self.min_val, self.max_val, 0, 1))

    def reset_idx(self):
        self.idx_train, self.idx_test = 0, 0

    def rgb2gray(self, img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return img_gray

    def next_train(self, batch_size=1):

        start, end = self.idx_train, self.idx_train + batch_size

        terminator = False
        if (end >= self.num_train):
            terminator = True
            self.train_df = shuffle(self.train_df)
            start = 0
            end = batch_size

        self.idx_train = end

        train_images = np.zeros((batch_size, self.height, self.width))
        train_labels = np.zeros((batch_size), dtype=int)

        for i in range(start, end):
            img_path = self.train_df.iloc[i]['filename']
            img_path = os.path.join(self.images_path, img_path)
            train_images[i - start] = self.rgb2gray(cv2.imread(img_path))
            train_labels[i - start] = 0

        train_images = np.ndarray.astype(train_images, np.float32)
        train_images = np.expand_dims(train_images, axis=3)

        if (self.normalize):
            min_x, max_x = train_images.min(), train_images.max()
            train_images = (train_images - min_x) / (max_x - min_x)

        train_images_torch = torch.from_numpy(np.transpose(train_images, (0, 3, 1, 2)))
        train_labels_torch = torch.from_numpy(train_labels)

        return train_images, train_images_torch, train_labels, train_labels_torch, terminator

    def next_test(self, batch_size=1):

        start, end = self.idx_test, self.idx_test + batch_size

        terminator = False
        if (end >= self.num_test):
            terminator = True
            self.test_df = shuffle(self.test_df)
            start = 0
            end = batch_size

        self.idx_test = end

        test_images = np.zeros((batch_size, self.height, self.width))
        test_labels = np.zeros((batch_size), dtype=int)
        for i in range(start, end):
            img_path = self.test_df.iloc[i]['filename']
            img_path = os.path.join(self.images_path, img_path)
            test_images[i - start] = self.rgb2gray(cv2.imread(img_path))

            label = self.test_df.iloc[i]['label']
            test_labels[i - start] = 0
            if (label == 'COVID-19'):
                # abnormal is 1
                test_labels[i - start] = 1

        test_images = np.ndarray.astype(test_images, np.float32)
        test_images = np.expand_dims(test_images, axis=3)

        if (self.normalize):
            min_x, max_x = test_images.min(), test_images.max()
            test_images = (test_images - min_x) / (max_x - min_x)

        test_images_torch = torch.from_numpy(np.transpose(test_images, (0, 3, 1, 2)))
        test_labels_torch = torch.from_numpy(test_labels)

        return test_images, test_images_torch, test_labels, test_labels_torch, terminator
