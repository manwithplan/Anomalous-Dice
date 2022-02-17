import cv2
import os
import random
import numpy as np
import time
import pickle


class preprocessor:
    def __init__(self) -> None:
        # hyper-parameters
        self.ROOT_FOLDER = "..\.."
        self.DATA_FOLDER = os.path.join(self.ROOT_FOLDER, "data")
        self.TRAIN_FOLDER = os.path.join(self.DATA_FOLDER, "train_set")
        self.TEST_FOLDER = os.path.join(self.DATA_FOLDER, "test_set")
        self.PROCESSED_DATA = os.path.join(self.ROOT_FOLDER, "processed_data")
        self.TEMPLATE_FOLDER = os.path.join(self.PROCESSED_DATA, "templates")

    def rotate_image(self, image, angle):
        """
        rotate an image by a given angle

        :angle: angle in degrees
        :return: rotated image
        """

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )

        return result

    def count_pixels_by_side(self, image, threshold, debug_mode=False):
        """
        cutting an image in 2 and counting the pixels above a certain value on either side of the
        cut

        :image: image as a numpy array
        :debug_mode: flag that selects the return type
        :threshold: int value representing threshold for counting pixels
        :return: top and bottom image if debug mode on, top and bottom pixel count.
        """

        top_pixel_dark = 0
        bottom_pixel_dark = 0

        top_half = image[0 : int(image.shape[0] / 2), 0 : image.shape[1]]
        bottom_half = image[
            int(image.shape[0] / 2) : image.shape[0], 0 : image.shape[1]
        ]

        top_pixel_dark = (top_half < threshold).sum()
        bottom_pixel_dark = (bottom_half < threshold).sum()

        if debug_mode:
            return top_half, bottom_half, top_pixel_dark, bottom_pixel_dark

        return None, None, bottom_pixel_dark, top_pixel_dark

    def split_image(self, image, debug_mode=False):
        """
        cutting an image in 2

        :image: image as a numpy array
        :debug_mode: flag that selects the return type
        :return: left and right image
        """

        left_half = image[0 : image.shape[0], 0 : int(image.shape[1] / 2)]
        right_half = image[0 : image.shape[0], int(image.shape[1] / 2) : image.shape[1]]

        return left_half, right_half

    def find_diagonal(self, image, threshold, debug_mode=False):
        """
        finds the diagonal that contains the most valus above a threshold and returns
        the count of values above the threshold

        :image: image as a numpy array
        :debug_mode: flag that selects the return type
        :threshold: int value representing threshold for counting pixels
        :return: the count of pixels above threshold on the halfway line
        """

        halfway_line = int(image.shape[0] / 2)
        return (image[halfway_line] < threshold).sum()

    def rotate_one_dice(self, threshold, image_path):
        """
        method that applies rotation and processing to one dice

        :image: str path to image
        :threshold: value that is used for the threshold of the dark pixels
        :return: rotated and cropped dice
        """

        # read it in openCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # in order to crop a circle from the image, a mask is made with the same dimensions as the input
        height, width = img.shape
        mask = np.zeros((height, width), np.uint8)

        # When you apply a mask, the area that's cropped is colored black, which conflicts with our detection of
        # dark pixels, so we invert the image.
        img = cv2.bitwise_not(img)

        # we create a circular mask and apply it to the image
        circle_img = cv2.circle(mask, (64, 64), 60, (255, 255, 255), thickness=-1)
        masked_data = cv2.bitwise_and(img, img, mask=circle_img)

        # now we have te reinvert the image to get back the normal scale
        masked_data = cv2.bitwise_not(masked_data)
        img = cv2.bitwise_not(img)

        # array that holds information of each angle.
        count_per_diagonal = []

        # hardcode the right side of the image
        right = np.full((128, 64), 255)

        for angle in range(360):
            # image is rotated for each angle in a 360° circle
            rotated_img = self.rotate_image(masked_data, angle)

            # it splitted in to a left and a right side
            left, _ = self.split_image(rotated_img)
            half = np.hstack([left, right]).astype(np.uint8)

            # trying to build in a hierarchy towards one side, this should get all the faktion
            # logos to run on 1 side only.
            _, _, top, bottom = self.count_pixels_by_side(half, threshold=threshold)
            if top < bottom:
                count_per_diagonal.append(0)
            else:
                # we apply a function that counts the amount of dark pixels it crosses.
                diff_left = self.find_diagonal(
                    half, threshold=threshold, debug_mode=False
                )

                count_per_diagonal.append(diff_left)

        index = count_per_diagonal.index(max(count_per_diagonal))
        img = self.rotate_image(img, index)

        img = cv2.bitwise_not(img)
        # we create a circular mask and apply it to the image
        circle_img = cv2.circle(mask, (64, 64), 60, (255, 255, 255), thickness=-1)
        masked_data = cv2.bitwise_and(img, img, mask=circle_img)

        # now we have te reinvert the image to get back the normal scale
        masked_data = cv2.bitwise_not(masked_data)

        data = np.array(masked_data)
        preprocessed_data = data / 255
        preprocessed_data = np.expand_dims(preprocessed_data, axis=3)

        return preprocessed_data
