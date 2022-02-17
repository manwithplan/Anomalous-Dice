import cv2
import os
import random
import numpy as np
import time
import pickle

# hyper-parameters
ROOT_FOLDER = "..\.."
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
TRAIN_FOLDER = os.path.join(DATA_FOLDER, "train_set")
TEST_FOLDER = os.path.join(DATA_FOLDER, "test_set")
PROCESSED_DATA = os.path.join(ROOT_FOLDER, "processed_data")
TEMPLATE_FOLDER = os.path.join(PROCESSED_DATA, "templates")

# functions
def rotate_image(image, angle):
    """
    rotate an image by a given angle

    :angle: angle in degrees
    :return: rotated image
    """

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def count_pixels_by_side(image, threshold, debug_mode=False):
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
    bottom_half = image[int(image.shape[0] / 2) : image.shape[0], 0 : image.shape[1]]

    top_pixel_dark = (top_half < threshold).sum()
    bottom_pixel_dark = (bottom_half < threshold).sum()

    if debug_mode:
        return top_half, bottom_half, top_pixel_dark, bottom_pixel_dark

    return None, None, bottom_pixel_dark, top_pixel_dark


def split_image(image, debug_mode=False):
    """
    cutting an image in 2

    :image: image as a numpy array
    :debug_mode: flag that selects the return type
    :return: left and right image
    """

    left_half = image[0 : image.shape[0], 0 : int(image.shape[1] / 2)]
    right_half = image[0 : image.shape[0], int(image.shape[1] / 2) : image.shape[1]]

    return left_half, right_half


def find_diagonal(image, threshold, debug_mode=False):
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


def make_template(category="00", categories=None, processed_data=True):
    """
    Function that takes all the images per category and makes a template out of them.

    :category: string name of the subfolder to make a template from
    :categories: list containing multiple string names of subfolder, None if not used
    :return: a 128x128 array that represents the mean values of all the images in each subfolder
    """

    if processed_data:
        data_folder = PROCESSED_DATA
    else:
        data_folder = DATA_FOLDER

    if categories:

        filepaths = []
        for x in categories:
            files = []
            # get the correct folder and filenames of each category
            train_data = os.path.join(data_folder, "train_set")
            folder = os.path.join(train_data, x)
            filenames = os.listdir(folder)
            files = [file for file in filenames if ".png" in file]

            for file in files:
                file_path = os.path.join(folder, file)
                filepaths.append(file_path)

    else:
        # get the correct folder and filenames
        train_data = os.path.join(data_folder, "train_set")
        folder = os.path.join(train_data, category)
        filenames = os.listdir(folder)
        files = [file for file in filenames if ".png" in file]

        # generate a list of each file's relative path
        filepaths = []
        for file in files:
            file_path = os.path.join(folder, file)
            filepaths.append(file_path)

    # read these images in one by one as a numpy array and average them.
    images = np.array(
        [np.array(cv2.imread(fname, cv2.IMREAD_GRAYSCALE)) for fname in filepaths]
    )
    arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)

    return arr


def rotate_one_per_class(threshold):
    """
    method that applies rotation to one random dice sample per category. Doesn't return anything but shows
    the image instead.

    :threshold: value that is used for the threshold of the dark pixels
    """
    # we are looping through each dice category
    all_types = os.listdir(TRAIN_FOLDER)
    for the_type in all_types[11:12]:

        # get the folder name, all filenames inside it, and make a list of all the image files inside
        folder = os.path.join(TRAIN_FOLDER, the_type)
        filenames = os.listdir(folder)
        files = [file for file in filenames if ".png" in file]

        # select a random file from the folder
        random_file = random.sample(files, 1)
        random_file_path = os.path.join(folder, random_file[0])

        # read it in openCV
        img = cv2.imread(random_file_path, cv2.IMREAD_GRAYSCALE)
        copy = img.copy()

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
            rotated_img = rotate_image(masked_data, angle)

            if angle % 10 == 0:
                example_img = rotated_img.copy()

                x1 = 0
                y1 = int(example_img.shape[1] / 2)

                x2 = example_img.shape[1]
                y2 = int(example_img.shape[1] / 2)

                line_thickness = 2
                cv2.line(
                    example_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    thickness=line_thickness,
                )

                cv2.imwrite(
                    "../../visuals/" + str(the_type) + "-" + str(angle) + ".png",
                    np.hstack([masked_data, example_img]),
                )

            # it splitted in to a left and a right side
            left, _ = split_image(rotated_img)
            half = np.hstack([left, right]).astype(np.uint8)

            # trying to build in a hierarchy towards one side, this should get all the faktion
            # logos to run on 1 side only.
            _, _, top, bottom = count_pixels_by_side(
                half, threshold=threshold, debug_mode=True
            )
            if top > bottom:
                count_per_diagonal.append(0)
            else:
                # we apply a function that counts the amount of dark pixels it crosses.
                diff_left = find_diagonal(half, threshold=threshold, debug_mode=False)

                count_per_diagonal.append(diff_left)

        index = count_per_diagonal.index(max(count_per_diagonal))
        img = rotate_image(img, index)

        img = cv2.bitwise_not(img)
        # we create a circular mask and apply it to the image
        circle_img = cv2.circle(mask, (64, 64), 60, (255, 255, 255), thickness=-1)
        masked_data = cv2.bitwise_and(img, img, mask=circle_img)

        # now we have te reinvert the image to get back the normal scale
        masked_data = cv2.bitwise_not(masked_data)
        example_masked_data = masked_data.copy()

        x1 = 0
        y1 = int(example_masked_data.shape[1] / 2)

        x2 = example_masked_data.shape[1]
        y2 = int(example_masked_data.shape[1] / 2)

        line_thickness = 2
        cv2.line(
            example_masked_data,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=line_thickness,
        )

        for x in range(20):

            cv2.imwrite(
                "../../visuals/"
                + str(the_type)
                + "-"
                + "cropped"
                + "-"
                + str(x)
                + ".png",
                np.hstack([masked_data, example_masked_data]),
            )

        # cv2.imshow("output", half)
        cv2.imshow("output", np.hstack([copy, masked_data]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def rotate_all_dice(threshold, train=True):
    """
    method that applies rotation to all the dice

    :threshold: value that is used for the threshold of the dark pixels
    """

    if train:
        source_folder = TRAIN_FOLDER
        dest_folder = "train_set"
    else:
        source_folder = TEST_FOLDER
        dest_folder = "test_set"

    # we are looping through each dice category
    all_types = os.listdir(source_folder)
    for the_type in all_types:

        # get the folder name, all filenames inside it, and make a list of all the image files inside
        folder = os.path.join(source_folder, the_type)
        filenames = os.listdir(folder)
        files = [file for file in filenames if ".png" in file]

        for file in files:

            file_path_read = os.path.join(folder, file)

            # read it in openCV
            img = cv2.imread(file_path_read, cv2.IMREAD_GRAYSCALE)
            copy = img.copy()

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
                rotated_img = rotate_image(masked_data, angle)

                # it splitted in to a left and a right side
                left, _ = split_image(rotated_img)
                half = np.hstack([left, right]).astype(np.uint8)

                # trying to build in a hierarchy towards one side, this should get all the faktion
                # logos to run on 1 side only.
                _, _, top, bottom = count_pixels_by_side(half, threshold=threshold)
                if top < bottom:
                    count_per_diagonal.append(0)
                else:
                    # we apply a function that counts the amount of dark pixels it crosses.
                    diff_left = find_diagonal(
                        half, threshold=threshold, debug_mode=False
                    )

                    count_per_diagonal.append(diff_left)

            index = count_per_diagonal.index(max(count_per_diagonal))
            img = rotate_image(img, index)

            img = cv2.bitwise_not(img)
            # we create a circular mask and apply it to the image
            circle_img = cv2.circle(mask, (64, 64), 60, (255, 255, 255), thickness=-1)
            masked_data = cv2.bitwise_and(img, img, mask=circle_img)

            # now we have te reinvert the image to get back the normal scale
            masked_data = cv2.bitwise_not(masked_data)

            processed_data = os.path.join(PROCESSED_DATA, dest_folder)
            folder_path = os.path.join(processed_data, the_type)
            file_path = os.path.join(folder_path, file)
            cv2.imwrite(file_path, masked_data)


def rotate_one_dice(threshold, image_path):
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
        rotated_img = rotate_image(masked_data, angle)

        # it splitted in to a left and a right side
        left, _ = split_image(rotated_img)
        half = np.hstack([left, right]).astype(np.uint8)

        # trying to build in a hierarchy towards one side, this should get all the faktion
        # logos to run on 1 side only.
        _, _, top, bottom = count_pixels_by_side(half, threshold=threshold)
        if top < bottom:
            count_per_diagonal.append(0)
        else:
            # we apply a function that counts the amount of dark pixels it crosses.
            diff_left = find_diagonal(half, threshold=threshold, debug_mode=False)

            count_per_diagonal.append(diff_left)

    index = count_per_diagonal.index(max(count_per_diagonal))
    img = rotate_image(img, index)

    img = cv2.bitwise_not(img)
    # we create a circular mask and apply it to the image
    circle_img = cv2.circle(mask, (64, 64), 60, (255, 255, 255), thickness=-1)
    masked_data = cv2.bitwise_and(img, img, mask=circle_img)

    # now we have te reinvert the image to get back the normal scale
    masked_data = cv2.bitwise_not(masked_data)

    # data = np.array(masked_data)
    # preprocessed_data = data/255
    # preprocessed_data = np.expand_dims(preprocessed_data, axis=3)

    return masked_data


# next step is to calculate the deviance from the templates


def match_template(category, image=None):
    """ """
    if not image:
        # get the folder name, all filenames inside it, and make a list of all the image files inside
        train_folder = os.path.join(PROCESSED_DATA, "train_set")
        folder = os.path.join(train_folder, category)
        filenames = os.listdir(folder)
        files = [file for file in filenames if ".png" in file]

        # select a random file from the folder
        random_file = random.sample(files, 1)
        random_file_path = os.path.join(folder, random_file[0])

        image = cv2.imread(random_file_path, cv2.IMREAD_GRAYSCALE)

    else:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # load in the appropriate template

    template = cv2.imread(
        os.path.join(TEMPLATE_FOLDER, str(int(category)) + ".png"), cv2.IMREAD_GRAYSCALE
    )
    diff = template - image
    errors = (diff > 220).sum()

    return errors, template, image
