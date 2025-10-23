"""
Purpose
-------
This script will wil generate a set of unified square sized crop of the images,
by cropping from image center, and generating the corespondent landmarks from
ANHIR data.

Contributors
------------
- TMS-Namespace
"""

import os
import sys

# add the parent directory to the PATH, so we can import files one level up
root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(root_path)

from source.framework.tools.pather import Pather
from source.framework.tools.patches_generator import random_array_patches

from PIL import Image
import numpy as np
import csv

IMAGES_ROOT_DIR = "data/ANHIR_challenge/images"
LANDMARK_ROOT_DIR = "data/ANHIR_challenge/landmarks"
TARGET_DIR = "data/ANHIR_challenge/processed_265"

# images will be cropped from the center at the beginning, to avoid empty parts of the images
TARGET_IMAGE_SIZE = (2000, 2000)  # max 2000x2000
# how many landmarks should be in cropped image to accept it
MIN_LANDMARKS_COUNT = 6  # 8 -> 67 images, 10 -> 53

GENERATE_LANDMARKS = False
AUGMENTATION = False

RANDOM_PATCHES_SIZE = (256, 256)
RANDOM_PATCHES_COUNT = 20  # 0 for no patch generation

# images that meant for validation by ANHIR, will be considered as test set, so we crop it also
TEST_RANDOM_PATCHES_SIZE = (256, 256)
TEST_RANDOM_PATCHES_COUNT = 3  # 0 for no patch generation

RANDOM_SEED = 42

ACCEPTED_IMAGES_MIN_STD = 10

Image.MAX_IMAGE_PIXELS = None  # to remove DecompressionBombWarning

target_pather = Pather(TARGET_DIR)
landmarks_pather = Pather(LANDMARK_ROOT_DIR)

valid_counter = 0
total_counter = 0

random_generator = np.random.default_rng(RANDOM_SEED)

# region Help functions


def crop_from_center(image_path: str):
    image_array = np.array(Image.open(image_path).convert("L"))
    center = (int(image_array.shape[0] / 2), int(image_array.shape[1] / 2))

    left_top_corner = (
        int(center[0] - TARGET_IMAGE_SIZE[0] / 2),
        int(center[1] - TARGET_IMAGE_SIZE[1] / 2),
    )
    right_bottom_corner = (
        int(center[0] + TARGET_IMAGE_SIZE[0] / 2),
        int(center[1] + TARGET_IMAGE_SIZE[1] / 2),
    )

    image_array = image_array[
        left_top_corner[0] : right_bottom_corner[0],
        left_top_corner[1] : right_bottom_corner[1],
    ]

    if (
        image_array.shape[0] != TARGET_IMAGE_SIZE[0]
        or image_array.shape[1] != TARGET_IMAGE_SIZE[1]
    ):
        raise Exception("bad cropping!")

    return image_array, left_top_corner, right_bottom_corner


def augment(image_array):
    images_arrays = [image_array]

    images_arrays.append(np.rot90(image_array, k=1))
    images_arrays.append(np.rot90(image_array, k=2))
    images_arrays.append(np.rot90(image_array, k=3))

    flipped = np.flip(image_array, axis=0)
    images_arrays.append(np.rot90(flipped, k=1))
    images_arrays.append(np.rot90(flipped, k=2))
    images_arrays.append(np.rot90(flipped, k=3))

    flipped = np.flip(image_array, axis=1)
    images_arrays.append(np.rot90(flipped, k=1))
    images_arrays.append(np.rot90(flipped, k=2))
    images_arrays.append(np.rot90(flipped, k=3))

    return images_arrays


def get_landmarks(csv_pather, left_top_corner, right_bottom_corner):
    filtered_landmarks = []
    with open(csv_pather.full_path) as csv_file:
        reader = csv.reader(csv_file)  # change contents to floats
        next(reader)  # skip header
        for row in reader:  # each row is a list
            x = int(float(row[1]))
            y = int(float(row[2]))

            if left_top_corner[1] <= x <= right_bottom_corner[1]:
                if left_top_corner[0] <= y <= right_bottom_corner[0]:
                    # shift coordinates
                    y -= left_top_corner[0]
                    x -= left_top_corner[1]
                    filtered_landmarks.append([y, x])

    return filtered_landmarks


def patches(images_arrays, patches_size, random_patches_count):
    patches = []
    for image in images_arrays:
        patches += random_array_patches(
            image, patches_size, random_patches_count, random_generator=random_generator
        )
    return patches


def save_image(image_array, target_pather, hierarchy, image_name):
    new_file_name = "_".join((*hierarchy, image_name))
    saving_path = target_pather.join(new_file_name, "jpg")

    if np.std(image_array) > ACCEPTED_IMAGES_MIN_STD:
        new_file_name = "_".join((*hierarchy, image_name))
        Image.fromarray(image_array).save(saving_path)

        return new_file_name
    else:
        saving_path = (
            target_pather.parent().deeper("skipped").join(new_file_name, "jpg")
        )
        Image.fromarray(image_array).save(saving_path)
        print("skipped file: " + saving_path)


def save_landmarks(filtered_landmarks, target_pather, new_file_name):
    with open(target_pather.join(new_file_name, "csv"), "w") as csv_file:
        write = csv.writer(csv_file)
        write.writerows(filtered_landmarks)


def save_training_image(
    image_array, image_pather, target_pather, hierarchy, valid_counter, total_counter
):
    images_arrays = [image_array]

    if AUGMENTATION:
        images_arrays = augment(image_array)

    if RANDOM_PATCHES_COUNT > 0:
        images_arrays = patches(
            images_arrays, RANDOM_PATCHES_SIZE, RANDOM_PATCHES_COUNT
        )

    saving_path = target_pather.deeper("train")

    for i in range(len(images_arrays)):
        new_file_name = save_image(
            images_arrays[i], saving_path, hierarchy, image_pather.name + f"_patch_{i}"
        )

    if new_file_name is None:
        print(
            "saved (train) #"
            + str(valid_counter)
            + "/"
            + str(total_counter)
            + " : "
            + saving_path.directory
        )
    else:
        print(
            "saved (train) #"
            + str(valid_counter)
            + "/"
            + str(total_counter)
            + " : "
            + saving_path.join(new_file_name, "jpg")
        )


def save_test_and_landmark(
    image_array,
    filtered_landmarks,
    target_pather,
    hierarchy,
    image_name,
    valid_counter,
    total_counter,
):
    images_arrays = [image_array]

    if TEST_RANDOM_PATCHES_COUNT > 0:
        images_arrays = patches(
            images_arrays, TEST_RANDOM_PATCHES_SIZE, TEST_RANDOM_PATCHES_COUNT
        )

    if AUGMENTATION:
        images_arrays = augment(image_array)

    saving_path = target_pather.deeper("test")
    # new_file_name = save_image(image_array, \
    #                             saving_path, \
    #                             hierarchy, \
    #                             image_name)

    for i in range(len(images_arrays)):
        new_file_name = save_image(
            images_arrays[i], saving_path, hierarchy, image_pather.name + f"_aug_{i}"
        )

    if new_file_name is not None:
        if filtered_landmarks is not None:
            save_landmarks(filtered_landmarks, saving_path, new_file_name)

        print(
            "saved (test) #"
            + str(valid_counter)
            + "/"
            + str(total_counter)
            + " : "
            + saving_path.join(new_file_name, "jpg")
        )
    else:
        print(
            "saved (test) #"
            + str(valid_counter)
            + "/"
            + str(total_counter)
            + " : "
            + saving_path.directory
        )


# endregion

for path in Pather(IMAGES_ROOT_DIR).contents(".jpg", True):
    total_counter += 1
    image_pather = Pather(path)
    # the name will include folder structure that it is within
    hierarchy = (image_pather.parent().parent().name, image_pather.parent().name)
    csv_pather = Pather(
        landmarks_pather.full_path, *hierarchy, image_pather.name + ".csv"
    )

    image_array, left_top_corner, right_bottom_corner = crop_from_center(path)

    # some images are originally for evaluation, and has no landmarks, so we can
    # use those for training only
    if csv_pather.exists():
        if GENERATE_LANDMARKS:
            filtered_landmarks = get_landmarks(
                csv_pather, left_top_corner, right_bottom_corner
            )

            # in some cases, we may have very few landmarks in the resultant patch,
            # we do not need those
            valid_counter += 1
            if len(filtered_landmarks) >= MIN_LANDMARKS_COUNT:
                save_test_and_landmark(
                    image_array,
                    filtered_landmarks,
                    target_pather,
                    hierarchy,
                    image_pather.name,
                    valid_counter,
                    total_counter,
                )
            else:
                save_training_image(
                    image_array,
                    image_pather,
                    target_pather,
                    hierarchy,
                    valid_counter,
                    total_counter,
                )
        else:
            valid_counter += 1
            save_test_and_landmark(
                image_array,
                None,
                target_pather,
                hierarchy,
                image_pather.name,
                valid_counter,
                total_counter,
            )
    else:  # if there is no csv, or not enough landmarks, consider it as training image
        valid_counter += 1
        save_training_image(
            image_array,
            image_pather,
            target_pather,
            hierarchy,
            valid_counter,
            total_counter,
        )
