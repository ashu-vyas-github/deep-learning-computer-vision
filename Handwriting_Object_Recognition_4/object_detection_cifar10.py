#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashutosh

A simple program to build CNN classifier to classify 10 objects from CIFAR10 dataset.
"""
import cv2
import numpy
from keras.models import load_model
from keras.datasets import cifar10

numpy.random.seed(seed=42)


def display_test_result(test_image, pred_class, window_title, scale, img_width, img_height, color=True):
    """
    Function to output class label text for a test sample and display with OpenCV.

    Parameters
    ----------
    test_image : ndarray
        Test image sample.
    pred_class : int
        Encoded class label as 0-9.
    window_title : string
        Title of the OpenCV result window.
    scale : float
        Image scaling factor.
    img_width : int
        Image width.
    img_height : int
        Image height.
    color : boolean, optional
        Flag to indicate color (True) or grayscale images (False). The default is True.

    Returns
    -------
    int
        DESCRIPTION.

    """

    black_rgb = [0, 0, 0]

    pred_class = int(pred_class)
    if pred_class == 0:
        pred = "airplane"
    if pred_class == 1:
        pred = "automobile"
    if pred_class == 2:
        pred = "bird"
    if pred_class == 3:
        pred = "cat"
    if pred_class == 4:
        pred = "deer"
    if pred_class == 5:
        pred = "dog"
    if pred_class == 6:
        pred = "frog"
    if pred_class == 7:
        pred = "horse"
    if pred_class == 8:
        pred = "ship"
    if pred_class == 9:
        pred = "truck"

    expanded_image = cv2.copyMakeBorder(
        test_image, 0, 0, 0, test_image.shape[0]*3, cv2.BORDER_CONSTANT, value=black_rgb)
    if color is False:
        expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (300, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    cv2.imshow(window_title, expanded_image)

    return 0


def image_classifier_cifar10(classifier, img_width=32, img_height=32, img_depth=3, scale=8, color=True):
    """
    CIFAR10 pretrained classifier.

    Parameters
    ----------
    classifier : Keras CNN classifier object.
        CNN classifier from the h5 file.
    img_width : int, optional
        Image width. The default is 32.
    img_height : int, optional
        Image height. The default is 32.
    img_depth : int, optional
        #channels for color images. The default is 3.
    scale : float, optional
        Scaling factor to enlarge images. The default is 8.
    color : boolean, optional
        Flag to indicate color (True) or grayscale images (False). The default is True.

    Returns
    -------
    int
        Null.

    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    for i in range(0, 10):
        random_test_image = x_test[numpy.random.randint(0, len(x_test))]
        test_image_enlarged = cv2.resize(random_test_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # reshape image to (#sample, width, height, #channels)
        random_test_image = random_test_image.reshape(1, img_width, img_height, img_depth)

        predicted_class = str(numpy.argmax(classifier.predict(random_test_image), axis=-1)[0])
        display_test_result(test_image_enlarged, predicted_class,
                            "Test image prediction label ", scale, img_width, img_height, color)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0


def main():
    """
    The main function to execute upon call.

    Returns
    -------
    int
        returns integer 0 for safe executions.
    """

    print("Simple classifier for object detection CIFAR10 dataset.")

    classifier = load_model('./trained_model/cifar_simple_cnn.h5')
    image_classifier_cifar10(classifier)

    print("\nDone")

    return 0


if __name__ == "__main__":
    main()
