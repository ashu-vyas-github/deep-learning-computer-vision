# -*- coding: utf-8 -*-
"""
@author: ashutosh

One line summary of the file

Explanation here. With 72 character limit.

"""

import cv2
import numpy
from keras.datasets import mnist
from keras.models import load_model
from preprocessors import x_cord_contour, makeSquare, resize_to_pixel


def draw_test(test_image, pred_class, window_title):
    """
    Function to show a selected test image and its predicted label side-by-side.

    Parameters
    ----------
    test_image : ndarray
        Test image in grayscale.
    pred_class : string
        Predicted class for the test image.
    window_title : string
        Text title for image viewer window.

    Returns
    -------
    int
        Null.

    """
    black_rgb = [0, 0, 0]
    results_stitched = cv2.copyMakeBorder(
        test_image, 0, 0, 0, test_image.shape[0], cv2.BORDER_CONSTANT, value=black_rgb)  # add black background
    results_stitched = cv2.cvtColor(results_stitched, cv2.COLOR_GRAY2BGR)
    # put predicted class result on the black background
    cv2.putText(results_stitched, str(pred_class), (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    cv2.imshow(window_title, results_stitched)

    return 0


def pretrained_saved_classifier(classifier):
    """
    Function to showcase a pretrained and saved classifier on MNIST dataset.

    Returns
    -------
    int
        Null.
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # loads the MNIST dataset

    for i in range(0, 10):
        random_img_idx = numpy.random.randint(0, len(x_test))
        random_test_image = x_test[random_img_idx]  # select a random test image
        image_enlarged = cv2.resize(random_test_image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        # reshape original image to satisfy classifier input requirements
        random_test_image = random_test_image.reshape(1, 28, 28, 1)
        predicted_class = str(numpy.argmax(classifier.predict(random_test_image), axis=-1)[0])  # get prediction
        draw_test(image_enlarged, predicted_class, "Test vs. Predicted Class")  # show results
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0


def test_pretrained_classifier(test_image, classifier):
    """
    Function to detect individual digits from an input number image and classify the digits using pretrained classifier.

    Parameters
    ----------
    test_image : ndarray
        Test image with numbers.
    classifier : CNN Keras object
        Pretrained CNN classifier on MNIST dataset.

    Returns
    -------
    int
        DESCRIPTION.

    """

    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Test Image", test_image)
    cv2.waitKey(0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # smoothing edges of the image
    edged = cv2.Canny(blurred, 30, 150)  # Canny edge detection of the image

    try:
        _, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    except ValueError:
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours

    contours = sorted(contours, key=x_cord_contour, reverse=False)  # sort contours left to right using x coordinates

    individual_digits = []  # storing individual digits

    # loop over the contours
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)  # compute the bounding box for the rectangle

        if w >= 5 and h >= 25:
            roi = blurred[y:y + h, x:x + w]
            ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
            roi = makeSquare(roi)
            roi = resize_to_pixel(28, roi)
            cv2.imshow("ROI", roi)
            roi = roi / 255.0
            roi = roi.reshape(1, 28, 28, 1)

            predicted_class = str(numpy.argmax(classifier.predict(roi), axis=-1)[0])  # get prediction
            individual_digits.append(predicted_class)
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(test_image, predicted_class, (x, y + 155), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
            cv2.imshow("Test Image", test_image)  # display predictions
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("The number is: " + ''.join(individual_digits))

    return 0


def main():
    """
    The main function to execute upon call.

    Returns
    -------
    int
        returns integer 0 for safe executions.
    """

    print("Simple classifier for MNIST dataset.")

    classifier = load_model('./trained_model/mnist_simple_cnn.h5')  # loads a saved CNN classifier
    pretrained_saved_classifier(classifier)  # performance of pretrained classifier

    image = cv2.imread('images/numbers.jpg')
    test_pretrained_classifier(image, classifier)

    print("\nDone")

    return 0


if __name__ == "__main__":
    main()
