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


def pretrained_saved_classifier():
    """
    Function to showcase a pretrained and saved classifier on MNIST dataset.

    Returns
    -------
    int
        Null.
    """

    classifier = load_model('./trained_model/mnist_simple_cnn.h5')  # loads a saved CNN classifier
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


def main():
    """
    The main function to execute upon call.

    Returns
    -------
    int
        returns integer 0 for safe executions.
    """

    print("Simple classifier for MNIST dataset.")

    pretrained_saved_classifier()

    print("\nDone")

    return 0


if __name__ == "__main__":
    main()
