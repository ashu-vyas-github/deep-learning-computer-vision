# -*- coding: utf-8 -*-
"""
@author: ashutosh

One line summary of the file

Explanation here. With 72 character limit.

"""


import os
import cv2
import numpy
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from preprocessors import x_cord_contour, makeSquare, resize_to_pixel

numpy.random.seed(seed=42)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def display_test_result(test_image, pred_class, window_title):
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
        # reshape original image to satisfy classifier input requirements (#sample, image height, image width, #channels)
        random_test_image = random_test_image.reshape(1, 28, 28, 1)
        predicted_class = str(numpy.argmax(classifier.predict(random_test_image), axis=-1)[0])  # get prediction
        display_test_result(image_enlarged, predicted_class, "Test vs. Predicted Class")  # show results
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


def train_mnist_classifier():

    # Training Parameters
    batch_size = 4
    epochs = 5

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows = x_train[0].shape[0]
    img_cols = x_train[0].shape[1]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # change datatype from uint8 to float32 and normalize
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # one hot encode target classes
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("Number of Classes: " + str(y_test.shape[1]))

    num_classes = y_test.shape[1]
    num_pixels = x_train.shape[1] * x_train.shape[2]

    classifier = Sequential()  # create classifier

    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(num_classes, activation='softmax'))

    classifier.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    print(classifier.summary())

    history = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                             verbose=1, validation_data=(x_test, y_test))

    score = classifier.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

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

    print("Part 1 of program.")
    classifier = load_model('./trained_model/mnist_simple_cnn.h5')  # loads a saved CNN classifier
    # pretrained_saved_classifier(classifier)  # performance of pretrained classifier

    print("Part 2 of program.")
    # image = cv2.imread('images/numbers.jpg')
    # test_pretrained_classifier(image, classifier)

    print("Part 3 of program.")
    train_mnist_classifier()

    print("\nDone")

    return 0


if __name__ == "__main__":
    main()
