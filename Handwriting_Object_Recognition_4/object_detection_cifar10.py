#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashutosh

A simple program to build CNN classifier to classify 10 objects from CIFAR10 dataset.
"""
import cv2
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop

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


def pretrained_saved_classifier(classifier, x_train, y_train, x_test, y_test, img_width=32, img_height=32, img_depth=3, scale=8, color=True):
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


def train_cifar10_classifier(x_train, y_train, x_test, y_test):

    # Training Parameters
    batch_size = 4
    num_classes = 10
    epochs = 5

    # change datatype from uint8 to float32 and normalize
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # one hot encode target classes
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("Number of Classes: " + str(y_test.shape[1]))

    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    classifier.add(Activation('relu'))
    classifier.add(Conv2D(32, (3, 3)))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(64, (3, 3), padding='same'))
    classifier.add(Activation('relu'))
    classifier.add(Conv2D(64, (3, 3)))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Flatten())
    classifier.add(Dense(512))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(num_classes))
    classifier.add(Activation('softmax'))

    # initiate RMSprop optimizer and configure some parameters
    rmsprop_optimizer = RMSprop(lr=0.0001, decay=1e-6)
    classifier.compile(loss='categorical_crossentropy', optimizer=rmsprop_optimizer, metrics=['accuracy'])

    print(classifier.summary())

    history = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                             validation_data=(x_test, y_test), shuffle=True)

    classifier.save("./trained_model/cifar10_exercise_cnn.h5")

    # Evaluate the performance of our trained model
    scores = classifier.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    history_dict = history.history

    plotting_loss_accuracy(history_dict, metric_train="loss", metric_test="val_loss", title="Loss CIFAR10",
                           label_train="Train loss", label_test="Test loss", x_label="Epochs", y_label="Loss")

    plotting_loss_accuracy(history_dict, metric_train="accuracy", metric_test="val_accuracy", title="Accuracy CIFAR10",
                           label_train="Train acc.", label_test="Test acc.", x_label="Epochs", y_label="Accuracy")

    return 0


def plotting_loss_accuracy(history_dict, metric_train, metric_test, title, label_train, label_test, x_label, y_label):

    metric_train_values = history_dict[metric_train]
    metric_test_values = history_dict[metric_test]
    epochs = range(1, len(metric_train_values) + 1)

    plt.figure()
    plt.title(title)
    line1 = plt.plot(epochs, metric_train_values, label=label_train)
    line2 = plt.plot(epochs, metric_test_values, label=label_test)
    plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
    plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.show()

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

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    classifier = load_model('./trained_model/cifar_simple_cnn.h5')

    pretrained_saved_classifier(classifier, x_train, y_train, x_test, y_test)

    train_cifar10_classifier(x_train, y_train, x_test, y_test)

    print("\nDone")

    return 0


if __name__ == "__main__":
    main()
