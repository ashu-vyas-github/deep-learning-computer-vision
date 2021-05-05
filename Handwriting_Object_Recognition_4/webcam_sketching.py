#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashutosh

A simple OpenCV program to generate sketch from live webcam video feed.
"""
import cv2


def sketch_generation(image):
    """
    Function to generate live sketch from webcam feed

    Parameters
    ----------
    image : ndarray
        One frame from the X-fps webcam video feed.

    Returns
    -------
    mask : ndarray
        Sketched image.

    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    canny_edges = cv2.Canny(img_gray_blur, 50, 10)
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY)  # binary image

    return mask


def webcam_video_capture():
    """
    Function to initialize webcam feed.

    Returns
    -------
    int
        Null.

    """
    webcam_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam_capture.read()
        cv2.imshow('Live Sketcher', sketch_generation(frame))
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break
    webcam_capture.release()  # release camera and close windows
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

    print("Program to create sketch from live webcam video feed.")
    webcam_video_capture()
    print("\nDone")

    return 0


if __name__ == "__main__":
    main()
