import cv2
import numpy as np


def remove_black_dots_in_circle(circle):
    return cv2.morphologyEx(circle, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=9)


def get_white_man_with_padding_and_black_background(man):
    return man - cv2.erode(man, np.ones((3, 3)), iterations=3)


def run():
    circle = cv2.imread('assets/img/circle.jpg', cv2.COLOR_BGR2GRAY)
    man = cv2.imread('assets/img/man.jpg', cv2.COLOR_BGR2GRAY)

    cv2.imshow('circle', circle)
    cv2.waitKey(0)
    cv2.imshow('circle', remove_black_dots_in_circle(circle))
    cv2.waitKey(0)

    cv2.imshow('man', man)
    cv2.waitKey(0)
    cv2.imshow('man', get_white_man_with_padding_and_black_background(man))
    cv2.waitKey(0)

    cv2.destroyAllWindows()
