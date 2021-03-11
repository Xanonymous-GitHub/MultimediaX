import cv2
import numpy as np


def run():
    circle = cv2.imread('assets/img/circle.jpg', cv2.COLOR_BGR2GRAY)
    man = cv2.imread('assets/img/man.jpg', cv2.COLOR_BGR2GRAY)

    cv2.imshow('circle', circle)
    cv2.waitKey(0)
    circle = cv2.morphologyEx(circle, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=9)
    cv2.imshow('circle', circle)
    cv2.waitKey(0)

    cv2.imshow('man', man)
    cv2.waitKey(0)
    d_man = cv2.erode(man, np.ones((3, 3)), iterations=3)
    man = man - d_man
    cv2.imshow('man', man)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
