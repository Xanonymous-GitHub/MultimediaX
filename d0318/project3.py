import cv2
import numpy as np


def base_d_img(img):
    height = img.shape[0]
    width = img.shape[1]
    return np.zeros((height, width, 3), np.uint8)


def clear_dirty(img, iters):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=iters)


def split_and_calculate_coins():
    print('project3-1: split coins')
    coin = cv2.imread('d0318/assets/coin.jpg')

    coin = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)
    # coin = cv2.erode(coin, np.ones((3, 3)), iterations=1)
    # for _ in range(10):
    #     coin = cv2.medianBlur(coin, 3)

    # ret, binary = cv2.threshold(coin, 65, 255, cv2.THRESH_BINARY)
    # cv2.imshow('coin', binary)
    # cv2.waitKey(0)

    # binary -= cv2.erode(binary, np.ones((3, 3)), iterations=2)
    # coin = clear_dirty(binary, 3)

    # cv2.imshow('clear', binary)
    # cv2.waitKey(0)

    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4, ltype=cv2.CV_32S)
    circles = cv2.HoughCircles(coin, cv2.HOUGH_GRADIENT, dp=1, minDist=400, param1=50, param2=30, minRadius=150, maxRadius=200)
    print(circles)
    print(len(circles[0]))

    # draw on new img
    blank = base_d_img(coin)
    for circle in circles[0]:
        x, y, r = circle
        cv2.rectangle(coin, (x-r, y-r), (x+r, y+r), (255, 255, 0), 5)

    # coin[:, :, 0] += blank[:, :, 0]
    # coin[:, :, 1] += blank[:, :, 1]
    # coin[:, :, 2] += blank[:, :, 2]
    cv2.imshow('owob', coin)
    cv2.waitKey(0)

    # print('num_labels =', num_labels)
    # print('stats =', stats)
    # print('centroids =', centroids)
    # print('labels = ', labels)

    cv2.destroyAllWindows()


def run():
    split_and_calculate_coins()
