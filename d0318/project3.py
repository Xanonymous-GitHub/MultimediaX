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

    coin = cv2.resize(cv2.imread('d0318/assets/coin.jpg'), (1000, 563))

    coin_gray = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(coin_gray, 90, 255, cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binary, 1)
    binary = cv2.GaussianBlur(binary, (1, 1), 0)
    binary = cv2.erode(binary, np.ones((2, 2)), iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=None)

    red = (0, 0, 255)
    orange = (0, 97, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    for stat in stats:
        if (50 < stat[2] < 150) and (50 < stat[3] < 150):
            avg_len = (stat[2] + stat[3]) * 0.5

            if avg_len < 100:
                color = red
            elif avg_len < 110:
                color = orange
            elif avg_len < 124:
                color = yellow
            else:
                color = green

            cv2.rectangle(coin, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), color, 2)

    cv2.imshow('coin', coin)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def run():
    split_and_calculate_coins()
