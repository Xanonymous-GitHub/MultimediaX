import cv2
import numpy as np


def split_and_calculate_coins():
    print('project3-1')

    # import and resize image
    coin = cv2.resize(cv2.imread('d0318/assets/coin.jpg'), (1000, 563))

    # convert image to grayscale
    coin_grey = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

    # threshold image
    ret, binary = cv2.threshold(coin_grey, 85, 255, cv2.THRESH_BINARY)

    # remove black area inside coins
    binary = cv2.medianBlur(binary, 1)
    binary = cv2.GaussianBlur(binary, (1, 1), 0)
    binary = cv2.erode(binary, np.ones((2, 2)), iterations=2)

    # get stats from connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=None)

    # set color tuple
    red = (0, 0, 255)
    orange = (0, 97, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    money = 0
    for stat in stats:
        x, y, width, height, area = stat
        avg_len = (width + height) * 0.5

        # process connected component with certain length
        if 50 < avg_len < 150:
            # set border color according to length
            if avg_len < 100:
                color = red
                money += 1
            elif avg_len < 110:
                color = orange
                money += 5
            elif avg_len < 124:
                color = yellow
                money += 10
            else:
                color = green
                money += 50
            # draw rectangle around coin
            cv2.rectangle(coin, (x, y), (x + width, y + height), color, 2)
    print('錢幣總額 =', money)

    cv2.imshow('coin', coin)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def run_1():
    split_and_calculate_coins()
