import cv2
import numpy as np


def get_coin_color(avg_len):
    red = (0, 0, 255)
    orange = (0, 97, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    if avg_len < 57:
        return red, 1
    elif avg_len < 65:
        return orange, 5
    elif avg_len < 75:
        return yellow, 10
    else:
        return green, 50


def get_cash_color(color):
    blue = (255, 0, 0)
    purple = (255, 0, 255)
    white = (255, 255, 255)

    if color[0] > color[1] and color[0] > color[2]:
        return white, 1000
    elif color[2] > 180:
        return blue, 100
    else:
        return purple, 500


def run_3():
    print('project3-3')

    # import and resize image
    coin = cv2.resize(cv2.imread('d0318/assets/coin2.jpg'), (1000, 563))

    # convert image to grayscale
    coin_grey = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

    # threshold image
    ret, binary = cv2.threshold(coin_grey, 80, 255, cv2.THRESH_BINARY)

    # remove black area inside coins
    binary = cv2.medianBlur(binary, 1)
    binary = cv2.GaussianBlur(binary, (1, 1), 0)
    binary = cv2.erode(binary, np.ones((2, 2)), iterations=1)

    # get stats from connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=None)

    # draw rectangle around coins and calculate total money
    total = 0
    for stat in stats:
        x, y, width, height, area = stat
        avg_len = (width + height) * 0.5

        # process connected component with certain length
        if 50 < avg_len < 150:
            # set border color according to length
            color, money = get_coin_color(avg_len)
            total += money
            # draw rectangle around coin
            cv2.rectangle(coin, (x, y), (x + width, y + height), color, 2)

        if (375 < width < 475 and 150 < height < 250) or (375 < height < 475 and 175 < width < 250):
            rgb = [round(np.average(coin[y: y + height, x: x + width, k])) for k in range(3)]
            color, money = get_cash_color(rgb)
            total += money
            cv2.rectangle(coin, (x, y), (x + width, y + height), color, 5)

    print('錢幣總額 =', total)
    cv2.imshow('coin', coin)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
