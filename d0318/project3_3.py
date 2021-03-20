import cv2
import numpy as np


boundaries = [
    ([17, 15, 100], [50, 56, 200]),
    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250]),
    ([103, 86, 65], [145, 133, 128])
]


def show_and_wait(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def get_avg_color(img, x, y, width, height):
    print('width =', width, ', height = ', height)
    arr = np.array([0, 0, 0], dtype='uint8')
    for i in range(x, x + width):
        for j in range(y, y + height):
            arr[0] += img[i, j, 0]
            arr[1] += img[i, j, 1]
            arr[2] += img[i, j, 2]
            # arr += img[i, j]
            # print(arr[i, j])
            # print(img[i, j])
    return arr / (width * height)


def run_3():
    print('project3-3')

    # import and resize image
    coin = cv2.resize(cv2.imread('d0318/assets/coin2.jpg'), (1000, 563))

    coin_grey = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(coin_grey, 80, 255, cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binary, 1)
    binary = cv2.GaussianBlur(binary, (1, 1), 0)
    binary = cv2.erode(binary, np.ones((2, 2)), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=None)
    show_and_wait('detection', binary)

    red = (0, 0, 255)
    orange = (0, 97, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    purple = (255, 0, 255)
    white = (255, 255, 255)

    money = 0
    for stat in stats:
        x, y, width, height, area = stat
        avg_len = (width + height) * 0.5

        # process connected component with certain length
        if 50 < avg_len < 150:
            # set border color according to length
            if avg_len < 57:
                color = red
                money += 1
            elif avg_len < 65:
                color = orange
                money += 5
            elif avg_len < 75:
                color = yellow
                money += 10
            else:
                color = green
                money += 50
            # draw rectangle around coin
            cv2.rectangle(coin, (x, y), (x + width, y + height), color, 2)

        if (375 < width < 475 and 150 < height < 250) or (375 < height < 475 and 175 < width < 250):
            rgb = [round(np.average(coin[y: y+height, x: x+width, k])) for k in range(3)]
            if rgb[0] > rgb[1] and rgb[0] > rgb[2]:
                color = white
            elif rgb[2] > 180:
                color = blue
            else:
                color = purple
            print(width, height)
            cv2.rectangle(coin, (x, y), (x + width, y + height), color, 5)

    print('錢幣總額 =', money)

    cv2.imshow('coin', coin)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
