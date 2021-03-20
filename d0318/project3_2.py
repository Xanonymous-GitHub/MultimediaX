import cv2
import numpy as np


def binarize(img, threshold):
    ret, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary


def filter_dots(img, iters):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((1, 1)), iterations=iters)


def draw_lines(img, lines):
    for line in lines:
        rad, angle = line[0]
        print('radius =', rad, ', angle =', angle)
        a, b = np.cos(angle), np.sin(angle)
        x0, y0 = a * rad, b * rad
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print(len(lines))
    return img


def run_2():
    print('project3-2')
    floor = cv2.resize(cv2.imread('d0318/assets/floor.jpg'), (562, 1000))
    floor_grey = cv2.cvtColor(floor, cv2.COLOR_BGR2GRAY)
    cv2.imshow('floor', floor_grey)
    cv2.waitKey(0)

    bin1 = binarize(floor_grey, 100)
    bin2 = binarize(floor_grey, 120)
    # cv2.imshow('binary', bin1)
    # cv2.waitKey(0)

    bin1 = filter_dots(bin1, 3)
    bin2 = filter_dots(bin2, 3)
    # cv2.imshow('filter dots', bin1)
    # cv2.waitKey(0)

    bin1 = cv2.GaussianBlur(bin1, (9, 9), 0)
    bin2 = cv2.GaussianBlur(bin2, (9, 9), 0)
    # cv2.imshow('blur', bin1)
    # cv2.waitKey(0)

    edges = cv2.Canny(bin1, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/2, 250)
    floor = draw_lines(floor, lines)

    edges = cv2.Canny(bin2, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/2, 250)
    floor = draw_lines(floor, lines)

    cv2.imshow('floor', floor)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
