import cv2
import numpy as np


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def base_d_img(img):
    height = img.shape[0]
    width = img.shape[1]
    return np.zeros((height, width, 3), np.uint8)


def to_r(img):
    _img = base_d_img(img)
    _img[:, :, 2] = img[:, :, 2]
    return _img


def to_g(img):
    _img = base_d_img(img)
    _img[:, :, 1] = img[:, :, 1]
    return _img


def to_b(img):
    _img = base_d_img(img)
    _img[:, :, 0] = img[:, :, 0]
    return _img


def draw():
    p = np.zeros((400, 400, 3), np.uint8)
    p.fill(200)
    cv2.line(p, (87, 87), (255, 255), (0, 0, 255), 5)
    cv2.rectangle(p, (20, 60), (120, 160), (0, 255, 0), 2)
    cv2.rectangle(p, (40, 80), (100, 140), (0, 255, 0), -1)
    cv2.circle(p, (90, 210), 30, (0, 255, 255), 3)
    cv2.circle(p, (140, 170), 15, (255, 0, 0), -1)
    return p


def main():
    img = cv2.imread('img/important2.jpg')
    name = 'da JJ'
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.imshow(name, to_gray(img))
    cv2.waitKey(0)
    cv2.imshow(name, to_ycrcb(img))
    cv2.waitKey(0)
    cv2.imshow(name, to_hsv(img))
    cv2.waitKey(0)
    cv2.imshow(name, to_r(img))
    cv2.waitKey(0)
    cv2.imshow(name, to_g(img))
    cv2.waitKey(0)
    cv2.imshow(name, to_b(img))
    cv2.waitKey(0)
    cv2.imshow('p', draw())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
