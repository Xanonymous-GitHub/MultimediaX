import cv2
import numpy as np


def show_img(*images):
    if not isinstance(images, list):
        images = [images]

    for image in images[0]:
        cv2.imshow('', image)
        cv2.waitKey(0)


def run_4():
    print('project3-4: combine fruits')

    pyramid_layer_amount = 5

    # get original image.
    apple = cv2.imread('d0318/assets/apple.jpg')
    orange = cv2.imread('d0318/assets/orange.jpg')

    # make a copy image of both apple and orange.
    apple_copy = apple.copy()
    orange_copy = orange.copy()

    # these lists will store the Gaussian pyramid layers of A or B, the layer 0 is equal to the original image.
    apple_gaussian_pyramid = [apple_copy]
    orange_gaussian_pyramid = [orange_copy]

    for _ in range(pyramid_layer_amount):
        # generate Gaussian pyramid for A & B.
        apple_copy = cv2.pyrDown(apple_copy)
        orange_copy = cv2.pyrDown(orange_copy)

        # store to list.
        apple_gaussian_pyramid.append(apple_copy)
        orange_gaussian_pyramid.append(orange_copy)

    # these lists will store the Laplacian pyramid layers of A or B, the layer 0 is the top of the Gaussian pyramid.
    apple_laplacian_pyramid = [apple_gaussian_pyramid[-1]]
    orange_laplacian_pyramid = [orange_gaussian_pyramid[-1]]

    for i in range(pyramid_layer_amount, 0, -1):
        # apple
        apple_laplacian_pyramid.append(
            cv2.subtract(
                apple_gaussian_pyramid[i - 1],
                cv2.resize(
                    cv2.pyrUp(
                        apple_gaussian_pyramid[i]
                    ), apple_gaussian_pyramid[i - 1].shape[-2::-1]
                )
            )
        )

        # orange
        orange_laplacian_pyramid.append(
            cv2.subtract(
                orange_gaussian_pyramid[i - 1],
                cv2.resize(
                    cv2.pyrUp(
                        orange_gaussian_pyramid[i]
                    ), orange_gaussian_pyramid[i - 1].shape[-2::-1]
                )
            )
        )

    # Now add left and right halves of images in each level
    combined_laplacian_layers = []
    for apple_layer, orange_layer in zip(apple_laplacian_pyramid, orange_laplacian_pyramid):
        rows, cols, dpt = apple_layer.shape
        combined_laplacian_layers.append(
            np.hstack(
                (
                    apple_layer[:, :cols // 2],
                    orange_layer[:, cols // 2:]
                )
            )
        )

    # merge all laplacian_layers to a single image.
    fused = combined_laplacian_layers[0]
    for _, layer in enumerate(combined_laplacian_layers[1:]):
        fused = cv2.pyrUp(fused)
        fused = cv2.resize(fused, layer.shape[-2::-1])
        fused = cv2.add(fused, layer)

    show_img(fused)
