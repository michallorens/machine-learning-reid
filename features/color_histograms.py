import cv2
import torch
import numpy as np


class ColorHistograms(object):
    def __call__(self, img):
        img = np.array(img)
        img_bgr = img[:, :, ::-1].copy()

        t_img = torch.Tensor(8, 256, 1)

        for i in range(0, 3):
            t_img[i] = torch.Tensor(cv2.calcHist([img_bgr], [i], None, [256], [0, 256]))

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for i in range(0, 3):
            t_img[i + 3] = torch.Tensor(cv2.calcHist([img_hsv], [i], None, [256], [0, 256]))

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        for i in range(1, 3):
            t_img[i + 5] = torch.Tensor(cv2.calcHist([img_yuv], [i], None, [256], [0, 256]))

        return t_img.cuda()
