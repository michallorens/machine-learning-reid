import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

from features.color_histograms import ColorHistograms
from my_reid import Reid

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
reid = Reid('logs/color-histograms/viper/inception/model.pt', ColorHistograms())

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = image.copy()

    img = np.array(image)
    img_bgr = img[:, :, ::-1].copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    for (x, y, w, h) in rects:
        id = image.copy()[x:x+w, y:y+h]
        print(reid(id, 10)[1])

    print("[INFO] {}: {} original boxes, {} after suppression".format(
        "costam", len(rects), len(pick)))

    frame = image

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()