import cv2
import numpy as np
from PIL import ImageGrab
from PIL import Image
import os


def SIFT_Detect(template, screen):

    sift = cv2.SIFT_create()

    template_cv = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2BGR)
    keypoints, descriptors = sift.detectAndCompute(template_cv, None)

    screen_cv = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
    keypoints_screen, descriptors_screen = sift.detectAndCompute(screen_cv, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors_screen, k=2)

    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.6 * m2.distance:
            good_matches.append(m1)

    match_points = np.float32([keypoints_screen[m.trainIdx].pt for m in good_matches])

    # for i in range(len(match_points)):
    #    cv2.circle(screen_cv, tuple(map(int, match_points[i])), 5, (0, 0, 255), -1)

    # cv2.imwrite("match_points.png", screen_cv)

    y, x = 0, 0

    for i in range(len(match_points)):
        y += match_points[i][0]
        x += match_points[i][1]

    if len(match_points) < 4:
        return -1, -1

    return int(y / len(match_points)), int(x / len(match_points))
