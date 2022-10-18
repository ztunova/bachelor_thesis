import random

import cv2
import numpy as np


def detectAproxPoly(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), "objects were found in this image.")
    #print("contours: ")
    #print(contours)
    image_copy = img.copy()

    for i, contour in enumerate(contours):  # loop over one contour area
        for j, contour_point in enumerate(contour):  # loop over the points
            # draw a circle on the current contour coordinate
            cv2.circle(image_copy, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")

    for contour in contours:
        x1, y1 = contour[0][0]
        b = random.randint(0, 255)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        #cv2.drawContours(image_copy, [approx], 0, (b, g, r), 2)
        # print(approx)
        if len(approx) == 4:
            cv2.putText(image_copy, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        elif 6 < len(approx) < 15:
            cv2.putText(image_copy, "Ellipse", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    cv2.imshow("Gray image", gray)
    cv2.imshow("Edged image", edged)
    cv2.imshow("contours", image_copy)

def blopDetect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_copy = img.copy()

    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True
    params.minCircularity = 0.75

    params.filterByInertia = True
    params.minInertiaRatio = 0.7

    detector = cv2.SimpleBlobDetector_create(params)

    #detect blops
    keypoints = detector.detect(gray)

    imgKeyPoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", imgKeyPoints)

def boundBox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding on the gray image to create a binary image
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #edged = cv2.Canny(blurred, 10, 100)

    # find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # take the first contour
    cnt = contours[1]
    #print(cnt)

    # compute the bounding rectangle of the contour
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        img = cv2.drawContours(img, [cnt], 0, (0, 255, 255), 2)

        # draw the bounding rectangle
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Bounding Rectangle", img)

if __name__ == '__main__':
    # load image
    img = cv2.imread('C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1/Norwalk.jpg')
    #img = cv2.imread('images/ERD_basic1_dig.png')
    #img = cv2.imread('images/shapes_hndw.png')
    #img = cv2.imread('images/ERD_simple_HW_noText_smaller.jpg')
    #img = cv2.imread('images/shapes.png')
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    # resize to half of the size
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    #detectAproxPoly(img)

    #blopDetect(img)

    boundBox(img)

    #cv2.imshow("Original image", img)

    # wait until key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
