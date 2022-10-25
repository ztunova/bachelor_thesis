import math
import os
import random
import webbrowser

import cv2
import numpy as np

from outputs import showResultsHTML


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

    '''
    for i, contour in enumerate(contours):  # loop over one contour area
        for j, contour_point in enumerate(contour):  # loop over the points
            # draw a circle on the current contour coordinate
            cv2.circle(image_copy, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 255), 2, cv2.LINE_AA)
    '''

    #cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")

    for contour in contours:
        x1, y1 = contour[0][0]
        b = random.randint(0, 255)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image_copy, [approx], 0, (b, g, r), 2)
        # print(approx)
        if len(approx) == 4:
            cv2.putText(image_copy, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        elif 6 < len(approx) < 15:
            cv2.putText(image_copy, "Ellipse", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    cv2.imshow("Gray image", gray)
    cv2.imshow("Edged image", edged)
    cv2.imshow("contours", image_copy)


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

        #img = cv2.drawContours(img, [cnt], 0, (0, 255, 255), 2)

        # draw the bounding rectangle
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Bounding Rectangle", img)


def detectLinesHough(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)
    # edged = cv2.dilate(edged, np.ones((3, 3), dtype=np.uint8))
    edged = cv2.dilate(edged, np.ones((10, 10), dtype=np.uint8))
    edged = cv2.erode(edged, np.ones((10, 10), dtype=np.uint8))

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # The resolution of the parameter theta in radians: 1 degree
    threshold = 35  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(edged, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            b = random.randint(0, 255)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            cv2.line(img_copy, (x1, y1), (x2, y2), (b, g, r), 2)

    #cv2.imshow("Edged image", edged)
    #cv2.imshow("Lines", img_copy)

    doHistogram(lines)
    return img_copy
    #return drawLines(img, lines)

def detectLinesLSD(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### Seconds try

    # Create default parametrization LSD
    lsd = cv2.createLineSegmentDetector(0)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    bw_swap = cv2.bitwise_not(thresholded)
    # dilated = cv2.dilate(bw_swap, np.ones((3, 3), dtype=np.uint8))
    eroded = cv2.erode(bw_swap, np.ones((2, 6), dtype=np.uint8))
    edged = eroded

    #cv2.imshow("title", eroded)

    # Detect lines in the image
    lines = lsd.detect(edged)[0]  # Position 0 of the returned tuple are the detected lines

    # Draw detected lines in the image
    drawn_img = lsd.drawSegments(img, lines)

    #print(lines)

    # Show image
    #cv2.imshow("LSD", drawn_img)
    return drawn_img
    #return drawLines(img, lines)


def drawLines(img, lines):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines = np.int0(lines)

    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        b = random.randint(0, 255)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        color = (b, r, g)
        cv2.line(img_copy, (x1, y1), (x2, y2), color, 2)

    #cv2.imshow("Corners", img_copy)
    return img_copy

def distanceOfPoints(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def doHistogram(lines_points):
    # https://stackoverflow.com/questions/9141732/how-does-numpy-histogram-work
    # https://realpython.com/python-histograms/
    # lines points: [[x_start1, y_start1, x_end1, y_end1], [x_start2, y_start2, x_end2, y_end2]...]
    lines_points = lines_points.ravel()
    #print(lines_points)
    start = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    end = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    all_distances = []

    for start_point in range(0, len(lines_points), 2):
        for end_point in range(start_point + 2, len(lines_points), 2):
            #print(f"distance from start point {start_point} {start_point +1} to end point {end_point} {end_point+1}")
            x1 = lines_points[start_point]
            y1 = lines_points[start_point + 1]
            x2 = lines_points[end_point]
            y2 = lines_points[end_point + 1]
            distance = distanceOfPoints(x1, y1 , x2, y2)
            all_distances.append(distance)
            #print("save dst")

    print(all_distances)
    print(len(all_distances))

def saveImage(dst_dir, img_name, description, res_img):
    start = len(img_name) - 6
    exten_index = img_name.find('.', start)
    result_name = img_name[:exten_index] + '_' + description + img_name[exten_index:]
    #print(result_name)
    all_images = os.listdir(dst_dir)

    result_path = dst_dir + '/' + result_name
    if result_name in all_images:
        os.remove(result_path)

    cv2.imwrite(result_path, res_img)


def getAllImages():
    folder_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    dst_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines"
    #dst_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_LSDlines"
    all_images = os.listdir(folder_dir)

    for image_name in all_images:
        path = folder_dir + '/' + image_name
        img = cv2.imread(path)
        img_hlines = detectLinesHough(img)
        saveImage(dst_dir, image_name, 'hough_lines', img_hlines)
        # img_LSDlines = detectLinesLSD(img)
        # saveImage(dst_dir, image_name, 'hough_lines', img_LSDlines)


if __name__ == '__main__':
    # load image
    # img = cv2.imread('C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1/ElCerrito.jpg')
    img = cv2.imread('images/ERD_basic1_dig.png')
    # img = cv2.imread('images/shapes_hndw.png')
    # img = cv2.imread('images/ERD_simple_HW_noText_smaller.jpg')
    # img = cv2.imread('images/shapes.png')
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    # resize to half of the size
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    img_mod = detectLinesHough(img)

    # detectCorners(img)

    #cv2.imshow("Original image", img)

    #getAllImages()

    #showResultsHTML()

    #saveImage('ERD_basic1_dig.png', 'lines', img_mod)

    #computeHistogram(0)

    # wait until key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
