import math
import os
import random
import webbrowser

import cv2
import matplotlib.pyplot as plt
import numpy as np

from outputs import showResultsHTML
from numpy import linalg as LA

def distanceLinePoint(line, point):
    line_start = [line[0], line[1]]
    line_end = [line[2], line[3]]
    line_vect = [line_end[0] - line_start[0], line_end[1] - line_start[1]]
    line_point_vec = [line_start[0] - point[0], line_start[1] - point[1]]

    print("start: ", line_start)
    print("end: ", line_end)
    dist = LA.norm(np.cross(line_vect, line_point_vec))/LA.norm(line_vect)
    print("distance: ", dist)

def connectLines(all_lines):
    pass

def detectLinesHough(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    bw_swap = cv2.bitwise_not(thresholded)

    dilated = cv2.dilate(bw_swap, np.ones((4, 4), dtype=np.uint8)) # 3,3
    #cv2.imshow("dilated", dilated)
    eroded = cv2.erode(dilated, np.ones((4, 4), dtype=np.uint8)) #aj 2,2 alebo 3,3
    #cv2.imshow("eroded", eroded)

    edged = eroded

    # edged = cv2.Canny(blurred, 10, 100)
    # # edged = cv2.dilate(edged, np.ones((3, 3), dtype=np.uint8))
    # edged = cv2.dilate(edged, np.ones((10, 10), dtype=np.uint8))
    # edged = cv2.erode(edged, np.ones((10, 10), dtype=np.uint8))

    rho = 0.7  # distance resolution in pixels of the Hough grid
    theta = 3*np.pi / 180  # The resolution of the parameter theta in radians: 1 degree
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(edged, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            b = random.randint(0, 255)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            cv2.line(img_copy, (x1, y1), (x2, y2), (b, g, r), 2)

    # cv2.imshow("Edged image", edged)
    cv2.imshow("Lines", img_copy)

    # doHistogram(lines)
    return img_copy, lines, edged
    #return drawLines(img, lines)


def distanceOfPoints(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def doHistogram(lines_points, img_name):
    dst_folder = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines_histogram"
    folder_content = os.listdir(dst_folder)
    # lines points: [[x_start1, y_start1, x_end1, y_end1], [x_start2, y_start2, x_end2, y_end2]...]
    lines_points = lines_points.ravel()
    all_distances = []

    for start_point in range(0, len(lines_points), 2):
        for end_point in range(start_point + 2, len(lines_points), 2):
            x1 = lines_points[start_point]
            y1 = lines_points[start_point + 1]
            x2 = lines_points[end_point]
            y2 = lines_points[end_point + 1]
            distance = distanceOfPoints(x1, y1 , x2, y2)
            all_distances.append(distance)

    n, bins, _ = plt.hist(all_distances, edgecolor="white", bins='auto')
    x = bins[1] - bins[0]
    plt.xlabel(f'Distance: interval {x: .4f}, min_dst: {min(all_distances): .4f}, max_dst: {max(all_distances): .4f}')
    plt.ylabel('Frequency')
    plt.title('Distances of line points')
    #plt.show()

    res_name = getResultName(img_name, "histogram")
    res_path = dst_folder + '/' + res_name
    # print(res_name)
    # print(res_path)
    if res_name in folder_content:
        os.remove(res_path)

    plt.savefig(res_path)

def getResultName(img_name, description):
    start = len(img_name) - 6
    exten_index = img_name.find('.', start)
    result_name = img_name[:exten_index] + '_' + description + img_name[exten_index:]
    return result_name

def saveImage(dst_dir, img_name, description, res_img):
    result_name = getResultName(img_name, description)
    #print(result_name)
    all_images = os.listdir(dst_dir)

    result_path = dst_dir + '/' + result_name
    if result_name in all_images:
        os.remove(result_path)

    cv2.imwrite(result_path, res_img)


def getAllImages():
    folder_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    dst_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines"
    input_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines_input"
    all_images = os.listdir(folder_dir)

    for image_name in all_images:
        path = folder_dir + '/' + image_name
        img = cv2.imread(path)
        img_hlines, lines, input_img = detectLinesHough(img)
        saveImage(dst_dir, image_name, 'hough_lines', img_hlines)
        saveImage(input_dir, image_name, 'input', input_img)
        # doHistogram(lines, image_name)


if __name__ == '__main__':
    # load image
    #img = cv2.imread('C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1/ElCerrito.jpg')
    #img = cv2.imread('images/ERD_basic1_dig.png')
    # img = cv2.imread('images/shapes_hndw.png')
    img = cv2.imread('images/ERD_simple_HW_noText_smaller.jpg')
    # img = cv2.imread('images/shapes.png')
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    # resize to half of the size
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #
    img_hl, lines, _ = detectLinesHough(img)

    # cv2.imshow("hlines1", img_hl)
    # doHistogram(lines, "pokus")
    #cv2.imshow("Original image", img)

    # getAllImages()
    # showResultsHTML()

    #distanceLinePoint((1,1,5,5), [7,7])
    #print(30 <= 25 <= 20)

    # wait until key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
