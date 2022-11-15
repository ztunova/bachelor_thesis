import copy
import math
import os
import random
import webbrowser

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

from outputs import showResultsHTML
from numpy import linalg as LA

def lineLength(line):
    x1, y1, x2, y2 = line
    # print(x1, y1, x2, y2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def paralelLines(orig_line, cmp_line):
    x1_orig, y1_orig, x2_orig, y2_orig = orig_line
    x1_cmp, y1_cmp, x2_cmp, y2_cmp = cmp_line

    con_l1_start = [x1_orig, y1_orig, x1_cmp, y1_cmp]
    dst_l1_start = lineLength(con_l1_start)
    con_l1_end = [x2_orig, y2_orig, x2_cmp, y2_cmp]
    dst_l1_end = lineLength(con_l1_end)
    con_l2_start = [x1_orig, y1_orig, x2_cmp, y2_cmp]
    dst_l2_start = lineLength(con_l2_start)
    con_l2_end = [x2_orig, y2_orig, x1_cmp, y1_cmp]
    dst_l2_end = lineLength(con_l2_end)

    diff = 3

    if (dst_l1_start <= diff) and (dst_l1_end <= diff):
        #print("orig: ", orig_line, " paralel with cmp: ", cmp_line, " dst strat: ", dst_l1_start, " dst end: ", dst_l1_end )
        return True
    elif (dst_l2_start <= diff) and (dst_l2_end <= diff):
        #print("orig: ", orig_line, " paralel with cmp: ", cmp_line, " dst strat: ", dst_l2_start, " dst end: ", dst_l2_end )
        return True
    else:
        #print("orig: ", orig_line, " NOT paralel with cmp: ", cmp_line, " dst strat: ", dst_l2_start, " dst end: ", dst_l1_end )
        return False


def filterLines(all_lines):
    filtered_lines = []
    #longest_paralel = copy.deepcopy(all_lines[0])
    for i in range(len(all_lines)):
        orig_line = copy.deepcopy(all_lines[i])[0]
        longest_paralel = copy.deepcopy(orig_line)
        if orig_line[0] > 0:
            for j in range(i, len(all_lines)):
                cmp_line = copy.deepcopy(all_lines[j])[0]
                if cmp_line[0] > 0:
                    is_paralel = paralelLines(orig_line, cmp_line)
                    if is_paralel and i != j:
                        length_orig = lineLength(orig_line)
                        length_cmp = lineLength(cmp_line)
                        if length_orig >= length_cmp:
                            all_lines[j][0] = -(all_lines[j][0])
                            #filtered_lines.remove(cmp_line)
                            length_longes = lineLength(longest_paralel)
                            if length_orig > length_longes:
                                longest_paralel = orig_line
                        else:
                            all_lines[i][0] = -(all_lines[i][0])
                            #filtered_lines.remove(orig_line)
                            length_longes = lineLength(longest_paralel)
                            if length_cmp > length_longes:
                                longest_paralel = cmp_line

            if longest_paralel not in filtered_lines:
            #if not(longest_paralel.isin(filtered_lines)):
                filtered_lines.append(longest_paralel)

    return filtered_lines


def distancePointToLineSegment(line, point):
    line_start = [line[0], line[1]]
    line_end = [line[2], line[3]]

    line_vector = [None, None]
    line_vector[0] = line_end[0] - line_start[0]
    line_vector[1] = line_end[1] - line_start[1]

    line_end_point_vector = [None, None]
    line_end_point_vector[0] = point[0] - line_end[0]
    line_end_point_vector[1] = point[1] - line_end[1]

    line_start_point_vector = [None, None]
    line_start_point_vector[0] = point[0] - line_start[0]
    line_start_point_vector[1] = point[1] - line_start[1]

    # dot product of line vector and vector line_end to point
    dot_linev_end_point = line_vector[0] * line_end_point_vector[0] + line_vector[1] * line_end_point_vector[1]
    # dot product of line vector and vector from line_start to point
    dot_linev_start_point = line_vector[0] * line_start_point_vector[0] + line_vector[1] * line_start_point_vector[1]

    dst_point_to_line = 0

    if dot_linev_end_point > 0:
        # bod je niekde za koncovym bodom usecky (v smere vektoru usecky, dalej za koncom)
        x = point[0] - line_end[0]
        y = point[1] - line_end[1]
        dst_point_to_line = math.sqrt(x*x + y*y)
        print("za koncom")

    elif dot_linev_start_point < 0:
        # bod je pred zaciatocnym bodom usecky (proti smeru vektoru usecky, pred zaciatocnym bodom usecky)
        x = point[0] - line_start[0]
        y = point[1] - line_start[1]
        dst_point_to_line = math.sqrt(x * x + y * y)
        print("pred zaciatkom")

    else:
        # bod je niekde medzi koncovymi bodmi usecky
        line_vec_x = line_vector[0]
        line_vec_y = line_vector[1]
        line_length = math.sqrt(line_vec_x * line_vec_x + line_vec_y * line_vec_y)

        line_start_point_x = line_start_point_vector[0]
        line_start_point_y = line_start_point_vector[1]

        dst_point_to_line = abs(line_vec_x * line_start_point_y - line_start_point_x * line_vec_y) / line_length
        print("medzi koncami usecky")

    return dst_point_to_line
def drawLines(img_copy, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            b = random.randint(0, 255)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            cv2.line(img_copy, (x1, y1), (x2, y2), (b, g, r), 2)

    return img_copy

def detect_horizontal_lines(img, copy = None):
    if copy is None:
        copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

    bw_swap = cv2.bitwise_not(threshold)
    dilated = cv2.dilate(bw_swap, np.ones((4, 1), dtype=np.uint8))  # vodorovne: 4,1
    eroded = cv2.erode(dilated, np.ones((1, 9), dtype=np.uint8))  # vodorovne: 1, 9

    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(copy2, [cnt], 0, (b, g, r), 2)
        cv2.drawContours(copy, [box], 0, (0, 255, 0), 2)

    return copy, eroded


def detect_vertical_lines(img, copy = None):
    if copy is None:
        copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

    bw_swap = cv2.bitwise_not(threshold)
    dilated = cv2.dilate(bw_swap, np.ones((1, 4), dtype=np.uint8))  # vodorovne: 4,1
    eroded = cv2.erode(dilated, np.ones((9, 1), dtype=np.uint8))  # vodorovne: 1, 9

    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(copy2, [cnt], 0, (b, g, r), 2)
        cv2.drawContours(copy, [box], 0, (0, 0, 255), 2)

    return copy, eroded

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

    img_copy = drawLines(img_copy, lines)

    # cv2.imshow("Edged image", edged)
    #cv2.imshow("Lines", img_copy)

    # doHistogram(lines)
    return img_copy, lines, edged
    #return drawLines(img, lines)


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
    # folder_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    # dst_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines"
    # input_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines_input"

    folder_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    dst_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_hlines"
    input_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_hlines_input"

    horizontal_lines_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_horizontalLines"
    horizontal_input_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_horizontalLines_input"

    vertical_lines_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_verticalLines"
    vertical_input_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_verticalLines_input"

    horizontal_vertical_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_horizontal_vertical"

    all_images = os.listdir(folder_dir)

    for image_name in all_images:
        path = folder_dir + '/' + image_name
        img = cv2.imread(path)
        #print(image_name)
        img_hlines, lines, input_img = detectLinesHough(img)
        saveImage(dst_dir, image_name, 'hough_lines', img_hlines)
        saveImage(input_dir, image_name, 'input', input_img)

        horizontal_lines, horizontal_lines_input = detect_horizontal_lines(img)
        saveImage(horizontal_lines_dir, image_name, 'horizontal_lines', horizontal_lines)
        saveImage(horizontal_input_dir, image_name, 'horizontal_input', horizontal_lines_input)

        vertical_lines, vertical_lines_input = detect_vertical_lines(img)
        saveImage(vertical_lines_dir, image_name, 'vertical_lines', vertical_lines)
        saveImage(vertical_input_dir, image_name, 'vertical_input', vertical_lines_input)

        horizontal_vertical, _ = detect_vertical_lines(img, horizontal_lines)
        saveImage(horizontal_vertical_dir, image_name, 'horizontal_vertical', horizontal_vertical)


if __name__ == '__main__':
    # load image
    img = cv2.imread('C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1/ElCerrito.jpg')
    #img = cv2.imread('images/ERD_basic1_dig.png')
    #img = cv2.imread('images/sudoku.png')
    #img = cv2.imread('images/ERD_simple_HW_noText_smaller.jpg')
    #img = cv2.imread('images/sampleLines.png')

    # # resize to half of the size
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # hlines, lines, edges = detectLinesHough(img)
    # # # # tutHlines, tut_input = tutorialLines(img)
    # cv2.imshow("hlines", hlines)
    # filtered = filterLines(lines)
    # filimg = drawLines(img.copy(), filtered)
    # cv2.imshow("filtered", filimg)

    getAllImages()
    showResultsHTML()

    #minRec(img)

    #distanceLinePoint((1,1,5,5), [7,7])
    # line = [0,0,8,0]
    # point = [7,4]
    # print(distancePointToLineSegment(line, point))

    # wait until key is pressed
    #cv2.imshow("pomoc", img)

    # print(lineLength([3,0,9,0]))

    # p = ['a', 'b', 'c', 'd', 'e']
    # c = ['1','2','3','4','5']
    # for i in range(len(p)):
    #     for j in range(i, len(c)):
    #         print(p[i], c[j])

    #paralelLines([1,1,7,7], [2,2,9,4])

    # pokus = [[3,0,9,0],[2,1,10,0],[2,7,4,2],[2,0,8,0],[2,7,3,2],[4,5,7,6],[2,8,4,2]]
    # res = filterLines(pokus)
    # print(res)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
