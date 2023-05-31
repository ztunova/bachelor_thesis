import argparse
import csv
import os
import random
import shutil

import cv2
import keras_ocr
import math
import numpy as np
import pandas
import pytesseract
import unicodedata

import digital_images_results
from jiwer import cer
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance as dist


from easyocr import Reader
from shapes import *


# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\zofka\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


def random_color():
    b = random.randint(0, 255)
    r = random.randint(0, 255)
    g = random.randint(0, 255)

    return b, g, r


def reorder_rect_points_horizontal_rec(rect):
    max_x = []

    max_x1 = max(rect, key=lambda x: x[0])
    max_x.append(max_x1)
    index = np.where(np.all(rect == max_x1, axis=1))
    if len(index[0]) > 1:
        index = index[0][0]
    rect = np.delete(rect, index, axis=0)
    max_x2 = max(rect, key=lambda x: x[0])
    max_x.append(max_x2)
    index = np.where(np.all(rect == max_x2, axis=1))
    min_x = np.delete(rect, index, axis=0)

    if min_x[0][1] <= min_x[1][1]:
        min_x_min_y = min_x[0].tolist()
        min_x_max_y = min_x[1].tolist()
    else:
        min_x_min_y = min_x[1].tolist()
        min_x_max_y = min_x[0].tolist()

    if max_x1[1] <= max_x2[1]:
        max_x_min_y = max_x1.tolist()
        max_x_max_y = max_x2.tolist()
    else:
        max_x_min_y = max_x2.tolist()
        max_x_max_y = max_x1.tolist()

    return [min_x_min_y, max_x_min_y, max_x_max_y, min_x_max_y]


def dst_of_points(start_point, end_point):
    x_start, y_start = start_point
    x_end, y_end = end_point

    x_diff = x_end - x_start
    y_diff = y_end - y_start

    dst = math.sqrt(x_diff ** 2 + y_diff ** 2)
    return dst


def get_result_name(img_name, description):
    if description == '':
        return img_name

    start = len(img_name) - 6
    exten_index = img_name.find('.', start)
    result_name = img_name[:exten_index] + '_' + description + img_name[exten_index:]
    return result_name


def save_image(dst_dir, img_name, description, res_img):
    result_name = get_result_name(img_name, description)
    all_images = os.listdir(dst_dir)

    result_path = dst_dir + '/' + result_name
    if result_name in all_images:
        os.remove(result_path)

    cv2.imwrite(result_path, res_img)


def get_all_images(orig_img_dir, folder_dir, digital_imgs_contour_dir, removed_shapes_dir, json_output_dir):
    clear_directory(json_output_dir)
    clear_directory(digital_imgs_contour_dir)
    clear_directory(removed_shapes_dir)

    all_images = os.listdir(folder_dir)

    # keras OCR
    pipline = keras_ocr.pipeline.Pipeline()

    # easyOCR
    text_reader = Reader(['sk'], gpu=False)

    # clear content of file
    # statistics_file = open('test/results/statistics/ocr_statistic.csv', 'w')
    # statistics_file.truncate()
    # statistics_file.close()

    # csv_header = ['image_name', 'id', 'easy_ocr', 'keras', 'tesseract']
    # writer.writerow(csv_header)
    # write_statistics_to_csv(csv_header)

    for image_name in all_images:
        path = folder_dir + '/' + image_name
        img = cv2.imread(path)
        print(image_name)

        digital_contours, detected_shapes = detect_shapes(img)

        removed_shapes, detected_lines = remove_shapes_from_image(img, detected_shapes)
        save_image(removed_shapes_dir, image_name, "", removed_shapes)

        # Keras OCR without statistics
        digital_contours = recognize_text_no_statistics(orig_img_dir, folder_dir, image_name, digital_contours, pipline, detected_shapes, False, True, False)

        # Keras OCR with statistics
        # digital_contours, statistic_data = recognize_text_with_statistics(orig_img_dir, folder_dir, image_name, digital_contours, pipline, text_reader, detected_shapes, 'keras')

        # EasyOCR
        # digital_contours = recognize_text_no_statistics(orig_img_dir, folder_dir, image_name, digital_contours, text_reader, detected_shapes, True, False, False)

        # Easy OCR with statistics
        # digital_contours, statistic_data = recognize_text_with_statistics(orig_img_dir, folder_dir, image_name, digital_contours, pipline, text_reader, detected_shapes, 'easy_ocr')

        # Tesseract OCR
        # digital_contours = recognize_text_no_statistics(orig_img_dir, folder_dir, image_name, digital_contours, None, detected_shapes, False, False, True)

        # Tesseract OCR with statistics
        # digital_contours, statistic_data = recognize_text_with_statistics(orig_img_dir, folder_dir, image_name, digital_contours, pipline, text_reader, detected_shapes, 'tesseract_ocr')

        save_image(digital_imgs_contour_dir, image_name, "", digital_contours)

        json_res = erd_data_to_json(detected_shapes, detected_lines)
        write_json_to_file(json_output_dir, json_res, image_name)


def get_new_image_size(orig_height, orig_width):
    new_longer_side_px = 1000

    if orig_width >= orig_height:
        new_width = new_longer_side_px
        new_height = (new_longer_side_px * orig_height) / orig_width
    else:
        new_height = new_longer_side_px
        new_width = (new_longer_side_px * orig_width) / orig_height

    return int(new_height), int(new_width)


def resize_all_images(source_dst, result_dir):
    all_images = os.listdir(source_dst)
    clear_directory(result_dir)

    for img_name in all_images:
        img_path = source_dst + '/' + img_name
        img = cv2.imread(img_path)

        orig_height, orig_width = img.shape[:2]
        new_height, new_width = get_new_image_size(orig_height, orig_width)

        # !!! cv2. resize has order of new values: (width, height)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        save_image(result_dir, img_name, '', resized_img)


def img_preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)

    bw_swap = cv2.bitwise_not(threshold)
    dilated = cv2.dilate(bw_swap, np.ones((3, 3), dtype=np.uint8))  # 2, 2

    result_img = dilated
    return result_img


def get_vectors_from_rect_points(box):
    reordered_box = reorder_rect_points_horizontal_rec(box)
    reordered_box = np.asarray(reordered_box)

    down_side_right_point = reordered_box[2]
    angle_point = reordered_box[3]

    vect_a = [angle_point[0], 0]
    vect_b_x = down_side_right_point[0] - angle_point[0]
    vect_b_y = down_side_right_point[1] - angle_point[1]
    vect_b = [vect_b_x, vect_b_y]

    return vect_a, vect_b


def angle_of_rectangle(rect):
    vect_a, vect_b = get_vectors_from_rect_points(rect)

    vect_a_x, vect_a_y = vect_a
    vect_b_x, vect_b_y = vect_b

    vect_a_length = math.sqrt(vect_a_x ** 2 + vect_a_y ** 2)
    vect_b_length = math.sqrt(vect_b_x ** 2 + vect_b_y ** 2)

    dot_product = (vect_a_x * vect_b_x) + (vect_a_y * vect_b_y)
    if dot_product == 0:
        return 90.0

    cos_angle = dot_product / (vect_a_length * vect_b_length)
    angle = math.degrees(math.acos(cos_angle))

    return round(angle, 2)


def get_vector(line_start, line_end):
    a_x, a_y = line_start
    b_x, b_y = line_end

    # smerovy vektor priamky AB
    s_x = b_x - a_x
    s_y = b_y - a_y

    # normalovy vektor kolmy na smerovy
    n_x = s_y
    n_y = -s_x

    c = (-n_x * a_x) - (n_y * a_y)

    return c, (n_x, n_y)


def distance_point_to_line(line_start, line_end, point):
    c, (n_x, n_y) = get_vector(line_start, line_end)
    point_x, point_y = point

    # vzdialenost bodu od priamky
    denominator = math.sqrt(n_x ** 2 + n_y ** 2)
    numerator = abs(n_x * point_x + n_y * point_y + c)

    dst_point_line = numerator / denominator

    return dst_point_line


def check_dst_point_to_line(selected_points, start_line, end_line, rightmost):
    distances = []
    summa = 0
    text_vect_length = dst_of_points(start_line, rightmost)

    for point in selected_points:
        point = point[0]

        dst_to_text_line = distance_point_to_line(start_line, rightmost, point)

        if dst_to_text_line / text_vect_length > 7 / text_vect_length:
            # cv2.circle(image_copy, point, 2, (0, 0, 255), -1)
            dst_to_line = distance_point_to_line(start_line, end_line, point)
            distances.append(dst_to_line)
            summa = summa + dst_to_line

    if len(distances) != 0:
        absolute_deviation = summa / len(distances)
    else:
        absolute_deviation = 10
    # print(absolute_deviation)

    return absolute_deviation


def find_shape_center(shape):
    M_shape = cv2.moments(shape)
    x_center = 0
    y_center = 0

    if M_shape['m00'] != 0.0:
        x_center = int(M_shape['m10'] / M_shape['m00'])
        y_center = int(M_shape['m01'] / M_shape['m00'])

    return x_center, y_center


def shape_inside_shape_test(shape_outer, shape_inner):
    for point in shape_inner.contour:
        point = point[0]
        point_x, point_y = point
        # print(point)
        inside = cv2.pointPolygonTest(shape_outer.contour, (int(point_x), int(point_y)), measureDist=False)
        # point is outside the polygon
        if inside < 0:
            return False

    # all points of inner shape are inside outer shape
    return True


def draw_shapes(img, shapes):
    for shape in shapes:
        if shape.shape_name == "triangle":
            cv2.drawContours(image=img, contours=[shape.contour], contourIdx=-1, color=(255, 0, 0), thickness=2,
                             lineType=cv2.LINE_AA)

        # else:
        #     cv2.drawContours(image=img, contours=[np.intp(cv2.boxPoints(shape.bounding_rectangle))], contourIdx=-1,
        #                      color=(0, 255, 0), thickness=2,
        #                      lineType=cv2.LINE_AA)
        #     cv2.ellipse(img, shape.bounding_ellipse, (0, 0, 255), 2)

        elif shape.shape_name == "rectangle":
            cv2.drawContours(image=img, contours=[np.intp(cv2.boxPoints(shape.bounding_rectangle))], contourIdx=-1,
                             color=(0, 255, 0), thickness=2,
                             lineType=cv2.LINE_AA)
        elif shape.shape_name == "ellipse":
            # cv2.drawContours(image=img, contours=[hull], contourIdx=-1, color=(0, 0, 255), thickness=2,
            #                  lineType=cv2.LINE_AA)
            cv2.ellipse(img, shape.bounding_ellipse, (0, 0, 255), 2)
        elif shape.shape_name == "diamond":
            cv2.drawContours(image=img, contours=[shape.contour], contourIdx=-1, color=(255, 0, 255), thickness=2,
                             lineType=cv2.LINE_AA)

    return img


def enlarge_contour(shape, img_size, hull):
    mask = np.ones(img_size, dtype="uint8") * 255
    cv2.drawContours(mask, [hull], -1, (0, 0, 0), -1)
    bw_swap = cv2.bitwise_not(mask)

    dilated = cv2.dilate(bw_swap, np.ones((25, 25), dtype=np.uint8))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape.set_enlarged_contour(contours[0])

    # cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=-2, lineType=cv2.LINE_AA)

    return mask


def find_closest_line(line, all_lines):
    for point in line.edge_points:
        point = point[0]
        min_dst = 100000
        closest_line = None
        for next_line in all_lines:
            if next_line != line:
                for next_point in next_line.edge_points:
                    next_point = next_point[0]

                    current_dst = dst_of_points(point, next_point)

                    if current_dst < min_dst:
                        min_dst = current_dst
                        closest_line = next_line

        if min_dst < 8:
            # cv2.line(img, closest_point, point, (0, 255, 255), 2)
            return closest_line

    return None


def connect_lines(all_lines):
    cl_index = 0

    while cl_index < len(all_lines):
        A = all_lines[cl_index]
        B = find_closest_line(A, all_lines)

        if B is not None:
            new_line_contour = np.concatenate((A.contour, B.contour), axis=0)
            new_edge_points = np.concatenate((A.edge_points, B.edge_points), axis=0)

            all_lines[cl_index].set_contour(new_line_contour)
            all_lines[cl_index].set_edge_points(new_edge_points)

            all_lines.remove(B)

        else:
            cl_index = cl_index + 1
            continue

    return all_lines


def match_shapes(img, shapes, lines):
    for line in lines:
        for shape in shapes:
            shape_centre = find_shape_center(shape.enlarged_contour)
            shape.set_shape_centre(shape_centre)
            cv2.circle(img, shape_centre, 4, (0, 0, 0), -1)
            for point in line.edge_points:
                point = point[0]
                point_x, point_y = point

                inside = cv2.pointPolygonTest(shape.enlarged_contour, (int(point_x), int(point_y)), measureDist=False)

                if inside >= 0:
                    # cv2.line(img, shape_centre, point, line.color, 1)
                    if shape not in line.connecting_shapes:
                        line.connecting_shapes.append(shape)
                        cv2.line(img, shape_centre, point, line.color, 1)

    return img


def draw_lines(img, all_lines):
    for line in all_lines:
        cv2.drawContours(image=img, contours=[line.contour], contourIdx=-1, color=line.color, thickness=2,
                         lineType=cv2.LINE_AA)

        # cv2.polylines(img, [line.contour], False, line.color, 2)

        # for point in line.edge_points:
        #     point = point[0]
        #     cv2.circle(img, point, 4, (0, 0, 255), -1)

    return img


def detect_lines(img, shapes):
    all_lines = []
    img_copy = img.copy()

    # vykreslenie dilatovanych tvarov
    # for shape in shapes:
    #     cv2.drawContours(image=img_copy, contours=[shape.enlarged_contour], contourIdx=-1, color=(0, 0, 0), thickness=2,
    #                      lineType=cv2.LINE_AA)

    dilated = img_preprocessing(img)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        color = random_color()
        # color = (255, 0, 0)

        # cv2.drawContours(image=img_copy, contours=[cnt], contourIdx=-1, color=color, thickness=-2, lineType=cv2.LINE_AA)

        peri = cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, False)

        line = Line(cnt, approx, color)
        all_lines.append(line)

        # cv2.drawContours(image=img_copy, contours=[approx], contourIdx=-1, color=color, thickness=-2, lineType=cv2.LINE_AA)

        # vykreslenie bodov ciary
        # for point in approx:
        #     point = point[0]
        #     cv2.circle(img_copy, point, 4, (0, 0, 255), -1)

    all_lines = connect_lines(all_lines)
    img_copy = draw_lines(img_copy, all_lines)
    img_copy = match_shapes(img_copy, shapes, all_lines)

    return img_copy, all_lines


def remove_shapes_from_image(img, shapes):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    deleted_shapes_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_size = img.shape[:2]
    mask = np.ones(img_size, dtype="uint8") * 255

    for shape in shapes:
        hull = cv2.convexHull(shape.contour, False)
        shape.set_convex_hull(hull)
        cv2.drawContours(mask, [hull], -1, (0, 0, 0), -1)
        enlarge_contour(shape, img_size, hull)

    # removal based on contour: dilate contour, detect again and draw white
    bw_swap = cv2.bitwise_not(mask)

    dilated = cv2.dilate(bw_swap, np.ones((13, 13), dtype=np.uint8))
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        color = (255, 255, 255)
        cv2.drawContours(image=deleted_shapes_img, contours=[cnt], contourIdx=-1, color=color, thickness=-2,
                         lineType=cv2.LINE_AA)

    img, detected_lines = detect_lines(deleted_shapes_img, shapes)
    return img, detected_lines


def remove_nested_shapes(all_shapes):
    cleared_shapes = []

    for tested_inner_shape in all_shapes:
        tested_inner_shape_is_inner = False

        for tested_outer_shape in all_shapes:

            if tested_inner_shape == tested_outer_shape:
                continue

            tested_inner_shape_in_tested_outer_shape_result = shape_inside_shape_test(tested_outer_shape,
                                                                                      tested_inner_shape)

            if tested_inner_shape_in_tested_outer_shape_result:
                tested_inner_shape_is_inner = True
                break

        if not tested_inner_shape_is_inner:
            cleared_shapes.append(tested_inner_shape)

    return cleared_shapes


def detect_shapes(img):
    all_shapes = []
    image_copy = img.copy()
    dilated = img_preprocessing(img)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    for component in zip(contours, hierarchy):
        cnt = component[0]
        cnt_hierarchy = component[1]
        cnt_area = cv2.contourArea(cnt)
        # color = random_color()
        # cv2.drawContours(image=image_copy, contours=[cnt], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)

        if cnt_area < 400:
            continue

        approx = cv2.approxPolyDP(cnt, 0.025 * cv2.arcLength(cnt, True), True)

        if len(approx) == 3:
            x_cnt, y_cnt = find_shape_center(cnt)
            x_approx, y_approx = find_shape_center(approx)

            x_diff = abs(x_cnt - x_approx)
            y_diff = abs(y_cnt - y_approx)
            if x_diff < 5 and y_diff < 5:
                rect = cv2.minAreaRect(cnt)

                triangle_shape = Shape(cnt, cnt_hierarchy, "triangle", rect)
                all_shapes.append(triangle_shape)
                # cv2.drawContours(image=image_copy, contours=[cnt], contourIdx=-1, color=(255, 0, 0), thickness=2,
                #                  lineType=cv2.LINE_AA)

        else:
            rect = cv2.minAreaRect(cnt)
            rect_width = rect[1][0]
            rect_heigh = rect[1][1]
            rect_area = rect_width * rect_heigh

            box = cv2.boxPoints(rect)
            box = np.intp(box)

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                ellipse_area = math.pi * MA / 2 * ma / 2

                cnt_rect_diff = rect_area - cnt_area
                cnt_ellipse_diff = ellipse_area - cnt_area

                # angle_of_rect_rotation = rect[2]
                angle_of_rect_rotation = angle_of_rectangle(box)

                # if ellipse_area < 10000:
                #     cv2.drawContours(image=image_copy, contours=[box], contourIdx=-1, color=(0, 255, 0), thickness=2,lineType=cv2.LINE_AA)
                #     cv2.ellipse(image_copy, ellipse, (0, 0, 255), 2)

                if cnt_rect_diff < cnt_ellipse_diff and cnt_rect_diff <= 900 and rect_area >= 500:
                    # cv2.drawContours(image=image_copy, contours=[box], contourIdx=-1, color=(0, 255, 0), thickness=2,
                    #                  lineType=cv2.LINE_AA)
                    if angle_of_rect_rotation > 10:
                        diamond_shape = Shape(cnt, cnt_hierarchy, "diamond", rect)
                        diamond_shape.set_bounding_ellipse(ellipse)
                        all_shapes.append(diamond_shape)
                        # cv2.drawContours(image=image_copy, contours=[box], contourIdx=-1, color=(255, 0, 255),
                        #                  thickness=2, lineType=cv2.LINE_AA)
                    else:
                        rectangle_shape = Shape(cnt, cnt_hierarchy, "rectangle", rect)
                        rectangle_shape.set_bounding_ellipse(ellipse)
                        all_shapes.append(rectangle_shape)
                        # cv2.drawContours(image=image_copy, contours=[box], contourIdx=-1, color=(0, 255, 0),
                        #                  thickness=2, lineType=cv2.LINE_AA)

                elif cnt_rect_diff > cnt_ellipse_diff and cnt_ellipse_diff <= 700 and ellipse_area >= 500:
                    # cv2.ellipse(image_copy, ellipse, (0, 0, 255), 2)

                    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

                    # cv2.circle(image_copy, leftmost, 5, (255, 0, 0), -1)
                    # cv2.circle(image_copy, rightmost, 5, (255, 0, 0), -1)
                    # cv2.circle(image_copy, topmost, 5, (255, 0, 0), -1)
                    # cv2.circle(image_copy, bottommost, 5, (255, 0, 0), -1)
                    # cv2.line(image_copy, leftmost, rightmost, (0, 0, 255), 1)

                    left_x, left_y = leftmost
                    top_x, top_y = topmost
                    bottom_x, bottom_y = bottommost

                    selected_upper = cnt[(left_y >= cnt[:, :, 1]) & (cnt[:, :, 0] <= top_x)]
                    shape1, shape3 = selected_upper.shape
                    selected_upper = selected_upper.reshape(shape1, 1, shape3)

                    selected_lower = cnt[(left_y <= cnt[:, :, 1]) & (cnt[:, :, 0] <= bottom_x)]
                    shape1, shape3 = selected_lower.shape
                    selected_lower = selected_lower.reshape(shape1, 1, shape3)

                    # # print(selected)
                    # cv2.drawContours(image=image_copy, contours=selected_lower, contourIdx=-1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    # cv2.drawContours(image=image_copy, contours=selected_upper, contourIdx=-1, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                    absolute_deviation_upper = check_dst_point_to_line(selected_upper, leftmost, topmost, rightmost)
                    absolute_deviation_lower = check_dst_point_to_line(selected_lower, leftmost, bottommost, rightmost)

                    upper_side_length = dst_of_points(leftmost, topmost)
                    lower_side_length = dst_of_points(leftmost, bottommost)

                    if upper_side_length > 0 and lower_side_length > 0:
                        if (absolute_deviation_upper / upper_side_length) < (1 / upper_side_length) or (
                                absolute_deviation_lower / lower_side_length) < (1 / lower_side_length):
                            diamond_shape = Shape(cnt, cnt_hierarchy, "diamond", rect)
                            diamond_shape.set_bounding_ellipse(ellipse)
                            all_shapes.append(diamond_shape)

                            # cv2.drawContours(image=image_copy, contours=[cnt], contourIdx=-1, color=(255, 0, 255),
                            #                  thickness=2, lineType=cv2.LINE_AA)

                            # x_center, y_center = find_shape_center(cnt)
                            # cv2.circle(image_copy, (x_center, y_center), 5, (255, 255, 51), -1)
                            #
                            # cv2.circle(image_copy, leftmost, 5, (0, 0, 0), -1)
                            # cv2.circle(image_copy, rightmost, 5, (0, 255, 0), -1)
                            # cv2.circle(image_copy, topmost, 5, (255, 0, 0), -1)
                            # cv2.circle(image_copy, bottommost, 5, (0, 255, 255), -1)

                        else:
                            ellipse_shape = Shape(cnt, cnt_hierarchy, "ellipse", rect)
                            ellipse_shape.set_bounding_ellipse(ellipse)
                            all_shapes.append(ellipse_shape)
                            # cv2.ellipse(image_copy, ellipse, (0, 0, 255), 2)
                            # cv2.drawContours(image=image_copy, contours=[box], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    all_shapes = remove_nested_shapes(all_shapes)
    image_copy = draw_shapes(image_copy, all_shapes)

    return image_copy, all_shapes


def order_points_new(points):
    # sort the points based on their x-coordinates
    x_sorted = points[np.argsort(points[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="intp")


def point_in_original_image(orig_img_dir, resized_img_dir, reordered_points, img_name):
    orig_img_path = orig_img_dir + '/' + img_name
    resized_img_path = resized_img_dir + '/' + img_name

    orig_img = cv2.imread(orig_img_path)
    resized_img = cv2.imread(resized_img_path)

    orig_height, orig_width = orig_img.shape[:2]
    resized_height, resized_width = resized_img.shape[:2]

    resize_ratio_height = orig_height / resized_height
    resize_ratio_width = orig_width / resized_width

    reordered_orig_points = []
    for point in reordered_points:
        x_resized, y_resized = point
        x_orig = x_resized * resize_ratio_width
        x_orig = np.intp(x_orig)
        y_orig = y_resized * resize_ratio_height
        y_orig = np.intp(y_orig)

        reordered_orig_points.append((x_orig, y_orig))

    return orig_img, reordered_orig_points


def recognize_text_no_statistics(orig_img_path, resized_img_path, img_name, img, recognizer, shapes, easy_ocr, keras, tesseract_ocr):
    black = (0, 0, 0)
    unicode_font = ImageFont.truetype("fonts/DejaVuSans.ttf", 15)

    for shape in shapes:
        shape_text = ""

        bbox = shape.bounding_rectangle
        bbox_points = cv2.boxPoints(bbox)
        bbox_points = np.intp(bbox_points)

        reordered_points = order_points_new(bbox_points)

        if (reordered_points < 0).any():
            reordered_points[reordered_points < 0] = 0

        top_left, top_right, bottom_right, bottom_left = reordered_points

        # OCR on original image
        orig_img, reordered_orig_points = point_in_original_image(orig_img_path, resized_img_path, reordered_points, img_name)
        top_left_orig, top_right_orig, bottom_right_orig, bottom_left_orig = reordered_orig_points

        if top_left_orig[1] <= top_right_orig[1]:
            shape_img_slice = orig_img[top_left_orig[1]: bottom_right_orig[1], bottom_left_orig[0]: top_right_orig[0]]
        else:
            shape_img_slice = orig_img[top_right_orig[1]: bottom_left_orig[1], top_left_orig[0]: bottom_right_orig[0]]

        if easy_ocr:
            results = recognizer.readtext(shape_img_slice)

            for (bbox, text, prob) in results:
                print(text)
                shape_text = shape_text + text + " "

        elif keras:
            shape_img_slice = [shape_img_slice]
            try:
                pred = recognizer.recognize(shape_img_slice)
                pred_res = pred[0]

                for text, box in pred_res:
                    print(text)
                    shape_text = shape_text + text + " "

            except:
                print("ERROR")
                shape_text = "ERROR"

        elif tesseract_ocr:
            gray = cv2.cvtColor(shape_img_slice, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (3, 3), 0)
            # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # # Morph open to remove noise and invert image
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            # invert = 255 - opening

            shape_text = pytesseract.image_to_string(gray, lang='slk', config='--tessdata-dir tessdata --psm 11')
            print(shape_text)

        pil_image = Image.fromarray(img)

        draw = ImageDraw.Draw(pil_image)
        text_position = [bottom_left[0] + 5, bottom_left[1] + 5]
        draw.text(text_position, shape_text, font=unicode_font, fill=black)

        shape.set_text(shape_text)

        img = np.array(pil_image)

    return img


def recognize_text_with_statistics(orig_img_path, resized_img_path, img_name, img, recognizer_keras, recognizer_easy_ocr, shapes, result_model):
    black = (0, 0, 0)
    unicode_font = ImageFont.truetype("fonts/DejaVuSans.ttf", 15)

    for shape in shapes:
        shape_centre_id = 'x' + str(shape.shape_centre[0]) + 'y' + str(shape.shape_centre[1])

        statistic_data = []
        statistic_data.extend((img_name, shape_centre_id))

        bbox = shape.bounding_rectangle
        bbox_points = cv2.boxPoints(bbox)
        bbox_points = np.intp(bbox_points)

        reordered_points = order_points_new(bbox_points)

        if (reordered_points < 0).any():
            reordered_points[reordered_points < 0] = 0

        top_left, top_right, bottom_right, bottom_left = reordered_points

        # OCR on original image
        orig_img, reordered_orig_points = point_in_original_image(orig_img_path, resized_img_path, reordered_points, img_name)
        top_left_orig, top_right_orig, bottom_right_orig, bottom_left_orig = reordered_orig_points

        if top_left_orig[1] <= top_right_orig[1]:
            shape_img_slice = orig_img[top_left_orig[1]: bottom_right_orig[1], bottom_left_orig[0]: top_right_orig[0]]
        else:
            shape_img_slice = orig_img[top_right_orig[1]: bottom_left_orig[1], top_left_orig[0]: bottom_right_orig[0]]

        # text recognition
        # easy OCR
        easy_ocr_text = ""
        easy_ocr_results = recognizer_easy_ocr.readtext(shape_img_slice)
        for (bbox, text, prob) in easy_ocr_results:
            # print(text)
            easy_ocr_text = easy_ocr_text + text + " "

        # Keras
        keras_text = ""
        keras_shape_img_slice = [shape_img_slice]
        try:
            pred = recognizer_keras.recognize(keras_shape_img_slice)
            pred_res = pred[0]

            for text, box in pred_res:
                # print(text)
                keras_text = keras_text + text + " "

        except:
            # print("ERROR")
            keras_text = ""

        # Tesseract
        tesseract_text = ""
        gray = cv2.cvtColor(shape_img_slice, cv2.COLOR_BGR2GRAY)
        tesseract_text = pytesseract.image_to_string(gray, lang='slk', config='--tessdata-dir tessdata')
        # print(shape_text)

        # draw text to img
        pil_image = Image.fromarray(img)

        draw = ImageDraw.Draw(pil_image)
        text_position = [bottom_left[0] + 5, bottom_left[1] + 5]

        text_to_img = shape_centre_id + ": "
        if result_model == 'easy_ocr':
            text_to_img = text_to_img + easy_ocr_text
            shape.set_text(easy_ocr_text)
        elif result_model == 'keras':
            text_to_img = text_to_img + keras_text
            shape.set_text(keras_text)
        elif result_model == 'tesseract_ocr':
            text_to_img = text_to_img + tesseract_text
            shape.set_text(tesseract_text)

        draw.text(text_position, text_to_img, font=unicode_font, fill=black)
        img = np.array(pil_image)

        statistic_data.extend((easy_ocr_text, keras_text, tesseract_text))
        write_statistics_to_csv(statistic_data)

    return img, statistic_data


def erd_data_to_json(all_shapes, all_lines):
    mapping = {}
    result = ImageResult()

    entity_id_counter = 1
    attribute_id_counter = 1
    relationship_id_counter = 1
    generalization_id_counter = 1

    for shape in all_shapes:
        match shape.shape_name:
            case "rectangle":
                shape_id = "E" + str(entity_id_counter)
                entity_id_counter = entity_id_counter + 1

            case "ellipse":
                shape_id = "A" + str(attribute_id_counter)
                attribute_id_counter = attribute_id_counter + 1

            case "diamond":
                shape_id = "R" + str(relationship_id_counter)
                relationship_id_counter = relationship_id_counter + 1

            case "triangle":
                shape_id = "G" + str(generalization_id_counter)
                generalization_id_counter = generalization_id_counter + 1

            case _:
                shape_id = "XXX"

        dto_shape = shape.to_dto(shape_id)

        mapping[shape] = dto_shape

        result.add_object(dto_shape.to_json())

    for line in all_lines:
        connection = []
        for shape in line.connecting_shapes:
            corresponding_dto_shape = mapping[shape]
            connection.append(corresponding_dto_shape.ID)

        if connection:
            result.add_connection(connection)

    return result.to_json()


def write_json_to_file(json_output_dir, json_data, name):
    start = len(name) - 6
    exten_index = name.find('.', start)
    name_no_extension = name[:exten_index]

    file_name = json_output_dir + '/' + name_no_extension + ".json"
    file = open(file_name, 'w+')
    file.write(json.dumps(json_data, indent=4))
    file.close()


def write_statistics_to_csv(row):
    with open('results/statistics/ocr_statistic.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def clear_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def copy_columns_csv():
    data = pandas.read_csv("results/statistics/ocr_statistic.csv")

    img_names = data['image_name'].tolist()
    ids = data['id'].tolist()

    header = ['image_name', 'id', 'original_text']
    csv_columns = list(zip(img_names, ids))

    with open('results/statistics/original_texts.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_columns)


def original_text_modify():
    data = pandas.read_csv("results/statistics/original_texts.csv")
    data = data.fillna('')
    data['lowercase'] = data['original_text'].str.lower()
    data['diakritika'] = data['lowercase'].map(lambda x: strip_accents(x))
    print(data.head())
    data.to_csv("test/results/statistics/original_texts.csv", index=False)


def strip_accents(s):
    print(s)
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def csv_work():
    original_texts = pandas.read_csv("results/statistics/original_texts.csv", nrows=572)
    print(original_texts.head())
    original_texts['count'] = original_texts['original_text'].str.len()
    print(original_texts.head())
    total = original_texts['count'].sum()
    print("Total: ", total)

    # ocr_texts = pandas.read_csv("results/statistics/ocr_statistic.csv", nrows=572)
    # ocr_texts = ocr_texts.fillna('')
    # cer_value = cer(original_texts['original_text'].tolist(), ocr_texts['easy_ocr'].tolist())
    # print(cer_value)
    # print(original_texts)
    # print(ocr_texts)

    # data = pandas.read_csv("test/results/statistics/shapes.csv")
    # names = pandas.DataFrame(data.image_name.unique(), columns=['image_name'])
    # names.to_csv("test/results/statistics/shapes.csv", index=False)
    # print(names.head())

    # total_correct = data['spravne'].sum()
    # not_found = data['nenajdene'].sum()
    # not_classified = data['nespravne_klasifikovane'].sum()
    # extra = data['navyse'].sum()
    # print("correct: ", total_correct)
    # print("not found: ", not_found)
    # print("not classified: ", not_classified)
    # print("extra: ", extra)
    # print("prerusovane: ", data['prerusovane'].sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digitalization of ERD app. '
                                                 'If any of the following path is not given, default will be used.'
                                                 ' It is located in subdirectory /demo in root directory of project.'
                                                 ' HTML file is located in root directory of project.')
    parser.add_argument('--demo', action="store_true", help="run demo app")
    parser.add_argument('--imgs-path', help="path to folder containing input images")
    parser.add_argument('--resized-imgs-path', help="path to folder where resized images will be stored")
    parser.add_argument('--shapes-path', help="path to folder where images with detected shapes will be stored")
    parser.add_argument('--lines-path', help="path to folder where images with line detection results will be stored")
    parser.add_argument('--json-path', help="path to folder where json files will be stored")
    # # Read arguments from command line
    args = parser.parse_args()
    # print("args: ", args)

    default_input = 'demo/test_imgs'
    default_resized = 'demo/resized_imgs'
    default_shapes = 'demo/detected_shapes'
    default_lines = 'demo/detected_lines'
    default_json = 'results/json_outputs'

    if args.demo:
        # print('demo true')
        resize_all_images(default_input, default_resized)
        get_all_images(default_input, default_resized, default_shapes, default_lines, default_json)
        digital_images_results.show_results_html(default_resized, default_shapes, default_lines)
    else:
        if args.imgs_path is not None:
            images_dir = args.imgs_path
            if not (os.path.exists(images_dir)) and not (os.path.isdir(images_dir)):
                print("Invalid path to images")
                raise SystemExit(1)
        else:
            images_dir = default_input

        if args.resized_imgs_path is not None:
            resized_dir = args.resized_imgs_path
            if not (os.path.exists(resized_dir)) and not (os.path.isdir(resized_dir)):
                print("Invalid path to resized images directory")
                raise SystemExit(1)
        else:
            resized_dir = default_resized

        if args.shapes_path is not None:
            shapes_dir = args.shapes_path
            if not (os.path.exists(shapes_dir)) and not (os.path.isdir(shapes_dir)):
                print("Invalid path to detected shapes directory")
                raise SystemExit(1)
        else:
            shapes_dir = default_shapes

        if args.lines_path is not None:
            lines_dir = args.lines_path
            if not (os.path.exists(lines_dir)) and not (os.path.isdir(lines_dir)):
                print("Invalid path to detected lines directory")
                raise SystemExit(1)
        else:
            lines_dir = default_lines

        if args.json_path is not None:
            json_dir = args.json_path
            if not (os.path.exists(json_dir)) and not (os.path.isdir(json_dir)):
                print("Invalid path to JSON output directory")
                raise SystemExit(1)
        else:
            json_dir = default_json

        resize_all_images(images_dir, resized_dir)
        get_all_images(images_dir, resized_dir, shapes_dir, lines_dir, json_dir)
        digital_images_results.show_results_html(resized_dir, shapes_dir, lines_dir)
