import copy
import math
import os
import random
import webbrowser

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

import digital_images_results
from output_lines_by_hist_script import lines_by_hist_html
from outputs import showResultsHTML
from numpy import linalg as LA


def identical_lists(l1, l2):
    pass


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
        # print("orig: ", orig_line, " paralel with cmp: ", cmp_line, " dst strat: ", dst_l1_start, " dst end: ", dst_l1_end )
        return True
    elif (dst_l2_start <= diff) and (dst_l2_end <= diff):
        # print("orig: ", orig_line, " paralel with cmp: ", cmp_line, " dst strat: ", dst_l2_start, " dst end: ", dst_l2_end )
        return True
    else:
        # print("orig: ", orig_line, " NOT paralel with cmp: ", cmp_line, " dst strat: ", dst_l2_start, " dst end: ", dst_l1_end )
        return False


def filterLines(all_lines):
    filtered_lines = []
    # longest_paralel = copy.deepcopy(all_lines[0])
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
                            # filtered_lines.remove(cmp_line)
                            length_longes = lineLength(longest_paralel)
                            if length_orig > length_longes:
                                longest_paralel = orig_line
                        else:
                            all_lines[i][0] = -(all_lines[i][0])
                            # filtered_lines.remove(orig_line)
                            length_longes = lineLength(longest_paralel)
                            if length_cmp > length_longes:
                                longest_paralel = cmp_line

            if longest_paralel not in filtered_lines:
                # if not(longest_paralel.isin(filtered_lines)):
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
        dst_point_to_line = math.sqrt(x * x + y * y)
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


def random_color():
    b = random.randint(0, 255)
    r = random.randint(0, 255)
    g = random.randint(0, 255)

    return b, g, r


def drawLines(img_copy, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            b = random.randint(0, 255)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            cv2.line(img_copy, (x1, y1), (x2, y2), (b, g, r), 2)

    return img_copy


def draw_rectangles(draw_img, rect_points, horizontal):
    if rect_points is not None:
        for rect in rect_points:
            if horizontal:
                cv2.drawContours(draw_img, [rect], 0, (0, 255, 0), 2)
            else:
                cv2.drawContours(draw_img, [rect], 0, (0, 0, 255), 2)

    return draw_img


def draw_connected_middle_points_max_length_vertical(draw_img, closest_rect, max_dst):
    if closest_rect is None:
        return draw_img

    radius = 2
    color_start = (255, 0, 0)
    thickness = 2

    for start_rec, end_rec in closest_rect.items():
        start_rec_left_upper = start_rec[0]
        start_rec_right_upper = start_rec[1]

        end_rec_left_lower = end_rec[0][3]
        end_rec_right_lower = end_rec[0][2]

        start_point = get_middle_point_of_side(start_rec_left_upper, start_rec_right_upper)
        end_point = get_middle_point_of_side(end_rec_left_lower, end_rec_right_lower)


        # end_rec[1] = upravena metrika
        # end_rec[2] = realna metrika
        dst = end_rec[2]

        if dst < max_dst:
            draw_img = cv2.circle(draw_img, start_point, radius, color_start, thickness)
            draw_img = cv2.circle(draw_img, end_point, radius, (255, 51, 255), thickness)
            draw_img = cv2.line(draw_img, start_point, end_point, (255, 255, 0), thickness)

    return draw_img


def draw_connected_middle_points_max_length_horizontal(draw_img, closest_rect, max_dst):
    if closest_rect is None:
        return draw_img

    radius = 2
    color_start = (255, 0, 0)
    thickness = 2

    for start_rec, end_rec in closest_rect.items():
        start_rec_right_upper = start_rec[1]
        start_rec_right_lower = start_rec[2]

        end_rec_left_upper = end_rec[0][0]
        end_rec_left_lower = end_rec[0][3]

        start_point = get_middle_point_of_side(start_rec_right_upper, start_rec_right_lower)
        end_point = get_middle_point_of_side(end_rec_left_upper, end_rec_left_lower)


        # end_rec[1] = upravena metrika
        # end_rec[2] = realna metrika
        dst = end_rec[2]

        if dst < max_dst:
            draw_img = cv2.circle(draw_img, start_point, radius, color_start, thickness)
            draw_img = cv2.circle(draw_img, end_point, radius, (255, 51, 255), thickness)
            draw_img = cv2.line(draw_img, start_point, end_point, (255, 51, 255), thickness)

    return draw_img

def draw_connected_middle_points_closest_horizontal_vertical(draw_img, closest_rect):
    if closest_rect is None:
        return draw_img
    radius = 2
    color_start = (255, 0, 0)
    thickness = 2

    for start_rec, end_rec in closest_rect.items():
        start_rec_right_upper = start_rec[1]
        start_rec_right_lower = start_rec[2]

        end_rec_upper_left = end_rec[0][0]
        end_rec_upper_right = end_rec[0][1]

        start_point = get_middle_point_of_side(start_rec_right_upper, start_rec_right_lower)
        end_point = get_middle_point_of_side(end_rec_upper_left, end_rec_upper_right)

        # color = random_color()

        draw_img = cv2.circle(draw_img, start_point, radius, color_start, thickness)
        if start_point[0] < end_point[0]:
            draw_img = cv2.circle(draw_img, end_point, radius, color_start, thickness)
            draw_img = cv2.line(draw_img, start_point, end_point, color_start, thickness)

        draw_img = cv2.circle(draw_img, end_point, radius, (51, 255, 255), thickness)
        draw_img = cv2.line(draw_img, start_point, end_point, (51, 255, 255), thickness)

        # cv2.imshow("connected", draw_img)

    return draw_img


def draw_connected_middle_points_closest_vertical(draw_img, closest_rect):
    radius = 2
    color_start = (255, 0, 0)
    thickness = 2

    for start_rec, end_rec in closest_rect.items():
        start_rec_upper_left = start_rec[0]
        start_rec_upper_right = start_rec[1]

        end_rec_lower_left = end_rec[0][3]
        end_rec_lower_right = end_rec[0][2]

        start_point = get_middle_point_of_side(start_rec_upper_left, start_rec_upper_right)
        end_point = get_middle_point_of_side(end_rec_lower_left, end_rec_lower_right)

        # color = random_color()

        draw_img = cv2.circle(draw_img, start_point, radius, color_start, thickness)
        if start_point[1] > end_point[1]:
            draw_img = cv2.circle(draw_img, end_point, radius, color_start, thickness)
            draw_img = cv2.line(draw_img, start_point, end_point, color_start, thickness)

        draw_img = cv2.circle(draw_img, end_point, radius, (255, 51, 255), thickness)
        draw_img = cv2.line(draw_img, start_point, end_point, (255, 51, 255), thickness)

        # cv2.imshow("connected", draw_img)

    return draw_img


def draw_connected_middle_points_closest_horizontal(draw_img, closest_rect):
    if closest_rect is None:
        return draw_img

    radius = 2
    color_start = (255, 0, 0)
    thickness = 2

    for start_rec, end_rec in closest_rect.items():
        start_rec_right_upper = start_rec[1]
        start_rec_right_lower = start_rec[2]

        end_rec_left_upper = end_rec[0][0]
        end_rec_left_lower = end_rec[0][3]

        start_point = get_middle_point_of_side(start_rec_right_upper, start_rec_right_lower)
        end_point = get_middle_point_of_side(end_rec_left_upper, end_rec_left_lower)

        # color = random_color()

        draw_img = cv2.circle(draw_img, start_point, radius, color_start, thickness)
        # if start_point[0] < end_point[0]:
        #     draw_img = cv2.circle(draw_img, end_point, radius, color_start, thickness)
        #     draw_img = cv2.line(draw_img, start_point, end_point, color_start, thickness)

        draw_img = cv2.circle(draw_img, end_point, radius, (255, 51, 255), thickness)
        draw_img = cv2.line(draw_img, start_point, end_point, (255, 51, 255), thickness)

        # cv2.imshow("connected", draw_img)

    return draw_img


def draw_connected_middle_points_closest_horizontal_histogram_color(draw_img, closest_rect, color, bin_start, bin_end):
    radius = 2
    thickness = 2

    for start_rec, end_rec in closest_rect.items():
        if bin_start <= end_rec[1] < bin_end:
            start_rec_right_upper = start_rec[1]
            start_rec_right_lower = start_rec[2]

            end_rec_left_upper = end_rec[0][0]
            end_rec_left_lower = end_rec[0][3]

            start_point = get_middle_point_of_side(start_rec_right_upper, start_rec_right_lower)
            end_point = get_middle_point_of_side(end_rec_left_upper, end_rec_left_lower)

            draw_img = cv2.circle(draw_img, start_point, radius, color, thickness)
            # if start_point[0] < end_point[0]:
            draw_img = cv2.circle(draw_img, end_point, radius, color, thickness)
            draw_img = cv2.line(draw_img, start_point, end_point, color, thickness)

    #cv2.imshow("connected", draw_img)

    return draw_img

def draw_connected_middle_points_histogram_colors(draw_img, closest_rect, colors, bins, bin_width):
    # rgb
    # bgr
    all_colors = {
        'aqua': (255, 255, 0),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'orange': (0, 128, 255),
        'green': (0, 128, 0),
        'purple': (128, 0, 128),
        'maroon': (0, 0, 128),
        'yellow': (0, 255, 255),
        'lime': (0, 255, 0)
    }

    if colors is None:
        draw_connected_middle_points_closest_horizontal(draw_img, closest_rect)

    if bins is None:
        return draw_img

    for i in range(len(bins)):
        bin_start = bins[i]
        bin_end = bin_start + bin_width
        color = colors[i]
        bgr_color = all_colors[color]
        # print(bin_start, " ", bin_end, " ", color)
        draw_img = draw_connected_middle_points_closest_horizontal_histogram_color(draw_img, closest_rect, bgr_color, bin_start, bin_end)

    return draw_img


def clear_hist_data(counts, bins, bin_width):
    cleared_bins = []

    for i in range(len(counts)):
        if counts[i] > 0:
            cleared_bins.append(bins[i])

    # last_value = cleared_bins[len(cleared_bins) - 1] + bin_width + 1
    # cleared_bins.insert(len(cleared_bins), last_value)
    #print(cleared_bins)
    return cleared_bins

# larger m => less outliers removed
def reject_outliers(data, m=6.):
    data = np.array(data)

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m].tolist()


def histogram_closest_distances(rect_hist_closest_dst_dir, closest_rectangles, img_name):
    #rect_hist_closest_dst_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_rect_hist_area"

    distances = []
    used_colors = None
    cleared_bins = None

    for value in closest_rectangles.values():
        # value[1] = upravena metrika, [2] realna metrika
        dst = value[2]
        distances.append(dst)

    binwidth = 5
    fig, ax = plt.subplots(figsize=(16, 4), facecolor='w')
    plt.ylabel('Frequency')

    # cnts = number of samples in each bin
    # values = lower bounds of bins
    # bars = rectangle definition of each histogram bar
    if len(distances) > 0:
        distances = reject_outliers(distances)

        # cnts, values, bars = ax.hist(distances, edgecolor='k',
        #                              bins=np.arange(min(distances), max(distances) + binwidth, binwidth))

        cnts, values, bars = ax.hist(distances, edgecolor='k', bins='auto')
        binwidth = values[1] - values[0]
        plt.xlabel(f'bin width: {binwidth: .3f}, min_dst: {min(distances): .3f}, max_dst: {max(distances): .3f}')

        number_of_bars = np.count_nonzero(cnts)

        colors = ['aqua', 'red', 'blue', 'orange', 'green', 'purple', 'maroon', 'yellow', 'lime']

        used_colors = []
        for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
            bar.set_facecolor(colors[i % len(colors)])
            if cnt > 0:
                used_colors.append(colors[i % len(colors)])

        cleared_bins = clear_hist_data(cnts, values, binwidth)

    else:
        plt.plot([])

    name = getResultName(img_name, "hist_dst")
    save_dst = rect_hist_closest_dst_dir + '/' + name

    all_images = os.listdir(rect_hist_closest_dst_dir)
    if name in all_images:
        os.remove(save_dst)

    plt.savefig(save_dst)
    plt.close(fig)

    #plt.show()

    return used_colors, cleared_bins, binwidth


def plot_histogram_angle():
    pass


def plot_histogram(horizontal_rect_box, vertical_rect_box, img_name):
    rect_hist_all_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_rect_hist_all"

    longer_side = []

    for rectangle in horizontal_rect_box:
        rectangle_size = rectangle[1]
        rec_width = rectangle_size[0]
        rec_height = rectangle_size[1]
        if rec_width > rec_height:
            longer_side.append(rec_width)
        elif rec_width == rec_height:
            longer_side.append(rec_width)
            longer_side.append(rec_height)
        else:
            longer_side.append(rec_height)

    for rectangle in vertical_rect_box:
        rectangle_size = rectangle[1]
        rec_width = rectangle_size[0]
        rec_height = rectangle_size[1]
        if rec_width > rec_height:
            longer_side.append(rec_width)
        elif rec_width == rec_height:
            longer_side.append(rec_width)
            longer_side.append(rec_height)
        else:
            longer_side.append(rec_height)

    # longer_side = reject_outliers(longer_side)

    if len(horizontal_rect_box) > 0 or len(vertical_rect_box) > 0:
        binwidth = 3
        n, bins, _ = plt.hist(longer_side, bins=np.arange(min(longer_side), max(longer_side) + binwidth, binwidth))

        plt.xlabel(f'min_len: {min(longer_side): .3f}, max_len: {max(longer_side): .3f}')
        plt.ylabel('Frequency')
        plt.title('Length of rect (longer side)')
    else:
        plt.plot([])

    # plt.show()

    name = getResultName(img_name, "hist_all")
    save_dst = rect_hist_all_dir + '/' + name

    all_images = os.listdir(rect_hist_all_dir)
    if name in all_images:
        os.remove(save_dst)

    plt.savefig(save_dst)
    plt.clf()


def reorder_rect_points_vertical_rec(rect):
    min_y = []
    reordered_rect = []

    min_y1 = min(rect, key=lambda rec: rec[1])
    min_y.append(min_y1)
    index = np.where(np.all(rect == min_y1, axis=1))
    if len(index[0]) > 1:
        index = index[0][0]
    rect = np.delete(rect, index, axis=0)
    min_y2 = min(rect, key=lambda rec: rec[1])
    min_y.append(min_y2)
    index = np.where(np.all(rect == min_y2, axis=1))
    max_y = np.delete(rect, index, axis=0)

    if max_y[0][0] <= max_y[1][0]:
        min_x_max_y = max_y[0].tolist()
        max_x_max_y = max_y[1].tolist()
    else:
        max_x_max_y = max_y[0].tolist()
        min_x_max_y = max_y[1].tolist()

    if min_y1[0] <= min_y2[0]:
        min_x_min_y = min_y1.tolist()
        max_x_min_y = min_y2.tolist()
    else:
        min_x_min_y = min_y2.tolist()
        max_x_min_y = min_y1.tolist()

    return [min_x_min_y, max_x_min_y, max_x_max_y, min_x_max_y]


def reorder_rect_points_horizontal_rec(rect):
    max_x = []
    reordered_rect = []

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


def get_middle_point_of_side(upper_point, lower_point):
    upper_x, upper_y = upper_point
    lower_x, lower_y = lower_point
    middle_x = int((upper_x + lower_x) / 2)
    middle_y = int((upper_y + lower_y) / 2)
    return [middle_x, middle_y]


def dst_of_points(start_point, end_point):
    x_start, y_start = start_point
    x_end, y_end = end_point

    x_diff = x_end - x_start
    y_diff = y_end - y_start

    dst = math.sqrt(x_diff ** 2 + y_diff ** 2)
    return dst


# weight on horizontal direction -> prioritized rectangles on vertical axis
def weighted_dst_vertical(start_point, end_point):
    x_start, y_start = start_point
    x_end, y_end = end_point

    x_diff = x_end - x_start
    y_diff = y_end - y_start

    weight = 5
    weighted_dst = math.sqrt((x_diff * weight) ** 2 + y_diff ** 2)

    return weighted_dst


# weight on vertical direction => prioritized rectangles on horizontal axis
def weighted_dst_horizontal(start_point, end_point):
    x_start, y_start = start_point
    x_end, y_end = end_point

    x_diff = x_end - x_start
    y_diff = y_end - y_start

    weight = 5
    weighted_dst = math.sqrt(x_diff ** 2 + (y_diff * weight) ** 2)

    return weighted_dst


def find_closest_vertical_to_horizontal_rec(all_hor_rect, all_ver_rect):
    # need fix: value = tuple(map(tuple, closest_rect_right)) 'NoneType' object is not iterable

    closest_results = {}
    closest_left = None
    closest_right = None

    for start_rec in all_hor_rect:
        start_rec = reorder_rect_points_horizontal_rec(start_rec)

        left_side_upper = start_rec[0]
        left_side_lower = start_rec[3]
        mid_left = get_middle_point_of_side(left_side_upper, left_side_lower)
        min_dst_left = 10000

        right_side_upper = start_rec[1]
        right_side_lower = start_rec[2]
        mid_right = get_middle_point_of_side(right_side_upper, right_side_lower)
        min_dst_right = 10000

        for end_rect in all_ver_rect:
            end_rect = reorder_rect_points_vertical_rec(end_rect)

            upper_side_left = end_rect[0]
            upper_side_right = end_rect[1]
            mid_upper = get_middle_point_of_side(upper_side_left, upper_side_right)

            dst_left_upper = dst_of_points(mid_left, mid_upper)
            dst_right_upper = dst_of_points(mid_right, mid_upper)

            lower_side_left = end_rect[3]
            lower_side_right = end_rect[2]
            mid_lower = get_middle_point_of_side(lower_side_left, lower_side_right)

            dst_left_lower = dst_of_points(mid_left, mid_lower)
            dst_right_lower = dst_of_points(mid_right, mid_lower)

            pass

def find_closest_vertical_rect(all_ver_rect):
    closest_results = {}
    closest_rect = None
    real_dst = 0

    for start_rec in all_ver_rect:
        start_rec = reorder_rect_points_vertical_rec(start_rec)
        mid_start = get_middle_point_of_side(start_rec[0], start_rec[1])
        min_dst = 10000
        for end_rec in all_ver_rect:
            end_rec = reorder_rect_points_vertical_rec(end_rec)
            if start_rec != end_rec:
                mid_end = get_middle_point_of_side(end_rec[3], end_rec[2])
                dst_act = weighted_dst_vertical(mid_start, mid_end)
                if dst_act < min_dst:
                    closest_rect = end_rec
                    min_dst = dst_act
        # mid_closest = get_middle_point_of_side(closest_rect[3], closest_rect[2])
        key = tuple(map(tuple, start_rec))
        if closest_rect is not None:
            value = tuple(map(tuple, closest_rect))
            mid_end_closest = get_middle_point_of_side(closest_rect[3], closest_rect[2])
            real_dst = dst_of_points(mid_start, mid_end_closest)
        else:
            value = key

        closest_results[key] = [value, min_dst, real_dst]

    return closest_results


def find_closest_horizontal_rect(all_hor_rect):
    closest_results = {}
    closest_rect = None
    real_dst = 0

    for start_rec in all_hor_rect:
        start_rec = reorder_rect_points_horizontal_rec(start_rec)
        mid_start = get_middle_point_of_side(start_rec[1], start_rec[2])
        min_dst = 10000
        for end_rec in all_hor_rect:
            end_rec = reorder_rect_points_horizontal_rec(end_rec)
            if start_rec != end_rec:
                mid_end = get_middle_point_of_side(end_rec[0], end_rec[3])
                dst_act = weighted_dst_horizontal(mid_start, mid_end)
                if dst_act < min_dst:
                    closest_rect = end_rec
                    min_dst = dst_act
        # mid_closest = get_middle_point_of_side(closest_rect[0], closest_rect[3])
        # if mid_start[0] < mid_closest[0]:
        key = tuple(map(tuple, start_rec))
        if closest_rect is not None:
            value = tuple(map(tuple, closest_rect))
            mid_end_closest = get_middle_point_of_side(closest_rect[0], closest_rect[3])
            real_dst = dst_of_points(mid_start, mid_end_closest)
        else:
            value = key
        closest_results[key] = [value, min_dst, real_dst]

    return closest_results


def connect_closest_horizontal_rect(all_hor_rect):
    pass


def detect_horizontal_lines(img, copy=None):
    all_rect_points = []
    all_rect_box = []  # (center(x,y), (width, height), angle of rotation)

    if copy is None:
        copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

    bw_swap = cv2.bitwise_not(threshold)
    dilated = cv2.dilate(bw_swap, np.ones((4, 2), dtype=np.uint8))  # vodorovne: 4,1
    eroded = cv2.erode(dilated, np.ones((1, 13), dtype=np.uint8))  # vodorovne: 1, 9

    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_length = 2
    height, width = img.shape[:2]
    max_length = max(height, width) / 2

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # if (rect[1][0] > min_length and rect[1][1] > min_length) and (
        #         rect[1][0] < max_length and rect[1][1] < max_length):
        if rect[1][0] * rect[1][1] > 20:
            all_rect_box.append(rect)
            all_rect_points.append(box)
            # cv2.drawContours(copy, [box], 0, (0, 255, 0), 2)

    all_rect = [all_rect_points, all_rect_box]

    copy = draw_rectangles(copy, all_rect_points, True)

    return copy, eroded, all_rect


def detect_vertical_lines(img, copy=None):
    all_rect_points = []
    all_rect_box = []  # (center(x,y), (width, height), angle of rotation)
    if copy is None:
        copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

    bw_swap = cv2.bitwise_not(threshold)
    dilated = cv2.dilate(bw_swap, np.ones((2, 4), dtype=np.uint8))  # vodorovne: 4,1
    eroded = cv2.erode(dilated, np.ones((13, 1), dtype=np.uint8))  # vodorovne: 1, 9, ... 9-15

    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_length = 3
    height, width = img.shape[:2]
    max_length = max(height, width) / 2

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if (rect[1][0] > min_length and rect[1][1] > min_length) and (
                rect[1][0] < max_length and rect[1][1] < max_length):
            all_rect_box.append(rect)
            all_rect_points.append(box)
            # cv2.drawContours(copy, [box], 0, (0, 0, 255), 2)

    all_rect = [all_rect_points, all_rect_box]

    copy = draw_rectangles(copy, all_rect_points, False)

    return copy, eroded, all_rect


def detectLinesHough(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    bw_swap = cv2.bitwise_not(thresholded)

    dilated = cv2.dilate(bw_swap, np.ones((4, 4), dtype=np.uint8))  # 3,3
    # cv2.imshow("dilated", dilated)
    eroded = cv2.erode(dilated, np.ones((4, 4), dtype=np.uint8))  # aj 2,2 alebo 3,3
    # cv2.imshow("eroded", eroded)

    edged = eroded

    # edged = cv2.Canny(blurred, 10, 100)
    # # edged = cv2.dilate(edged, np.ones((3, 3), dtype=np.uint8))
    # edged = cv2.dilate(edged, np.ones((10, 10), dtype=np.uint8))
    # edged = cv2.erode(edged, np.ones((10, 10), dtype=np.uint8))

    rho = 0.7  # distance resolution in pixels of the Hough grid
    theta = 3 * np.pi / 180  # The resolution of the parameter theta in radians: 1 degree
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(edged, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    img_copy = drawLines(img_copy, lines)

    # cv2.imshow("Edged image", edged)
    # cv2.imshow("Lines", img_copy)

    # doHistogram(lines)
    return img_copy, lines, edged
    # return drawLines(img, lines)


def getResultName(img_name, description):
    if description == '':
        return img_name

    start = len(img_name) - 6
    exten_index = img_name.find('.', start)
    result_name = img_name[:exten_index] + '_' + description + img_name[exten_index:]
    return result_name


def saveImage(dst_dir, img_name, description, res_img):
    result_name = getResultName(img_name, description)
    # print(result_name)
    all_images = os.listdir(dst_dir)

    result_path = dst_dir + '/' + result_name
    if result_name in all_images:
        os.remove(result_path)

    cv2.imwrite(result_path, res_img)


def getAllImages():
    # folder_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    # dst_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines"
    # input_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines_input"

    # folder_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    # folder_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1_resized"
    folder_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1_digital_resized"

    dst_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_hlines"
    input_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_hlines_input"

    horizontal_lines_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_horizontalLines"
    horizontal_input_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_horizontalLines_input"

    vertical_lines_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_verticalLines"
    vertical_input_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_verticalLines_input"

    horizontal_vertical_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_horizontal_vertical"

    hor_rect_hist_closest_dst_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_hor_rect_hist_closest_dst"
    ver_rect_hist_closest_dst_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_ver_rect_hist_closest_dst"

    digital_imgs_contour_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs_ru1_digital_contours"

    all_images = os.listdir(folder_dir)
    # print(all_images)

    for image_name in all_images:
        path = folder_dir + '/' + image_name
        img = cv2.imread(path)
        print(image_name)

        digital_contours = find_contours(img)
        saveImage(digital_imgs_contour_dir, image_name, "", digital_contours)


        # img_hlines, lines, input_img = detectLinesHough(img)
        # saveImage(dst_dir, image_name, 'hough_lines', img_hlines)
        # saveImage(input_dir, image_name, 'input', input_img)

        # horizontal_lines, horizontal_lines_input, horiz_data = detect_horizontal_lines(img)
        # closest_horizontal = find_closest_horizontal_rect(horiz_data[0])

        # horizontal_lines_connected = draw_connected_middle_points_closest_horizontal(horizontal_lines.copy(), closest_horizontal)
        # horizontal_lines_connected = draw_connected_middle_points_max_length_horizontal(horizontal_lines.copy(), closest_horizontal, 30)
        # saveImage(horizontal_lines_dir, image_name, 'horizontal_lines', horizontal_lines)
        # saveImage(horizontal_lines_dir, image_name, 'horizontal_lines', horizontal_lines_connected)
        # saveImage(horizontal_input_dir, image_name, 'horizontal_input', horizontal_lines_input)

        # vertical_lines, vertical_lines_input, vertical_data = detect_vertical_lines(img)
        # closest_vertical = find_closest_vertical_rect(vertical_data[0])

        # vertical_lines_connected = draw_connected_middle_points_closest_vertical(vertical_lines, closest_vertical)
        # vertical_lines_connected = draw_connected_middle_points_max_length_vertical(vertical_lines.copy(), closest_vertical, 30)
        # saveImage(vertical_lines_dir, image_name, 'vertical_lines', vertical_lines)
        # saveImage(vertical_lines_dir, image_name, 'vertical_lines', vertical_lines_connected)
        # saveImage(vertical_input_dir, image_name, 'vertical_input', vertical_lines_input)

        # horizontal_vertical, _, _ = detect_vertical_lines(img, horizontal_lines)
        # hor_ver_connected = draw_connected_middle_points_max_length_horizontal(horizontal_vertical, closest_horizontal, 30)
        # hor_ver_connected = draw_connected_middle_points_max_length_vertical(hor_ver_connected, closest_vertical, 30)
        # closest_hor_ver = find_closest_vertical_to_horizontal_rec(horiz_data[0], vertical_data[0])
        # horizontal_vertical = draw_connected_middle_points_closest_horizontal_vertical(horizontal_vertical, closest_hor_ver)
        # saveImage(horizontal_vertical_dir, image_name, 'horizontal_vertical', horizontal_vertical)
        # saveImage(horizontal_vertical_dir, image_name, 'horizontal_vertical', hor_ver_connected)

        # hor_rect_box = horiz_data[1]
        # ver_rect_box = vertical_data[1]
        # plot_histogram(hor_rect_box, ver_rect_box, image_name)
        # plot_histogram_area(hor_rect_box, ver_rect_box, image_name)

        # colors, bins, binwidth = histogram_closest_distances(hor_rect_hist_closest_dst_dir, closest_horizontal, image_name)
        # hor_lines_colors = draw_connected_middle_points_histogram_colors(horizontal_lines, closest_horizontal, colors, bins, binwidth)
        # saveImage(horizontal_lines_dir, image_name, 'hstColors', hor_lines_colors)

        # histogram_closest_distances(ver_rect_hist_closest_dst_dir, closest_vertical, image_name)

        # print(image_name)


def lines_by_hist_bins(bins, bin_width, closest_data, img, boundary, dir, name):

    for i in range(boundary):
        max_length = bins[i] + bin_width

        for start_rec, end_rec in closest_data.items():
            dst = end_rec[1]
            if dst < max_length:
                start_rec_right_upper = start_rec[1]
                start_rec_right_lower = start_rec[2]

                end_rec_left_upper = end_rec[0][0]
                end_rec_left_lower = end_rec[0][3]

                start_point = get_middle_point_of_side(start_rec_right_upper, start_rec_right_lower)
                end_point = get_middle_point_of_side(end_rec_left_upper, end_rec_left_lower)

                img = cv2.circle(img, end_point, 2, (255, 51, 255), 2)
                img = cv2.line(img, start_point, end_point, (255, 51, 255), 2)

    #cv2.imshow("limited by bin", img)

    #dst = dir + '/' + name

    #cv2.imwrite(dst, img)
    saveImage(dir, name, str(boundary), img)


def depict_all_bins_separetly(bins, binwidth, closest, img, dir, name):
    for i in range(1, len(bins)):
        lines_by_hist_bins(bins, binwidth, closest, img, i, dir, name)


def lines_by_hist_for_certain_images():
    all_img_dir = 'C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1'
    results_parent_dir = 'C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/hranice_hist/fix_bins'
    #results_parent_dir = 'C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/hranice_hist/auto_bins'
    text_file_dir = 'C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/hranice_hist/vzorove_obr_pre_histogrami.txt'

    results_dirs = os.listdir(results_parent_dir)

    file = open(text_file_dir, 'r')
    lines = file.readlines()

    for i in range(len(lines)):
        name = lines[i].replace("\n", "")
        img_dir = all_img_dir + '/' + name
        img = cv2.imread(img_dir)

        result_dir = results_parent_dir + '/' + results_dirs[i]

        hor_lines, hor_lines_in, hor_all_rec = detect_horizontal_lines(img)
        hor_all_rec_points = hor_all_rec[0]
        closest = find_closest_horizontal_rect(hor_all_rec_points)
        colors, bins, binwidth = histogram_closest_distances(result_dir, closest, name)
        depict_all_bins_separetly(bins, binwidth, closest, hor_lines, result_dir, name)


def get_new_image_size(orig_height, orig_width):
    new_longer_side_px = 1000

    if orig_width >= orig_height:
        new_width = new_longer_side_px
        new_height = (new_longer_side_px * orig_height) / orig_width
    else:
        new_height = new_longer_side_px
        new_width = (new_longer_side_px * orig_width) / orig_height

    return int(new_height), int(new_width)


def resize_all_images():
    source_dst = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    result_dir = "C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1_resized"

    all_images = os.listdir(source_dst)

    for img_name in all_images:
        img_path = source_dst + '/' + img_name
        img = cv2.imread(img_path)

        orig_height, orig_width = img.shape[:2]
        new_height, new_width = get_new_image_size(orig_height, orig_width)

        # !!! cv2. resize has order of new values: (width, height)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        saveImage(result_dir, img_name, '', resized_img)


def find_contours(img):
    template_orig = cv2.imread('images/vzorovy_obdlznik2.png')
    template = cv2.cvtColor(template_orig, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(template, 127, 255, 0)
    bw_swap1 = cv2.bitwise_not(thresh1)

    # cv2.imshow('vzor', bw_swap1)

    template_contours, template_hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(template_contours, key=cv2.contourArea, reverse=True)
    template_contour = sorted_contours[1]
    # print(template_contour)

    cv2.drawContours(template_orig, [template_contour], -1, (0, 255, 0), 3)

    # cv2.imshow('templ cnt', template_orig)
    # print(len(template_contours))

    image_copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)

    bw_swap = cv2.bitwise_not(threshold)
    dilated = cv2.dilate(bw_swap, np.ones((3, 3), dtype=np.uint8))  # 2, 2
    eroded = cv2.erode(dilated, np.ones((2, 2), dtype=np.uint8))  # 2, 2

    # cv2.imshow('eroded', bw_swap)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # temp = contours[3]
    # cv2.drawContours(image=image_copy, contours=[temp], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # cv2.imshow('temp', image_copy)

    # print("Number of Contours is: " + str(len(contours)))

    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        color = random_color()
        # if cnt_area < 400 or cnt_area > 20000:
        #     continue

        rect = cv2.minAreaRect(cnt)
        rect_width = rect[1][0]
        rect_heigh = rect[1][1]
        rect_area = rect_width * rect_heigh

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # box = box.reshape((4, 1, 2))
        # match_rect = cv2.matchShapes(box, cnt, 3, 0.0)
        # # # print(match)
        # #
        # if match_rect < 0.02:
        # cv2.drawContours(image=image_copy, contours=[cnt], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            ellipse_area = math.pi * MA/2 * ma/2

            cnt_rect_diff = rect_area - cnt_area
            cnt_ellipse_diff = ellipse_area - cnt_area

            if cnt_rect_diff < cnt_ellipse_diff:
                cv2.drawContours(image=image_copy, contours=[box], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            elif cnt_rect_diff > cnt_ellipse_diff:
                cv2.ellipse(image_copy, ellipse, (0, 0, 255), 2)
            else:
                cv2.drawContours(image=image_copy, contours=[cnt], contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        #     # print(ellipse)
        #     # if ellipse_area > 5000:
        #     cv2.ellipse(image_copy, ellipse, (0, 0, 255), 3)



        # else:
        #     cv2.drawContours(image=image_copy, contours=[cnt], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)

        # if 6000 > cnt_area > 400:
        #     cv2.drawContours(image=image_copy, contours=[cnt], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)


    # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # cv2.imshow('cnts', image_copy)
    return image_copy

if __name__ == '__main__':

    # resize_all_images()

    img = cv2.imread('C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/dbs2022_riadna_uloha1_digital_resized/Arkadelphia.jpg')
    # cv2.imshow("img orig", img)

    # find_contours(img)
    # # img_copy = img.copy()
    # # resize to half of the size
    # # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # hor_lines, hor_lines_in, hor_all_rec = detect_horizontal_lines(img)
    # hor_all_rec_points = hor_all_rec[0]
    # # hor_all_rec_box = hor_all_rec[1]

    # hor_lines_copy = copy.deepcopy(hor_lines)
    # cv2.imshow("hlc", hor_lines_copy)

    # # ver_lines, ver_lines_in, ver_all_rec = detect_vertical_lines(hor_lines)
    #
    # # ver_all_rec_points = ver_all_rec[0]
    # # ver_all_rec_box = ver_all_rec[1]

    # closest = find_closest_horizontal_rect(hor_all_rec_points)
    # print(closest)
    # # hor_lines_points = draw_connected_middle_points_closest_horizontal(hor_lines, closest)
    # hor_lines_points = draw_connected_middle_points_max_length(hor_lines, closest, 80)

    # cv2.imshow("colors", hor_lines_points)
    #
    # # #closest_ver_hor = find_closest_vertical_to_horizontal_rec(hor_all_rec_points, ver_all_rec_points)
    # # #hor_lines_points = draw_connected_middle_points_closest_horizontal_vertical(ver_lines, closest_ver_hor)
    # # cv2.imshow("hor with points", hor_lines_points)
    #
    # pokus = 'C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/hranice_hist'
    # colors, bins, binwidth = histogram_closest_distances(pokus, closest, 'Anaheim.jpg')
    # img_copy = draw_connected_middle_points_histogram_colors(hor_lines_copy, closest, colors, bins, binwidth)
    # cv2.imshow("colors", img_copy)
    #
    # #lines_by_hist_bins(bins, binwidth, closest, hor_lines_copy, 2, pokus, 'Alhambra.jpg')
    # depict_all_bins_separetly(bins, binwidth, closest, hor_lines_copy, pokus, 'Anaheim.jpg')

    #
    # print(bins)

    getAllImages()
    digital_images_results.show_results_html()
    #showResultsHTML()

    #print(os.listdir('C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/hranice_hist'))

    #lines_by_hist_for_certain_images()
    # lines_by_hist_html()
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print(get_new_image_size(550, 561))
    # print(get_new_image_size(1058, 522))

    # res = getResultName("pokus.pg", '')
    # print(res)
