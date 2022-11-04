import os
import webbrowser

import cv2


def showResultsHTML():
    original_images_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    hlines_images_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines"
    hlines_input_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines_input"

    original_images = os.listdir(original_images_dir)
    hlines_images = os.listdir(hlines_images_dir)
    hlines_input_images = os.listdir(hlines_input_dir)

    html_head = """
    <head>
        <style>
        table,
        th,
        td {
            border: 1px solid black;
            border-collapse: collapse;
        }
        </style>
    </head>
    """

    html_table = """
    <table>
    <thead>
        <tr><th>Original</th><th>Hough Lines</th><th>Hlines input</th></tr>
    </thead>
    <tbody>
    """

    for i in range(len(original_images)):
        original_img_path = original_images_dir + '/' + original_images[i]
        hlines_img_path = hlines_images_dir + '/' + hlines_images[i]
        hlines_input_path = hlines_input_dir + '/' + hlines_input_images[i]

        img = cv2.imread(original_img_path)
        height, width = img.shape[:2]
        table_rov_info = "<tr>" \
                            "<td>" + original_images[i] + "\n hxw: "+ str(height) + " x " + str(width) + "</td>"\
                         "</tr>"
        table_rov = "<tr>" \
                        "<td><img src= \"" + original_img_path + "\" alt=\"Sth went wrong\" width=\"450\" height=\"500\"></td>" \
                        "<td><img src=\"" + hlines_img_path + "\" alt=\"Sth went wrong\" width=\"450\" height=\"500\"></td>" \
                        "<td><img src=\"" + hlines_input_path + "\" alt=\"Sth went wrong\" width=\"450\" height=\"500\"></td>" \
                    "</tr>\n"
        html_table = html_table + table_rov_info + "\n" + table_rov

    html_table = html_table + "</tbody>\n</table>"
    html_all = html_head + "\n" + html_table
    f = open('Results.html', 'w')
    f.write(html_table)

    # close the file
    f.close()

    webbrowser.open('Results.html')