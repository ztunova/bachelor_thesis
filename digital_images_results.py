import os
import webbrowser

import cv2


def show_results_html(original_images_dir, contours_dir, removed_shapes_dir):
    original_images_names = os.listdir(original_images_dir)

    html_table = """
    <table>
    <thead>
        <tr><th>Original</th><th>Contours</th><th>Lines</th></tr>
    </thead>
    <tbody>
    """

    for i in range(len(original_images_names)):
        orig_img_path = original_images_dir + '/' + original_images_names[i]
        contour_img_path = contours_dir + '/' + original_images_names[i]
        removed_shapes_img_path = removed_shapes_dir + '/' + original_images_names[i]

        img = cv2.imread(orig_img_path)
        height, width = img.shape[:2]
        table_rov_info = "<tr>""<td>" + original_images_names[i] + "\n hxw: " + str(height) + " x " + str(width) + "</td>""</tr>"

        table_rov = "<tr>" \
                    "<td><img src= \"" + orig_img_path + "\" alt=\"Sth went wrong\" width=\"500\" height=\"400\"></td>" \
                    "<td><img src=\"" + contour_img_path + "\" alt=\"Sth went wrong\" width=\"500\" height=\"400\"></td>" \
                    "<td><img src=\"" + removed_shapes_img_path + "\" alt=\"Sth went wrong\" width=\"500\" height=\"400\"></td>" \
                    "</tr>\n"
        html_table = html_table + table_rov_info + "\n" + table_rov

    html_table = html_table + "</tbody>\n</table>"
    f = open('digital_images_results.html', 'w')
    f.write(html_table)
    # close the file
    f.close()

    webbrowser.open('digital_images_results.html')
