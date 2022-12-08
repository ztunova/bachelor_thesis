import os
import webbrowser

import cv2


def get_order_from_img_name(img_name):
    exten_index = img_name.find('.')
    underscore_index = img_name.find('_') + 1

    order = img_name[underscore_index:exten_index]

    if order == 'hist_dst':
        return 0
    elif order == 'hist_dst_res':
        return -1
    else:
        return int(order)


def lines_by_hist_html():
    parent_dir = 'C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/hranice_hist/fix_bins'
    #parent_dir = 'C:/Users/zofka/OneDrive/Dokumenty/FEI_STU/bakalarka/hranice_hist/auto_bins'
    subdirs = os.listdir(parent_dir)
    #subdirs.remove('vzorove_obr_pre_histogrami.txt')


    html_table = """
        <table>
        <tbody>
        """

    for dir in subdirs:
        i = 0
        path = parent_dir + '/' + dir
        content = os.listdir(path)
        content.sort(key=get_order_from_img_name)
        #print(content)

        sample_img_path = path + '/' + content[0]
        img = cv2.imread(sample_img_path)
        height, width = img.shape[:2]

        table_rov_info = "<tr>"
        table_rov = "<tr>"

        for img in content:
            img_path = path + '/' + img
            img = cv2.imread(sample_img_path)
            height, width = img.shape[:2]

            table_rov_info_data = "<td>" + content[i] + "\n hxw: " + str(height) + " x " + str(width) + "</td>"

            table_data = "<td><img src= \"" + img_path + "\" alt=\"Sth went wrong\" width=\"500\" height=\"450\"></td>"

            table_rov_info = table_rov_info + table_rov_info_data
            table_rov = table_rov + table_data
            i = i + 1

        table_rov_info = table_rov_info + '</tr>'
        table_rov = table_rov + '</tr>'
        html_table = html_table + table_rov_info + "\n" + table_rov + '\n'

    html_table = html_table + "</tbody>\n</table>"

    f = open('output_lines_by_hist_bins.html', 'w')
    f.write(html_table)

    # close the file
    f.close()

    webbrowser.open('output_lines_by_hist_bins.html')

