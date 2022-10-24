import os
import webbrowser


def showResultsHTML():
    # https://realpython.com/html-css-python/#handle-html-with-python

    original_images_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs2022_riadna_uloha1"
    hlines_images_dir = "C:/Users/HP/Desktop/zofka/FEI_STU/bakalarka/dbs_ru1_hlines"
    original_images = os.listdir(original_images_dir)
    hlines_images = os.listdir(hlines_images_dir)

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
        <tr><th>Original</th><th>Hough Lines</th><th>Name</th></tr>
    </thead>
    <tbody>
    """

    for i in range(len(original_images)):
        original_img_path = original_images_dir + '/' + original_images[i]
        hlines_img_path = hlines_images_dir + '/' + hlines_images[i]
        table_rov = "<tr><td><img src= \"" + original_img_path + "\" alt=\"Sth went wrong\" width=\"250\" height=\"300\"></td><td><img src=\"" + hlines_img_path + "\" alt=\"Sth went wrong\" width=\"250\" height=\"300\"></td></tr>\n"
        html_table = html_table + table_rov

    # all_emoji = "ABCDEFGH"
    # columns = ["#", "Emoji", "Name"]
    #
    # table_head = f"<thead>\n<tr><th>{'</th><th>'.join(columns)}</th></tr>\n</thead>"
    #
    # table_body = "\n<tbody>\n"
    # for i, emoji in enumerate(all_emoji, start=1):
    #     emoji_data = [f"{i}.", emoji, unicodedata.name(emoji).title()]
    #     table_body += f"<tr><td>{'</td><td>'.join(emoji_data)}</td></tr>\n"
    # table_body += "</tbody>\n"
    #
    # all = f"<table>\n{table_head}{table_body}</table>"
    #
    # f = open('Results.html', 'w')
    # f.write(all)
    #
    # # close the file
    # f.close()

    html_table = html_table + "</tbody>\n</table>"
    html_all = html_head + "\n" + html_table
    f = open('Results.html', 'w')
    f.write(html_table)

    # close the file
    f.close()

    webbrowser.open('Results.html')