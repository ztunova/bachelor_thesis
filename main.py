import cv2
import numpy as np

if __name__ == '__main__':
    # load image
    #img = cv2.imread('images/ERD_basic1_dig.png')
    #img = cv2.imread('images/ERD_simple_HW_noText.jpg')
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = cv2.imread('images/tvary_biele_na_ciernom.png')
    # resize to half of the size
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    # find the contours in the dilated image - BETTER WITHOUT
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    # dilate = cv2.dilate(edged, kernel, iterations=1)
    # contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = img.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")

    i = 0
    for contour in contours:
        x1, y1 = contour[0][0]
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            cv2.putText(image_copy, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        elif 6 < len(approx) < 15:
            cv2.putText(image_copy, "Ellipse", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # cv2.imshow("Dilated image", dilate)
    cv2.imshow("Edged image", edged)
    cv2.imshow("contours", image_copy)

    cv2.imshow("Original image", img)

    # wait until key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
