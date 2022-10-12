import cv2 as cv

if __name__ == '__main__':
    # 0 -> grayscale
    img = cv.imread('images/ERD_basic1_dig.png', 0)
    # resize to half of the size
    img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)

    cv.imshow('Image', img)
    # wait until key is pressed
    cv.waitKey(0)
    cv.destroyAllWindows()
