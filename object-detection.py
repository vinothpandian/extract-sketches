# import the necessary packages
import numpy as np
import cv2
import os


def detect(filename, output_folder='output' ):

    # load the image and get its shape
    image = cv2.imread(filename)

    # grayscale the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # detect edges in the image
    edged = cv2.Canny(gray, 10, 250)

    # construct and apply a closing kernel to 'close' gaps between 'white' pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # find contours (i.e. the 'outlines') in the image
    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0

    # loop over the contours
    for c in cnts:

        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Store the bounding rectangle
        x,y,w,h = cv2.boundingRect(c)

        # if the image is too small then ignore
        if w < 50 or h < 50:
            continue

        # offset the found contour to compensate the rectangular outline
        o = 0

        # mark the contour as region of interest and save it in output folder
        roi=image[y+o:y+h-o,x+o:x+w-o]
        cv2.imwrite(os.path.join(output_folder, '{0}-{1}.jpg'.format(os.path.splitext(filename)[0], str(total))), roi)
        total += 1

    # display the number of elements found in each file
    print('Found {0} elements in file {1}'.format(total, filename))


# loop through all files in input folder and process the images

file = 'sketch.png'
OUT_FOLDER = 'sketch_objs'

detect(file, output_folder=OUT_FOLDER)
