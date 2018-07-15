# import the necessary packages
import numpy as np
import cv2
import os
import shutil
import matplotlib.pyplot as plt

FOLDER_CREATE_THRESHOLD = 700
ITEMS = [
    'Checkbox Off',
    'Checkbox On',
    'Radio button Off',
    'Radio button On',
    'Floating action button',
    'Button',
    'Slider',
    'Drop down button',
    'Text Area',
    'Text Field',
    'Switch button Off',
    'Switch button On',
    'Chip',
    'Data Table',
    'Menu',
    'List',
    'Alert',
    'Bottom sheet',
    'Bottom navigation',
    'Time picker',
    'Date picker',
    'TabBar',
    'Snackbar',
    'Tooltip',
    'Grid list',
    'Card',
]


class Extract:

    def label_files(self):
        """
        Moves files in current folder to labelled folders
        """

        files = os.listdir(self.current_folder)

        for i in range(len(files)):
            shutil.move(os.path.join(self.current_folder, files[i]), os.path.join(
                self.output_folder, ITEMS[i]))

        # Delete folder after moving all files
        os.rmdir(self.current_folder)

    def move_incomplete_folder(self):
        """
        Moves current folder to unsorted
        """
        shutil.move(self.current_folder, self.unsorted_folder)

    def find_contours(self, image):
        """Find contours from the scanned image

        Arguments:
            image {Image} -- [Scanned image read by OpenCV]

        Returns:
            List -- [List of contours]
        """
        # settings
        gaussian_kernel = (3, 3)
        gaussian_sigmaX = 0

        canny_threshold1 = 10
        canny_threshold2 = 250

        structuring_element_kernel = (3, 3)

        # grayscale the image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, gaussian_kernel, gaussian_sigmaX)

        # detect edges in the image
        edged_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)

        # construct and apply a closing kernel to 'close' gaps between 'white' pixels
        structuring_element = cv2.getStructuringElement(
            cv2.MORPH_RECT, structuring_element_kernel)
        closed = cv2.morphologyEx(
            edged_img, cv2.MORPH_CLOSE, structuring_element)

        # find contours (i.e. the 'outlines') in the image
        (_, contours, _) = cv2.findContours(closed.copy(),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Return the contours
        return contours

    def extract(self):
        # settings
        crop_offset = 200
        bounding_rectangle_width_threshold = 50
        bounding_rectangle_height_threshold = 50
        contour_outline_offset = 50

        file_list = os.listdir(self.input_folder)
        file_list.sort()

        for filename in file_list:

            if not filename.endswith('.jpg'):
                continue

            detected = 0

            # load the image
            loaded_image = cv2.imread(
                os.path.join(self.input_folder, filename))

            # get image shape
            height, width, _ = loaded_image.shape

            # crop the right side where we have sketches (half width - offset)
            image = loaded_image[:height, ((width // 2) - crop_offset):width]

            contours = self.find_contours(image)

            # if number of detected contours > threshold (happens only in consent form page)
            # create a new folder for user with filename
            if len(contours) > FOLDER_CREATE_THRESHOLD:

                # skip first time
                if self.current_folder == self.output_folder:
                    pass
                # if all elements found -> label files
                elif len(os.listdir(self.current_folder)) == 26:
                    self.label_files()
                # else move to unsorted folder
                else:
                    self.move_incomplete_folder()

                # Create new folder for next user
                self.current_folder = os.path.join(
                    self.output_folder, os.path.splitext(filename)[0])
                os.makedirs(self.current_folder, exist_ok=True)

            else:
                # loop over the contours
                for c in contours:

                    # approximate the contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                    # process only the rectangles (polygon with 4 points)
                    if len(approx) == 4:

                        x, y, w, h = cv2.boundingRect(c)

                        # if the image is too small then ignore
                        if w < bounding_rectangle_width_threshold or h < bounding_rectangle_height_threshold:
                            continue

                        # offset the found contour to compensate the rectangular outline
                        [y0, y1] = [y + contour_outline_offset,
                                    y + h - contour_outline_offset]
                        [x0, x1] = [x + contour_outline_offset,
                                    x + w - contour_outline_offset]

                        # mark the contour as region of interest and save it in output folder
                        roi = image[y0:y1, x0:x1]
                        out_file = os.path.join(
                            self.current_folder, '{0}-{1}.jpg'.format(os.path.splitext(filename)[0], str(detected)))
                        cv2.imwrite(out_file, roi)
                        detected += 1

            print("Processed {}..".format(filename))

        print("Data extraction successful...")

    def __init__(self, input_folder, output_folder):
        super(Extract, self).__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.current_folder = output_folder
        self.unsorted_folder = os.path.join(self.output_folder, 'Unsorted')

        # # Create necessary subfolders in output folder
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.unsorted_folder, exist_ok=True)

        for item in ITEMS:
            os.makedirs(os.path.join(self.output_folder, item), exist_ok=True)

        self.extract()
