import numpy as np
import cv2
import os
import astimp
import json

def xyR(img: np.ndarray, img_name: str)-> list:

    '''
    takes as input:
    img      - the image to be processed
    img_name - the name of the image being processed
               (needed to retrieve and build masks accordingly)

    outputs a list:
    mask_variables - where first element is the name of the image and
                     preceding elements have the format [x_center (float), y_center(float), radius(float)]
                     corresponding to the center coordinates and radius
                     of inhibition circles
    '''
    ast = astimp.AST(img)
    mask_variables = [img_name]

    # Load the original and cropped images
    original_img = img
    cropped_img = ast.crop.copy()

    # Convert images to grayscale (needed for template matching)
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Template matching
    result = cv2.matchTemplate(original_gray, cropped_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # The top-left corner of the cropped area in the original image
    top_left_crop = max_loc

    height, width, channels = img.shape

    for i, inhib in enumerate(ast.inhibitions):
      center_x, center_y = ast.circles[i].center
      diameter = inhib.diameter

      #saves in mask_variables a list as [x_center, y_center, radius]
      mask_variables.append([ float(center_x) + float(top_left_crop[0]),
                              float(top_left_crop[1]) + float(center_y),
                              ast.px_per_mm * diameter/2 ])
    ''''!!!!! there may be problems with radius estimation ast.px_per_mm * diameter/2 
              this may not fit the the way annotation tool calculates the radii
    '''
    return mask_variables

def main():
    '''
    as the input path (e.g. /home/antibiogo/images/) and
    output_path of json file (e.g. /home/anitibiogo/data.json) is provided
    it write to json file all needed x_center, y_center, radius variables of images
    the written file has the list format
    [elem, elem, ...]
    where elem = [img_name, [x_center, y_center, radius], [x_center, y_center, radius], ...]
          elem stores the name of an image file and all the necessary variables of inhibition masks
    '''
    input_path ="" #path where images are stored
    output_path = "" #path to json file where the final result will be stored

    inhibition_variables = []
    for file in os.listdir(input_path):
        if file.endswith(".jpg" or ".png"):
            img_path = os.path.join(input_path, file)
            img = cv2.imread(img_path)

            inhibition_variables.append(xyR(img, file))
    with open(output_path, 'w') as f:
        json.dump(inhibition_variables, f)
if __name__ == "__main__":
    main()


