## imports
from __future__ import print_function

import json
import argparse
import logging
import sys
import numpy as np
import os
import shutil
import glob


np.set_printoptions(suppress=True)

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--build_path", help="Path where to store the final dataset. Should be an empty or non existing folder.", type=str, required=True)
    parser.add_argument("--data_path", help="Path to the jpg files. The specified folder should contain a train and a test subfolder, created from 'convert_tif.py'.", type=str, required=True)
    parser.add_argument("--label_path", help="Path to the label data", type=str, required=False, default="/data/DTLD/v2.0")
    parser.add_argument('-at', "--arrow_threshold", help="A bbox for an arrow traffic light is removed from the labels, if there is < threshold percentage of it inside of the image.", type=float, required=False, default=0.9)
    parser.add_argument('-ct', "--circle_threshold", help="A bbox for a circle traffic light is removed from the labels, if there is < threshold percentage of it inside of the image.", type=float, required=False, default=0.8)
    parser.add_argument('--small_tl_threshold', help="Set a threshold for the width in pixel. Traffic lights with bboxes up to this width are declared as small and as a result only the color is detected and not the pictogram", required=False, type=int, default=11)
    parser.add_argument('--extra_label_small_tl', help="Whether to use a separate label for small traffic lights.", required=False, type=bool, default=False)
    return parser.parse_args()



def convert_labels(build_path:str, label_path:str, arrow_treshold:int, circle_threshold:int, small_tl_threshold:int, extra_label_small_tl:bool):
    """
    Converting all the .json files to to .txt files containing the label of the images in the correct YOLO format.

    Labels with bboxes outside of the picture are removed if the percentage of the bbox that is inside the picture boundaries is lower than a certain threshold.
    The threshold is set via arguments and differs for circle and arrow traffic lights.
    If the threshold is above this value, the bbox is cut the the image boundaries.

    CURRENTLY ALL PICTURES ARE REMOVED, WHERE AT LEAST ONE LABEL IS REMOVED BECAUSE OF THE PREVIOUS RULE.
    """

    all_classes = ['circle_green', 'circle_red', 'off', 'circle_red_yellow', 'arrow_left_green', 'circle_yellow', 'arrow_right_red', 'arrow_left_red', 'arrow_straight_red', 'arrow_left_red_yellow', 'arrow_left_yellow', 'arrow_straight_yellow', 'arrow_right_red_yellow', 'arrow_right_green', 'arrow_right_yellow', 'arrow_straight_green', 'arrow_straight_left_green', 'arrow_straight_red_yellow', 'arrow_straight_left_red', 'arrow_straight_left_yellow']



    ## file level ##
    for split in ["train", "test"]:

        save_path = f"{build_path}/{split}/labels" # paths and variables where to store the labels

        # load labels
        f = open(os.path.join(label_path, f"DTLD_{split}.json"))
        label_file = json.load(f)
        f.close()

        label_list = label_file["images"]

        # count labels whichs bounding box has been correcte / cut down to image boundaries
        n_corrected_labels = 0
        # count labels that are valid and just converted to YOLO format
        n_valid_labels = 0
        # count labels that are for tram or pedestrian traffic lights
        n_irrelevant_labels = 0
        # count labels that are classified as unknown 
        n_unknown_labels = 0
        # count labels that are removed because of invalid bounding box
        n_removed_labels = 0
        # complete amount of labels
        n_labels = 0
        # removed images because of at least one invalid label
        n_removed_images = 0

        # Load the list of poor labeled images from the JSON file
        json_file = open("poor_labeled_images.json")
        poor_labeled_images = json.load(json_file)
        
        ## image level ##
        for label_entry in label_list:

            filename = label_entry["image_path"].split("/")[-1].split(".")[0]

            # check whether image was poorly labeled
            if filename not in poor_labeled_images:

                # store the final labels and bbox here
                print_buffer = []

                # if set the complete labels for the image are removed
                remove_image = False

                ## label level ##
                for label in label_entry["labels"]:

                    n_labels += 1

                    # if set the label is removed because the bbox is outside of the picture and cannot be corrected
                    remove_label = False
                    
                    # filter out label that is labeled as unknown
                    if label["attributes"]["state"] == "unknown" or label["attributes"]["pictogram"] == "unknown":

                        n_unknown_labels += 1

                    else:
                        if label["attributes"]["aspects"] != "one_aspects" :
                            # declare separate class for small traffic lights under threshold
                            if extra_label_small_tl:
                                if label["w"] <= small_tl_threshold:
                                    label_class = f'small_{label["attributes"]["state"]}'
                                else:
                                    label_class = f'normal_{label["attributes"]["pictogram"]}_{label["attributes"]["state"]}'
                            # use normal label but don't use pictogram when bbox_width is smaller than threshold
                            else:
                                if label["w"] <= small_tl_threshold:
                                    label_class = f'circle_{label["attributes"]["state"]}'
                                else:
                                    label_class = f'{label["attributes"]["pictogram"]}_{label["attributes"]["state"]}'

                            # filter out label for pedestrian or trams
                            if label["attributes"]["pictogram"] in ["tram", "pedestrian", "bicycle", "pedestrian_bicycle"]:

                                n_irrelevant_labels += 1

                            else:

                                if "off" in label_class:
                                    label_class = "off"

                                image_width, image_height = 2048, 1024

                                h = label["h"]
                                w = label["w"]
                                x = label["x"]
                                y = label["y"]

                                x2 = x + w
                                y2 = y + h

                                # check if the bounding box is inside of the picture boundaries
                                if (y < 0 or y2 > (image_height - 1) or x < 0 or x2 > (image_width - 1)):

                                    # calculate area of the bbox that is outside of the picture
                                    bbox_area = h * w
                                    bbox_area_picture = (min(y2,image_height-1) - max(y,0)) * (min(x2,image_width-1) - max(x,0))
                                    percentage = bbox_area_picture / bbox_area

                                    # correct bbox if it lays in tolerance area
                                    if ("circle" in label_class and percentage > circle_threshold) or ("arrow" in label_class and percentage > arrow_treshold):
                                        
                                        # correct bbox
                                        x = max(x,0)
                                        y2 = min(y2,image_height-1) 
                                        y = max(y,0)
                                        x2 = min(x2,image_width-1)
                                        x = max(x,0)
                                        h = y2 - y
                                        w = x2 - x

                                        n_corrected_labels += 1
                                    
                                    else:  
                                        # remove label
                                        remove_label = True
                                    
                                else:

                                    n_valid_labels += 1

                                if remove_label:

                                    n_removed_labels += 1
                                    remove_image = True

                                else:

                                    # add label to label collection list

                                    class_name_to_id_mapping = {i: j for j, i in enumerate(all_classes)}
                                    class_id = class_name_to_id_mapping[label_class]
                                    #class_id = 0
                                    # BBOX COORDINATES
                                    # Transform the bbox co-ordinates as per the format required by YOLO v5
                                    b_center_x = (x + x2) / 2 
                                    b_center_y = (y + y2) / 2
                                    b_width    = w
                                    b_height   = h

                                    # Normalise the co-ordinates by the dimensions of the image
                                    b_center_x /= image_width
                                    b_center_y /= image_height
                                    b_width    /= image_width
                                    b_height   /= image_height


                                    #Write the bbox details to the file
                                    print_buffer.append(f"{class_id} {b_center_x} {b_center_y} {b_width} {b_height}")

                # remove image if one label was removed because of incorrect bbox
                if remove_image:

                    n_removed_images += 1

                # save label file if there is at least one label
                if len (print_buffer) > 0:

                    # save label file
                    filename = label_entry["image_path"].split("/")[-1].split(".")[0]

                    with open(f'{save_path}/{filename}.txt', 'w') as filehandle:
                        for line in print_buffer:
                            filehandle.write(("".join(line) + "\n"))
        

        # logging at split level
        logging.info(f"""
                    Execution information for {split} folder :\n 
                    Originally contained labels : {n_labels} \n
                    Amount of unknown labels: {n_unknown_labels} \n
                    Amount of irrelevant labels: {n_irrelevant_labels} \n
                    Amount of removed labels: {n_removed_labels} \n
                    Amount of valid labels: {n_valid_labels} \n
                    Amount of corrected labels: {n_corrected_labels} \n
                    Amount of removed images because of removed label: {n_removed_images} \n
                    """)

    logging.info(f"There are {len(all_classes)} classes: {all_classes}")


def copy_images(build_path:str, data_path:str):
    """
    Copying the images, for which a label file exists, to the train / test folder in the build_path directory
    """
    for split in ["train", "test"]:

        for file in os.listdir(f"{build_path}/{split}/labels"):
            
            id = file.split('.')[0]

            src = f"{data_path}/{split}/{id}.jpg"
            dst = f"{build_path}/{split}/images/{id}.jpg"
            
            shutil.copyfile(src, dst)

        n_original_files = len(glob.glob(f"{data_path}/{split}/*.jpg"))
        n_copies = len(glob.glob(f"{build_path}/{split}/images/*.jpg"))
        logging.info(f"Copied {n_copies} of {n_original_files} images from {split} set.")


def main(args):
        
    data_path = args.data_path
    if not (os.path.exists(f"{data_path}/train") and os.path.exists(f"{data_path}/test")):
        raise Exception("Invalid data path. Make sure to provide the data as described at the --data_path")

    label_path = args.label_path
    if not os.path.exists(label_path):
        raise Exception("Invalid label path.")
    
    build_path = args.build_path
    if os.path.exists(build_path):
        if input("Folder already exists. Are you sure to overwrite the data? Please type yes. \n")!='yes':
            raise Exception("Stopping Execution! Please Change build_name !")
            
    if os.path.exists(build_path):
        shutil.rmtree(build_path)

    arrow_threshold = args.arrow_threshold
    circle_threshold = args.circle_threshold
    small_tl_threshold = args.small_tl_threshold
    extra_label_small_tl = args.extra_label_small_tl


    os.makedirs(f"{build_path}/train/labels", exist_ok=True)
    os.makedirs(f"{build_path}/test/labels", exist_ok=True)
    convert_labels(build_path, label_path, arrow_threshold, circle_threshold, small_tl_threshold, extra_label_small_tl)

    os.makedirs(f"{build_path}/train/images", exist_ok=True)
    os.makedirs(f"{build_path}/test/images", exist_ok=True)
    copy_images(build_path, data_path)

if __name__ == "__main__":
    main(parse_args())
