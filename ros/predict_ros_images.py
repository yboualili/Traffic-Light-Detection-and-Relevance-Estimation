# python3
import sys
import os
import argparse

# ROS
import rospy
import cv2
import torch
from sensor_msgs.msg import Image
import cv_bridge

#CV
from ultralytics import YOLO
import json

bridge = cv_bridge.CvBridge()
model = YOLO("best.pt")

def drawBoundingBoxes(imageData, imageOutputPath, inferenceResults):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    for res in inferenceResults:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['right'])
        bottom = int(res['bottom'])
        label = res['label']
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 1200)
        if 'red' in label:
            color = (0, 3, 252)
        elif 'yellow' in label:
            color = (3, 111, 252)
        elif 'green' in label:
            color = (28, 252, 3)
        else:
            color = (92, 91, 91)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        # avoid out of image text
        if left > 1500:
            left -= 11 * len(label)
        cv2.putText(imageData, label, (left, top - 10), 0, 0.00075 * imgHeight, color, thick)
    cv2.imwrite(imageOutputPath, imageData)


# Define callback function
def image_callback(msg):
    # Convert sensor_msgs/Image to numpy array
    
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    name = msg.header.seq
    # # Preprocess image
    image = cv2.resize(image, (1920, 1200)) # resize to 224x224

    result = model.predict(image, verbose=False, imgsz=(1920, 1216), conf=0.4, iou=0.7, device=0)

    to_draw = []

    for box in json.loads(result[0].tojson()):

        box_dims = box['box']

        args = {
            "left": box_dims['x1'],
            "top": box_dims['y1'],
            "right": box_dims['x2'],
            'bottom': box_dims['y2'],
            'label': box['name']
        }
        to_draw.append(args)

    drawBoundingBoxes(image, f'images/{name}.jpg', to_draw)


def main():
    
    # load model
    

    # Create ROS node and subscribe to image topic
    rospy.init_node("image_subscriber")
    image_sub = rospy.Subscriber("/sensors/camera/front_medium/resized/image_rect_color", Image, image_callback)

    # Spin the node until Ctrl+c
    rospy.spin()


if __name__ == "__main__":
    main()
