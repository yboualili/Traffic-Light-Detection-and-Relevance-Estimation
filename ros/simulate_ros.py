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

bridge = cv_bridge.CvBridge()
model = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', help="Path to the ultralytics model", type=str, required=True)
    parser.add_argument('--out_dir', '-o', type=str, default='ccng_jpgs', help='Directory where to store the evaluation results')
    return parser.parse_args()

def convert_to_image(msg):
    # Convert the ROS image message to an OpenCV image
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    # Generate a file name based on the message header
    out_name: str = f'{msg.header.frame_id.split("/")[-1]}_{msg.header.stamp.secs}-{msg.header.stamp.nsecs}.jpg'
    # Save the image as a jpg file
    cv2.imwrite(os.path.join(out_dir_bag,topicpath,out_name), cv_image)


def get_newest_data(self):
    if self.mutex.acquire(blocking=True, timeout=1/self.publish_rate):
        img_msg = self.img_msg
        self.camera_info_sync = self.camera_info
        
        self.img_msg = None
        #self.camera_info = None #allow old cam_infos
        self.mutex.release()
    if img_msg is None or self.camera_info_sync is None: #global self.img_msg was none so no new image since last feedforward
        return False
    self.img_header_sync = img_msg.header
    p = self.camera_info_sync.P
    self.g_p_sync = [[p[0], p[1], p[2]], [p[4],p[5],p[6]], [p[8], p[9], p[10]]]
    if (img_msg.header.stamp != self.camera_info_sync.header.stamp):
        rospy.logwarn_throttle(10.0, "Camera_Info and Image do not have the same timestamp")

    self.input_latency = round((rospy.Time.now() - img_msg.header.stamp).to_sec(), 3)
    self.input_latency_sum += self.input_latency
    return img_msg



def draw_boxes(img, labels):
    return



# Define callback function
def image_callback(msg, bridge, model):
    # Convert sensor_msgs/Image to numpy array
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    # Preprocess image
    image = cv2.resize(image, (224, 224)) # resize to 224x224
    image = image / 255.0 # normalize to [@, 1]
    image = image.transpose(2, 0, 1) # change channel order to CHW
    image = torch.from_numpy(image).float() # convert to torch tensor
    image = image.unsqueeze(0) # add batch dimension
    # Feed image to model and get prediction
    output = model( image)
    prediction = output.argmax(dim=1).item() # get the index of the max value
    # Do something with prediction
    print(f"Prediction: {prediction}")


def main(args):
    
    # load model
    model = YOLO(args.model_path)

    # Create ROS node and subscribe to image topic
    rospy.init_node("image_subscriber")
    image_sub = rospy.Subscriber("/front_medium/RGB", Image, image_callback(bridge=bridge, model=model))

    # Spin the node until Ctrl+c
    rospy.spin()


def evaluate_image():
    # Load PyTorch model
    model = torch. load("my_model.pth")
    model.eval()


if __name__ == "__main__":
    main(parse_args())