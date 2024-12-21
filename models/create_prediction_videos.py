from ultralytics import YOLO
import glob
from natsort import natsorted
import cv2
import json
import os
import moviepy.video.io.ImageSequenceClip
from tqdm import tqdm

# https://gist.github.com/kodekracker/1777f35e8462b2b21846c14c0677f611 Shoutouts
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

        cv2.rectangle(imageData,(left, top), (right, bottom), (255, 255, 255), thick)

        dim_label_img = 70

        label_img = cv2.imread(f"icons/{label}.jpg")
        label_img = cv2.resize(label_img, (dim_label_img, dim_label_img)) 
        
        x_offset_old = int((left + right) / 2 - dim_label_img/2)
        y_offset = bottom

        x_offset = min(x_offset_old, imgWidth - dim_label_img)
        if x_offset < 0:
            x_offset = 0

        print(x_offset_old, x_offset)

        imageData[y_offset:y_offset+dim_label_img, x_offset:x_offset+dim_label_img] = label_img        

    cv2.imwrite(imageOutputPath, imageData)

model = YOLO('../epoch46.pt')

videos = ['/data/ccng/2023-11-23-16-10-09.bag']

for idx, video in enumerate(videos):

    print(f"Drawing pics for video_{idx}")
    
    os.makedirs(f'../presentation/video{idx}', exist_ok=True)

    image_files = glob.glob(f'/data/ccng/2023-11-23-16-10-09.bag/front_medium/*.jpg')

    for image_file in tqdm(image_files):

        filename = image_file.split('/')[-1]

        result = model.predict(image_file, verbose=False, imgsz=(1920, 1216), conf=0.5, iou=0.7)

        imcv = cv2.imread(image_file)

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

        drawBoundingBoxes(imcv, f'../presentation/video{idx}/{filename}', to_draw)


    print(f"Building video_{idx}")

    image_files = glob.glob(f'../presentation/video{idx}/*.jpg')
    image_files = natsorted(image_files)

    fps=45

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'/workspace/traffic-light-detection/presentation/video_{idx}.mp4')
