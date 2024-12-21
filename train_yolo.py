from ultralytics import YOLO, RTDETR

# Load a model
#model = RTDETR('rtdetr-l.pt')  build a new model from YAML
# model = YOLO("yolov11x.yaml)
model = YOLO("yolov8x.yaml")
# Train the model
results = model.train(data='custom.yaml', epochs=200, imgsz=1920, device=0, batch=11, patience=0, mosaic=0.6, scale=0.6, translate=0, pretrained=True, save_period=1, name="dtld_x_augmented", cos_lr=True, fliplr=0,  mixup=0.2, shear=0.3, copy_paste=0.4)