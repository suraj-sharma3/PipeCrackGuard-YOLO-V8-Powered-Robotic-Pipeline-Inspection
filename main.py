from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch, from all the different versions of yolov8, we are using the nano version, that's why yolov8n.yaml

# Use the model
model.train(data="config.yaml", epochs=20)  # train the model
