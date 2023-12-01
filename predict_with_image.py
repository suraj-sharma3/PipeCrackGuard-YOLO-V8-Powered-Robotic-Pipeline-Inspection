# Testing the yolo model with image data

import os
import cv2
from ultralytics import YOLO

# Load the image
image_path = "C:\\Users\\OMOLP094\\Desktop\\Research Projects\\Laksh _ Vardaan - Pipe Crack Detection Project\\test_images\\test_images\\cracked_pipe2.jpg"
image = cv2.imread(image_path)

# Load the YOLO model
model_path = os.path.join('.', 'best.pt') # here, '.' means the root directory 
model = YOLO(model_path)

# Detection threshold
threshold = 0.5

# Perform inference
results = model(image)[0]

# print(results)
print(type(results))

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        # Get the class label from the results.names dictionary
        class_label = results.names[int(class_id)].upper()

        # Prepare the text to display
        text = f"{class_label}: {score:.2f}"

        # Draw the class label and confidence score on the image
        cv2.putText(image, text, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Display the image with bounding boxes and labels
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()