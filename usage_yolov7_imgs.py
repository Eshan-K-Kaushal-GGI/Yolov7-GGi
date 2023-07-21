import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

class_names = ['fire', 'smoke', 'vegetation', 'human', 'background', 'sparks', 'arc']
danger_names = ['fire', 'smoke', 'sparks', 'arc']

def detect(image_path, model_path):
    # Configuration
    view_img = True
    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.45

    # Directories
    save_dir = Path("output")  # Directory to save results (not used in this modified code)

    # Initialize
    set_logging()
    device = select_device('')  # Automatically select a CUDA device if available
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load model
    model = attempt_load(model_path, map_location=device)  # Load the model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(imgsz, s=stride)  # Check img_size

    if half:
        model.half()  # Convert the model to FP16 (half precision)

    # Set Dataloader
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Lists to store detected results
    detected_classes = []
    detected_confidences = []
    detected_boxes = []
    danger_scores = []  # List to store confidence values of dangerous detections

    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # Convert to FP16/32
        img /= 255.0  # Normalize to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        model(img, augment=False)[0]

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # Detections per image
            im0 = im0s
            if isinstance(im0, list):  # Multiple frames as input
                im0 = im0[i].copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Process detections and draw bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    class_name = names[int(cls)]
                    label = f'{class_name} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    detected_classes.append(class_name.lower())
                    detected_confidences.append(conf.item())
                    detected_boxes.append([int(x) for x in xyxy])

                    # Check if the detected class is in danger_names
                    if class_name.lower() in danger_names:
                        danger_scores.append(conf.item())

            # Display the image with detections
            if view_img:
                cv2.imshow('Inference Results', im0)
                cv2.waitKey(0)  # Wait for a key press

    cv2.destroyAllWindows()

    # Get the max score of dangerous detections
    print(f'Danger_Scores: {danger_scores}')
    max_score = max(danger_scores) if len(danger_scores) > 0 else 0.0

    # Create the output dictionary
    out_dict = {
        "detected_names": detected_classes,
        "scores": detected_confidences,
        "boxes": detected_boxes,
        "max_score": max_score
    }

    return out_dict


if __name__ == '__main__':
    image_path = 'Images/Random_Testing/11.png'  # Replace with the path to your image
    model_path = 'runs/train/train/weights/best.pt'  # Replace with the path to your model

    with torch.no_grad():
        out_dict = detect(image_path, model_path)

        # Print the detected classes, their confidence values, and bounding boxes
        for i in range(len(out_dict["detected_names"])):
            print(f'Detected: {out_dict["detected_names"][i]}, Confidence: {out_dict["scores"][i]}, Bounding Box: {out_dict["boxes"][i]}')

        # Print the max score of dangerous detections
        print(f'Max Score of Dangerous Detections: {out_dict["max_score"]}')
print(out_dict)