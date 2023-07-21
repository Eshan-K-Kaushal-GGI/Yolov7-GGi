''' Author - Eshan K Kaushal'''
# ONLY for IN-HOUSE GGi usage
# Dated - 20_07_2023 (dd_mmm_yyyy)

import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box, plot_one_box_lower_right
from utils.torch_utils import select_device, time_synchronized, TracedModel

class_names = ['fire', 'smoke', 'vegetation', 'human', 'background', 'sparks', 'arc']
danger_names = ['fire', 'smoke', 'sparks', 'arc']

def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def detect(video_path, model_path):
    # Configuration
    view_img = True
    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.47
    display_scale = 0.8

    prev_time = 0
    fps_sum = 0
    num_frames = 0

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
    if video_path.startswith("rtsp://"):
        dataset = LoadStreams(video_path, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(video_path, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Define the colors for each class (fire, smoke, vegetation, background, human, sparks, arc)
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (42, 42, 165), (255, 0, 0), (128, 0, 128), (255, 0, 255)]

    # Lists to store detected results
    detected_classes = []
    detected_confidences = []
    detected_boxes = []
    danger_scores = []  # List to store confidence values of dangerous detections
    max_danger_score_class = ""

    # Run inference on video frames
    for _, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # Convert to FP16/32
        img /= 255.0  # Normalize to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        model(img, augment=False)[0]
        current_time = time.time()
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

                clip_coords(det, im0.shape)

                # Process detections and draw bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    class_name = names[int(cls)]
                    label = f'{class_name} {conf:.2f}'
                    plot_one_box_lower_right(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                    detected_classes.append(class_name.lower())
                    detected_confidences.append(conf.item())
                    detected_boxes.append([int(x) for x in xyxy])

                    # Check if the detected class is in danger_names
                    if class_name.lower() in danger_names:
                        danger_scores.append(conf.item())

                        if conf.item() == max(danger_scores):
                            max_danger_score_class = class_name

            # Display the image with detections

            # Resize the frame for larger display
            new_width = int(im0.shape[1] * display_scale)
            new_height = int(im0.shape[0] * display_scale)
            im0 = cv2.resize(im0, (new_width, new_height))

            # time calc
            elapsed_time = time.time() - current_time
            fps = 1.0/elapsed_time
            fps_text = f'FPS: {fps:.2f}'
            cv2.putText(im0, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            #update FPS sum and number of Frames
            fps_sum += fps
            num_frames += 1

            # Display the max score on the frame
            max_score = max(danger_scores) if len(danger_scores) > 0 else 0.0
            max_danger_score = f'Max Danger Score: {max_score:.2f}'
            class_responsible = f'Danger Class: {max_danger_score_class}'
            text_size = cv2.getTextSize(max_danger_score, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = im0.shape[1] - text_size[0] - 10
            text_y = im0.shape[0] - text_size[1] - 10
            cv2.putText(im0, max_danger_score, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(im0, class_responsible, (text_x, text_y + text_size[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

            if view_img:
                cv2.imshow('Inference Results', im0)
                if cv2.waitKey(1) == 113:  # Press 'q' to stop
                    break
    avg_fps = fps_sum/num_frames
    cv2.destroyAllWindows()

    # Get the max score of dangerous detections
    max_score = max(danger_scores) if len(danger_scores) > 0 else 0.0

    # Get image width and height
    img_width, img_height = img.shape[2], img.shape[3]

    # Create the output dictionary
    out_dict = {
        "image_width": img_width,
        "image_height": img_height,
        "detected_names": detected_classes,
        "scores": detected_confidences,
        "boxes": detected_boxes,
        "max_score": max_score,
        "danger_class_name": max_danger_score_class,
        "avf_fps": avg_fps,
    }

    return out_dict

if __name__ == '__main__':
    video_path = 'fire_test_1.mp4'  # Replace with the path to your video file
    model_path = 'runs/train/train/weights/best.pt'  # Replace with the path to your model
    IP_Path = "rtsp://ADMIN:GGi2011!@63.46.167.57/stream3"
    with torch.no_grad():
        out_dict = detect(video_path, model_path)

        # # Print the detected classes, their confidence values, and bounding boxes
        #for i in range(len(out_dict["detected_names"])):
            #print(f'Detected: {out_dict["detected_names"][i]}, Confidence: {out_dict["scores"][i]}, Bounding Box: {out_dict["boxes"][i]}')
            #print(f'{out_dict["max_score"]:.2f}')
        # # Print the max score of dangerous detections
        #print(f'Max Score of Dangerous Detections: {out_dict["max_score"]}')

print("Out_dict here: ", out_dict)
