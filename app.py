import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load models
vehicle_model = YOLO("models/yolov8n.pt")
license_plate_model = YOLO("models/best.pt")

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

EXAMPLE_IMAGE_PATH = "example_image.jpg"
EXAMPLE_VIDEOS = ["video.mp4", "video2.mp4"]

def draw_l_shape_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_len=150):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, thickness)
    


track_id_confidence = {}
stop_processing = False  # Global stop flag for video processing

def detect_vehicles_and_plates_video_live(video_path):
    global stop_processing
    stop_processing = False
    cap = cv2.VideoCapture(video_path)
    vehicle_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        vehicle_results = vehicle_model(frame)
        vehicle_boxes = []

        for box, cls_id, conf in zip(vehicle_results[0].boxes.xyxy, vehicle_results[0].boxes.cls, vehicle_results[0].boxes.conf):
            cls_id = int(cls_id)
            if cls_id in vehicle_classes and float(conf) >= 0.65:
                vehicle_boxes.append(box.cpu().numpy().astype(int))

        plate_crops_to_draw = []

        for (x1, y1, x2, y2) in vehicle_boxes:
            draw_l_shape_border(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3, line_len=20)
            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue

            plate_results = license_plate_model(vehicle_roi)

            for pbox, pconf in zip(plate_results[0].boxes.xyxy, plate_results[0].boxes.conf):
                conf = float(pconf)
                if conf < 0.65:
                    continue

                px1, py1, px2, py2 = pbox.cpu().numpy().astype(int)
                abs_px1 = x1 + px1
                abs_py1 = y1 + py1
                abs_px2 = x1 + px2
                abs_py2 = y1 + py2

                cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 0, 255), 2)

                plate_crop = frame[abs_py1:abs_py2, abs_px1:abs_px2]
                if plate_crop.size == 0:
                    continue

                lp_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, lp_crop_thresh = cv2.threshold(lp_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                H, W = plate_crop.shape[:2]
                double_size_plate_crop = cv2.resize(plate_crop, (2 * W, 2 * H))

            

                # Calculate safe placement position for drawing later
                top_left_x = max(0, min(int((x1 + x2 - 2 * W) / 2), frame.shape[1] - 2 * W))
                top_left_y = max(0, int(y1 - 2 * H - 25))

                plate_crops_to_draw.append((top_left_y, top_left_x, double_size_plate_crop))

                vehicle_counter += 1

        # Draw all plate crops after loop
        for top_y, left_x, crop_img in plate_crops_to_draw:
            h, w = crop_img.shape[:2]
            if top_y + h <= frame.shape[0] and left_x + w <= frame.shape[1]:
                frame[top_y:top_y + h, left_x:left_x + w] = crop_img

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if stop_processing:
            break
        yield rgb_frame
        
    cap.release()

# New function for image license plate detection and visualization
def detect_license_plates_on_image(image_np):
    results = license_plate_model(image_np)
    # results[0].plot() returns numpy RGB image with bounding boxes drawn
    return results[0].plot()

# Gradio interface
with gr.Blocks(title="Plate Detection") as demo:
    with gr.Tab("Image"):
        image_input = gr.Image(value=EXAMPLE_IMAGE_PATH, type="numpy", label="Input Image")
        image_output = gr.Image(label="Detected License Plates")
        detect_button = gr.Button("Detect License Plates")
        detect_button.click(detect_license_plates_on_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Video"):
        video_dropdown = gr.Dropdown(choices=EXAMPLE_VIDEOS, value=EXAMPLE_VIDEOS[0], label="Select Example Video")
        video_output = gr.Image(label="Processed Video (Frame by Frame)")
        detect_video_button = gr.Button("Detect Vehicles and License Plates")
        stop_button = gr.Button("Stop Processing")
        
        stop_button.click(lambda: setattr(globals(), 'stop_processing', True), outputs=None)
        detect_video_button.click(detect_vehicles_and_plates_video_live, inputs=video_dropdown, outputs=video_output)

demo.launch()
