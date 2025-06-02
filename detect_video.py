import cv2
import numpy as np
from easyocr import Reader
from ultralytics import YOLO
from sort.sort import Sort


dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None

# Initialize SORT tracker
mot_tracker = Sort()

# Load YOLOv8 model for vehicle detection
vehicle_model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 model for vehicles
license_plate_model = YOLO('models/license_plate_detector.pt')  # Custom-trained model for license plate detection

# Initialize EasyOCR for license plate recognition
reader = Reader(['en'])

# Define vehicle classes
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Load video
video_path = './video2.mp4'
cap = cv2.VideoCapture(video_path)

frame_nmr = -1
ret = True

# Function to check license plate format
def license_complies_format(text):
    return len(text) == 7 and all(char.isalnum() for char in text)

# Draw border function for both vehicle and license plate bounding boxes
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Vehicle bounding box with custom border
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Initialize an empty dictionary to store tracking IDs and their maximum confidence
# Initialize an empty dictionary to store tracking IDs, their maximum confidence, and license plate text
track_id_confidence = {}

# Main loop for video processing
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 vehicle detection
    vehicle_results = vehicle_model(frame)[0]
    detections = []

    # Ensure correct detection format: [x1, y1, x2, y2, confidence]
    for detection in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = detection
        if int(cls_id) in vehicle_classes and conf > 0.5:  # Filter by class and confidence
            detections.append([x1, y1, x2, y2, conf])  # No need to convert to scalar

    # Check if detections list is non-empty
    if len(detections) > 0:
        # SORT tracking
        tracked_objects = mot_tracker.update(np.array(detections))

        # Process each tracked object
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)

            # Draw the vehicle bounding box with the custom border
            frame = draw_border(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=5, line_length_x=50, line_length_y=50)

            # Update the maximum confidence for the current tracked ID
            if track_id not in track_id_confidence:
                track_id_confidence[track_id] = {'confidence': conf, 'plate_text': None}
            else:
                track_id_confidence[track_id]['confidence'] = max(track_id_confidence[track_id]['confidence'], conf)

            cv2.putText(frame, f"ID: {track_id}", (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            # Crop the detected vehicle for license plate detection
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size > 0:
                license_plate_results = license_plate_model(vehicle_crop)[0]

                for lp_detection in license_plate_results.boxes.data.tolist():
                    lp_x1, lp_y1, lp_x2, lp_y2, lp_conf, lp_cls_id = lp_detection
                    if lp_conf > 0.5:  # Filter by confidence
                        # Adjust coordinates relative to the full frame
                        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, [lp_x1 + x1, lp_y1 + y1, lp_x2 + x1, lp_y2 + y1])

                        # Draw the license plate bounding box with the custom border
                        frame = cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 2)

                        # Crop license plate for OCR
                        lp_crop = frame[lp_y1:lp_y2, lp_x1:lp_x2]
                        H, W, _ = lp_crop.shape
                        if lp_crop.size > 0:
                            lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                            _, lp_crop_thresh = cv2.threshold(lp_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                            cv2.imshow('License Plate', lp_crop_thresh)
                            lp_text, score = read_license_plate(lp_crop_thresh)

                            # Resize the plate crop to double its size
                            double_size_plate_crop = cv2.resize(lp_crop, (2 * W, 2 * H))  # Double the width and height

                            try:
                                # Check if the OCR confidence is higher than the tracking confidence
                                if lp_text is not None:
                                    # If OCR confidence is better than stored confidence, update the dictionary
                                    if track_id_confidence[track_id]['confidence'] < score:
                                        track_id_confidence[track_id]['confidence'] = score
                                        track_id_confidence[track_id]['plate_text'] = lp_text

                                    # Place the resized plate crop into the frame
                                    frame[int(y1) - 2 * H - 25:int(y1) - 25,  # Adjusted Y-coordinates to fit the larger crop
                                          int((x2 + x1 - 2 * W) / 2):int((x2 + x1 + 2 * W) / 2), :] = double_size_plate_crop

                                    # Make the white space around the plate bigger
                                    frame[int(y1) - 2 * H - 125:int(y1) - 2 * H - 25,  # Adjusted Y-coordinates to fit the white space
                                          int((x2 + x1 - 2 * W) / 2):int((x2 + x1 + 2 * W) / 2), :] = (255, 255, 255)

                                    # Draw the license plate text (using the updated or stored text)
                                    cv2.putText(frame,
                                                track_id_confidence[track_id]['plate_text'],  # Use the updated plate text
                                                (int((x2 + x1 - 2 * W) / 2), int(y1 - 2 * H - 35)),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                2.5,  # Adjust font scale
                                                (0, 0, 0),
                                                3)  # Adjust thickness
                            except Exception as e:
                                print(f"Error: {e}")
                                pass

    # Show the updated frame with vehicle and license plate bounding boxes
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(30)

    # Wait for a key press, break if 'q' is pressed
    if key & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


