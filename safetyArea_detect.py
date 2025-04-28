# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import serial

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

def load_labels(label_path: str) -> dict:
    """Loads the labels file and returns a mapping from ID to label name."""
    with open(label_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, label_path: str) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
    label_path: Path to the label file.
    serial_port: Serial port to communicate with Arduino..
  """

  # Initialize serial communication with Arduino
  ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
  ser.flush()
  
  # Variables to calculate FPS 
  counter, fps = 0, 0
  start_time = time.time()
  
  # Load labels
  labels = load_labels(label_path)

  # Start capturing video input from the camera
  cap = cv2.VideoCapture("http://192.168.98.139:4747/video", cv2.CAP_FFMPEG)
  # cap = cv2.VideoCapture(camera_id)
  # cap = cv2.VideoCapture(2)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  person_prev_state = None

  last_send_time = 0  # Waktu terakhir kirim data
  send_interval = 0.5  # 500 ms = 0.5 detik

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        sys.exit(
            'ERROR: Unable to read from webcam. Please verify your webcam settings.'
        )

    counter += 1
    # image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Filter detections to only show 'person'
    person_detections = [d for d in detection_result.detections
                        if d.categories[0].index in labels and labels[d.categories[0].index] == 'person']

    # Define zones: left (danger), center (safe), right (danger)
    left_zone_end = int(width * 0.33)
    right_zone_start = int(width * 0.66)

    # Draw zone boundaries on the image
    cv2.rectangle(image, (0, 0), (left_zone_end, height), (0, 0, 255), 2)           # Left - Red
    cv2.rectangle(image, (left_zone_end, 0), (right_zone_start, height), (0, 255, 0), 2)  # Center - Green
    cv2.rectangle(image, (right_zone_start, 0), (width, height), (0, 0, 255), 2)    # Right - Red

    danger_detected = False
    count_safe = 0
    count_danger = 0

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
    fps_text = 'FPS = {:.1f}'.format(fps)  
    cv2.putText(image, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    # Check each person if they are in danger zone
    for detection in person_detections:
        bbox = detection.bounding_box
        center_x = int(bbox.origin_x + bbox.width / 2)
        center_y = int(bbox.origin_y + bbox.height / 2)

        # Draw center point
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

        if center_x < left_zone_end:
            zone_status = "LEFT (DANGER)"
            danger_detected = True
            count_danger += 1
        elif center_x > right_zone_start:
            zone_status = "RIGHT (DANGER)"
            danger_detected = True
            count_danger += 1
        else:
            zone_status = "CENTER (SAFE)"
            count_safe += 1

        if "DANGER" in zone_status:
            text_color_zone = (0, 0, 255)  # Red
        else:
            text_color_zone = (0, 255, 0)  # Green

        cv2.putText(image, zone_status, (bbox.origin_x, bbox.origin_y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, text_color_zone, 1)

    # Cek apakah sudah waktunya kirim data?
    current_time = time.time()
    if current_time - last_send_time >= send_interval:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        if danger_detected:
            cv2.putText(image, f"SAFE: {count_safe}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.putText(image, f"DANGER: {count_danger}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            print(f"[{timestamp}] WARNING: Person detected in DANGER zone!")
            ser.write(b"DANGER\n")  # Uncomment if using serial communication
        else:
            cv2.putText(image, f"SAFE: {count_safe}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.putText(image, f"DANGER: {count_danger}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            print(f"[{timestamp}] All clear. No one in DANGER zone.")
            ser.write(b"SAFE\n")

        last_send_time = current_time  # Update waktu kirim terakhir

    # Stop the program if the ESC key is pressed.
    cv2.imshow('object_detector', image)
    if cv2.waitKey(1) == 27:
      break
    
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
      # default='best.tflite') # file hasil train palet mas arif
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(
      '--labelPath',
      help='Path to the label file.',
      required=False,
      default='labelmap.txt')
  args = parser.parse_args()
  

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU),  args.labelPath)

if __name__ == '__main__':
  main()