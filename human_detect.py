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
#   ser = serial.Serial('/dev/ttyUSB1', 9600, timeout=1)
#   ser.flush()
  
  # Variables to calculate FPS 
  counter, fps = 0, 0
  start_time = time.time()
  
  # Load labels
  labels = load_labels(label_path)

  # Start capturing video input from the camera
  cap = cv2.VideoCapture("http://192.168.50.41:4747/video", cv2.CAP_FFMPEG)
  # cap = cv2.VideoCapture(camera_id)
  # cap = cv2.VideoCapture(2)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
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

    # Detection result on terminal (show only person detections)
    if detection_result and detection_result.detections:
        person_detections = [d for d in detection_result.detections
                            if d.categories and d.categories[0].index in labels and 
                            labels[d.categories[0].index] == 'person']
        
        # Determine current detection state
        person_current_state = len(person_detections) > 0
        # Only print message if state changes
        if person_current_state != person_prev_state:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                if person_current_state:
                    print(f"[{timestamp}] Person detected!")
                else:
                    print(f"[{timestamp}] No person detected.")
                person_prev_state = person_current_state
    else:
        person_detections = []
                    
    # Visualize only 'person' detections
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()
    
    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    
    # Display the number of detected persons
    persons_text = f'Person: {len(person_detections)}'
    cv2.putText (image, persons_text, (left_margin, row_size + 20),
                cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
    
    # # Take picture when serial data is received          
    # if ser.in_waiting > 0:
    #   line = ser.readline().decode('utf-8').rstrip()
    #   print(line)     

    #   # Take 5 pictures if the signal is received
    #   if int(line) == 1:
    #     ser.write(b"WARNING\n")    
    #     for i in range(5):
    #       success, image = cap.read()
    #       if success:
    #         cv2.imwrite(f"/home/indraa/Documents/Coding AI/Dimas_ITATS/result_images/result_{i}.jpg", image_result)
    #         time.sleep(1)
    #     ser.write(b"CAPTURED\n")
    
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)
    
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