#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ultralytics import YOLO
import os
import cv2
import math
import time

class ObjectDetection(Node):
    def __init__(self):
        super().__init__("object_detection_node")
        
        self.create_subscription(String, 'start_detection', self.capture_callback, 10)

        self.object_name_publisher = self.create_publisher(String, 'recognized_object', 10)
        self.class_object_name_publisher = self.create_publisher(String, 'class_recognized_object', 10)
        
        self.get_logger().info("Object Detection Node has Started")
        
        os.makedirs("captured_images", exist_ok=True)
        self.started = False
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        model = YOLO("runs/detect/yolov8n_custom4/weights/best.pt")
        
        classNames = [
    'Cutlery-cup', 'Cutlery-fork', 'Cutlery-knife', 'Cutlery-mug', 'Cutlery-plate', 
    'Cutlery-spoon', 'Pantry-items-coffee', 'Pantry-items-sweetener', 
    'cleaning-supplies-cloth', 'cleaning-supplies-detergent', 'cleaning-supplies-sponge', 
    'cutlery-bottle', 'others-Garbage-Bin', 'others-bucket', 'others-insecticide', 
    'pantry-items-coffee-filter'
        ]
        self.timer = self.create_timer(0.1, self.detection_loop)


    def capture_callback(self, msg):
        if not self.started:
            self.started = True
            self.get_logger().info(f'Recieved: "{msg.data}"')
        else:
            self.get_logger().info('Detection already started')
            
            
    def detection_loop(self):
        if self.started:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to grab frame")
                return  # Skip if frame not grabbed

            # Prediction
            results = self.model.predict(frame, stream=True, conf=0.9)

            # Results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    self.process_box(box, frame)

            cv2.imshow('Webcam', frame)
            cv2.waitKey(1)

    def process_box(self, box, frame):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil(box.conf[0] * 100)
            cls = int(box.cls[0])

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            label = f"{self.classNames[cls]} ({confidence}%)"
            org = (x1, y1 - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, label, org, font, fontScale, color, thickness)

            # Publish recognized object name and class
            self.object_name_publisher.publish(String(data=self.classNames[cls]))
            self.class_object_name_publisher.publish(String(data=label))

            # Save image
            timestamp = int(time.time())
            img_filename = f"captured_images/{self.classNames[cls]}_{timestamp}.jpg"
            cv2.imwrite(img_filename, frame)
            self.get_logger().info(f"Saved image: {img_filename}")

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    
    object_detection_node = ObjectDetection()
    
    rclpy.spin(object_detection_node)

    object_detection_node.destroy_node()
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
