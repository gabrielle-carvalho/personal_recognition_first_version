#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

import os
import cv2
import time

 # node that capture images
class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__("image_capture_node")
        self.camera = cv2.VideoCapture(0) #change to the correct range
        
        if not self.camera.isOpened():
            self.get_logger().error("Failed to open camera.")
            return
            
        self.create_subscription(String, 'person_name', self.capture_callback, 10)
        self.ok_publisher = self.create_publisher(String, 'capture_status', 10)
        self.get_logger().info("Image capture Node started")

        self.bridge = CvBridge()  # Initialize CvBridge once
        
    def capture_callback(self, msg):
        name = msg.data 
        path = f"./images/{name}"
        os.makedirs(path, exist_ok=True)
        
        for i in range(3):  # Capture 3 images for each message revieve from person_name
            #turn left up
            #turn left down
            #turn right up
            #turn right down
            #straight to camera
            #straight to camera up
            #straight to camera down
            
            ret, frame = self.camera.read()
            if not ret:
                self.get_logger().error("Failed to capture image.")
                return
            
            picnumber = len(os.listdir(path))
            cv2.imwrite(f'{path}/{picnumber}.png', frame)
            self.get_logger().info(f"Image saved as {path}/{picnumber}.png")
            time.sleep(2)  # Pause between captures
            self.get_logger().info("sleep mode for 2 seconds...")
                        
        self.ok_publisher.publish(String(data="OK"))  # Publish OK message
        self.get_logger().info("Info", "Images Captured Successfully!")
        
    def destroy_node(self):
        self.camera.release()
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    
    image_capture_node = ImageCaptureNode() 
    
    try:
        rclpy.spin(image_capture_node)
    finally:
        image_capture_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    