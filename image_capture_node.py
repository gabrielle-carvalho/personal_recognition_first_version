#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

import os
import cv2
import time

class ImageCaptureNode(Node): # node that capture images
    def __init__(self):
        super().__init__("image_capture_node")
        self.camera = cv2.VideoCapture(0) #change to the correct range  ls /dev/video*
        
        if not self.camera.isOpened():
            self.get_logger().error("Failed to open camera.")
            return
            
        self.create_subscription(String, 'person_name', self.capture_callback, 10)
        # self.create_subscription(String, 'next_capture_step', self.next_capture_step, 10)
        
        self.ok_publisher = self.create_publisher(String, 'capture_status', 10)
        self.status_final_callback = self.create_publisher(String, 'vision_capture_status', 10)
        
        self.get_logger().info("Image capture Node has  started")
        self.stages=0

        self.bridge = CvBridge()  # Initialize CvBridge once
        
        self.path=None
        
    def capture_callback(self, msg):
        # name = msg.data 
        self.path = f"./images/{msg.data}"
        os.makedirs(self.path, exist_ok=True)
        
    def next_capture_step(self, msg):
        if not self.path:
            self.get_logger().error("Path not set. Ensure person_name message received first.")
            return
        
        for i in range(3):  # Capture 3 images for each message revieve from person_name
            
            ret, frame = self.camera.read()
            if not ret:
                self.get_logger().error("Failed to capture image.")
                return
            
            picnumber = len(os.listdir(self.path))
            cv2.imwrite(f'{self.path}/{picnumber}.png', frame)
            self.get_logger().info(f"Image saved as {self.path}/{picnumber}.png")
            
            self.get_logger().info("sleep mode for 4 seconds...")
            time.sleep(4)  # Pause between captures
            
        self.ok_publisher.publish(String(data="ok"))  # Publish OK message
        self.get_logger().info("Images Captured Successfully!")
        self.stages+=1  
        # self.status_final_callback()   
    
    # def status_final_callback(self):
    #     if self.stages==7: #depois que as 7 etapas sao percorridas manda o ok final
    #         self.status_final_callback.publish(String(data="finish"))  # Publish OK message
    #         self.get_logger().info("The train has finished!")
        
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
    
