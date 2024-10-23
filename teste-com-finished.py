#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cv_bridge import CvBridge
import os
import cv2
import time
from sensor_msgs.msg import Image


class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__("image_capture_node")
        self.camera = cv2.VideoCapture(0) #change to the correct range  ls /dev/video*
        
        if not self.camera.isOpened():
            self.get_logger().error("Failed to open camera.")
            self.destroy_node()
            return
        
        self.image_publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.create_subscription(String, 'person_name', self.capture_callback, 10)
        self.create_subscription(String, 'next_capture_step', self.next_capture_step_callback, 10)
        
        self.ok_publisher = self.create_publisher(String, 'capture_status', 10)
        self.publisher_training_status = self.create_publisher(String, 'training_finished', 10)
        
        self.get_logger().info("Image capture Node has started")
        self.bridge = CvBridge()
        self.path = ""
        self.capture_count = 0
        self.finished = 0
        
                
    def capture_callback(self, msg):
        self.name = msg.data 
        self.path = f"./images/{self.name}"
        os.makedirs(self.path, exist_ok=True)
        self.capture_count = 0 
        self.get_logger().info(f"Ready to capture images for: {self.name}")
        self.next_capture_step_callback()
        
        
    def next_capture_step_callback(self, msg=None):
        if not self.path:
            self.get_logger().error("Path not set.")
            return

        if self.capture_count < 3: 
            ret, frame = self.camera.read()
            if not ret:
                self.get_logger().error("Failed to capture image.")
                return

            # Salva a imagem no diretório local
            picnumber = len(os.listdir(self.path))
            cv2.imwrite(f'{self.path}/{picnumber}.png', frame)
            self.get_logger().info(f"Image saved as {self.path}/{picnumber}.png")

            # Publica a imagem capturada no tópico /camera/image_raw
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_publisher.publish(msg)
            self.get_logger().info("Published image to /camera/image_raw")

            # Atualiza o contador de capturas
            self.capture_count += 1
            self.get_logger().info("Sleeping for 8 seconds...")
            time.sleep(8)  # Espera antes de capturar a próxima imagem

            # Chama a função novamente para continuar a captura
            self.next_capture_step_callback()

        else:
            # Quando terminar de capturar, publica o 'ok' e reseta o contador
            self.ok_publisher.publish(String(data="ok"))
            self.finished+=1
            self.get_logger().info("Images Captured Successfully!")
            self.capture_count = 0
            
        if self.finished == 3:
            self.publisher_training_status.publish(String(data="round180"))
            
    def destroy_node(self):
        if self.camera.isOpened():
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
