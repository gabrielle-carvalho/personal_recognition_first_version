#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CaptureStatusNode(Node):
    def __init__(self):
        super().__init__("capture_status_node")
        
        self.publisher_ = self.create_publisher(String, 'capture_status', 10)  # Nome do tópico corrigido
        self.timer_period = 8.0  # Publica a cada 2 segundos
        self.timer = self.create_timer(self.timer_period, self.publish_ok)
    
    def publish_ok(self):
        msg = String()
        msg.data = "ok"
        self.publisher_.publish(msg)
        self.get_logger().info("ok")
                
def main(args=None):
    rclpy.init(args=args)
    node = CaptureStatusNode()
    rclpy.spin(node)
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
