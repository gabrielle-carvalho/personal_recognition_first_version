#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TurnAround(Node):
    def __init__(self):
        super().__init__("turn_around_node")
        
        self.turned_publisher = self.create_publisher(String, 'turned_around', 10)
        self.get_logger().info("Node has started")
        
        # Publica a mensagem a cada 2 segundos
        self.timer = self.create_timer(2.0, self.publish_message)

    def publish_message(self):
        msg = String()
        msg.data = "Turned around"
        self.turned_publisher.publish(msg)
        self.get_logger().info(f"Published: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    
    node = TurnAround()
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
