#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import speech_recognition as sr
from std_msgs.msg import String
import noisereduce as nr
import numpy as np

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')
        self.publisher_ = self.create_publisher(String, 'speech_recognition', 10)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.get_logger().info("Person Name Node has started")
        self.timer = self.create_timer(5, self.recognize_speech)
    
    def recognize_speech(self):
        with self.microphone as source:
            self.get_logger().info("Adjusting for ambient noise...")
            audio = self.recognizer.listen(source)
        
        # Converter o áudio para dados em numpy
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
        
        # Aplicar redução de ruído usando noisereduce
        reduced_noise = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)
        
        # Converter de volta para áudio
        audio_reduced = sr.AudioData(reduced_noise.tobytes(), source.SAMPLE_RATE, 2)
        
        try:
            # Usar o aúdio filtrado para reconhecimento
            text = self.recognizer.recognize_google(audio_reduced)
            self.get_logger().info(f"Recognized: {text}")
            msg = String()
            msg.data = text
            self.publisher_.publish(msg)
        except sr.UnknownValueError:
            self.get_logger().warn("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            self.get_logger().error(f"Could not request results; {e}")
        
def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()