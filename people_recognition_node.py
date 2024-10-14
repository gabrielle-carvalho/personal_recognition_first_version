#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from tkinter import *

class PeopleRecognition(Node):
    def __init__(self):
        super().__init__("people_recognition_node")
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        self.labels, self._descriptors = self.prepare_training_data("images")
        self.name_publisher = self.create_publisher(String, 'recognized_person', 10)
        self.count_publisher = self.create_publisher(String, 'count_people', 10)
        self.get_logger().info("Face Recognition Node has Started")
        os.makedirs('predict', exist_ok=True)
        
        self.total_faces_detected = 0  

    def prepare_training_data(self, data_folder_path):
        labels = []
        descriptors = []

        if not os.path.exists(data_folder_path):
            self.get_logger().error(f"Data folder {data_folder_path} does not exist.")
            return labels, descriptors

        for label in os.listdir(data_folder_path):
            person_dir = os.path.join(data_folder_path, label)
            if not os.path.isdir(person_dir):
                self.get_logger().warning(f"{person_dir} is not a directory.")
                continue

            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    self.get_logger().warning(f"Could not read image {image_path}. Skipping.")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                for face in faces:
                    shape = self.sp(gray, face)
                    face_descriptor = self.facerec.compute_face_descriptor(img, shape)
                    descriptors.append(np.array(face_descriptor))
                    labels.append(label)

        return labels, descriptors

    def recognition_callback(self, msg):
        self.get_logger().info("Recognition callback called.")
        bridge = CvBridge()
        try:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        current_faces_detected = len(faces)  # Número de rostos detectados na tela

        self.get_logger().info(f"Detected {current_faces_detected} faces on screen.")

        for face in faces:
            shape = self.sp(gray, face)
            face_descriptor = self.facerec.compute_face_descriptor(frame, shape)
            name = self.match_face(np.array(face_descriptor), self.labels, self._descriptors)

            # Desenhar a bounding box na imagem
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name if name != "Unknown" else "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if name != "Unknown":
                self.get_logger().info(f"Recognized: {name}")
                self.name_publisher.publish(String(data=name))
                self.save_recognized_face(frame, name)

        self.total_faces_detected = current_faces_detected  # Atualiza o total com rostos detectados na tela
        self.get_logger().info(f"Total faces detected on screen: {self.total_faces_detected}")
        
        self.count_publisher.publish(String(data=str(self.total_faces_detected)))
        

    def save_recognized_face(self, frame, name):
        self.get_logger().info(f"Attempting to save recognized face for: {name}")

        if frame is None or not isinstance(frame, np.ndarray):
            self.get_logger().error("Invalid frame, cannot save.")
            return

        file_name = os.path.join('predict', f"{name}.jpg")
        count = 1
        while os.path.exists(file_name):
            file_name = os.path.join('predict', f"{name}_{count}.jpg")
            count += 1

        self.get_logger().info(f"Saving image to: {file_name}")

        try:
            if cv2.imwrite(file_name, frame):
                self.get_logger().info(f"Saved recognized face: {file_name}")
            else:
                self.get_logger().error(f"Failed to save image: {file_name}")
        except Exception as e:
            self.get_logger().error(f"Exception when saving image: {e}")

    def match_face(self, face_descriptor, labels, descriptors):
        distances = [distance.euclidean(face_descriptor, descriptor) for descriptor in descriptors]
        min_distance = min(distances)

        if min_distance < 1.0:  # Threshold
            index = distances.index(min_distance)
            self.get_logger().info(f"Recognized: {labels[index]} with distance: {min_distance}")
            return labels[index]

        self.get_logger().info(f"No match found, closest distance: {min_distance}")
        return "Unknown"

class App:
    def __init__(self, master, people_recognition_node):
        self.master = master
        self.people_recognition_node = people_recognition_node
        self.frame_counter = 0

        self.camera_label = Label(self.master)
        self.camera_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.people_recognition_node.get_logger().error("Failed to capture video frame.")
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.people_recognition_node.detector(gray)

        self.frame_counter += 1
        if self.frame_counter % 15 == 0:  # Capture every 15 frames
            bridge = CvBridge()
            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.people_recognition_node.recognition_callback(msg) 

        # Desenhar a bounding box e o nome para cada face detectada
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.master.after(10, self.update_frame)

def main(args=None):
    rclpy.init(args=args)
    
    people_recognition_node = PeopleRecognition()
    
    root = Tk()
    app_instance = App(root, people_recognition_node)
    root.mainloop()

    app_instance.cap.release()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

