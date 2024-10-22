#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
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


       # Inicialização do detector de faces e dos modelos
       self.detector = dlib.get_frontal_face_detector()
       self.sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
       self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')


       self.create_subscription(String, 'turned_around', self.turned_callback, 10)


       self.labels, self._descriptors = self.prepare_training_data("images")


       self.name_publisher = self.create_publisher(String, 'recognized_person', 10)
       self.count_publisher = self.create_publisher(String, 'count_people', 10)


       self.get_logger().info("Face Recognition Node has Started")


       os.makedirs('predict', exist_ok=True)


       self.started = False
       self.recognition_enabled = False  # Active recognition
       self.total_faces_detected = 0
       self.detection_count = {}  # Contar deteções para cada pessoa


   def turned_callback(self, msg):
       self.get_logger().info("turned_callback called.")
       self.get_logger().info("message received: " + msg.data)


       if not self.started:
           self.get_logger().info("Preparing training data")
           self.labels, self._descriptors = self.prepare_training_data("images")
           self.started = True  # Preparação completa


           self.recognition_enabled = True
           self.get_logger().info(f'Recognition enabled: {self.recognition_enabled}')
           self.update_frame()
          
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


               if len(faces) == 0:
                   self.get_logger().warning(f"No faces detected in image {image_path}.")
                   continue


               for face in faces:
                   shape = self.sp(gray, face)
                   face_descriptor = self.facerec.compute_face_descriptor(img, shape)
                   descriptors.append(np.array(face_descriptor))
                   labels.append(label)


       self.get_logger().info(f"Total labels: {len(labels)}, Total descriptors: {len(descriptors)}")
       return labels, descriptors


   def recognition(self, msg):
       if self.recognition_enabled == False:
           self.get_logger().info("Recognition is not enabled.")
           return


       bridge = CvBridge()
      
       try:
           frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
       except Exception as e:
           self.get_logger().error(f"Failed to convert image: {e}")
           return


       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = self.detector(gray)
       current_faces_detected = len(faces)


       self.get_logger().info(f"Detected {current_faces_detected} faces on screen.")


       if current_faces_detected == 0:
           return


       for face in faces:
           shape = self.sp(gray, face)
           face_descriptor = self.facerec.compute_face_descriptor(frame, shape)


           if face_descriptor is None:
               self.get_logger().warning("Face descriptor could not be computed.")
               continue 


           name = self.match_face(np.array(face_descriptor), self.labels, self._descriptors)


           # Inicializar contador de detecções para o nome reconhecido
           if name not in self.detection_count:
               self.detection_count[name] = 0


           if name != "Unknown":
               self.detection_count[name] += 1
               self.get_logger().info(f"Recognized: {name}, Count: {self.detection_count[name]}")
               self.save_recognized_face(frame, name, face)  # Passando a face


               if self.detection_count[name] == 8:
                   self.name_publisher.publish(String(data=name))
                   self.get_logger().info(f"Published recognized name: {name}")
                   self.count_publisher.publish(String(data=str(self.total_faces_detected)))


                   self.detection_count[name] = 0  # Reinitialize counter after publisher
           else:
               # Para rostos desconhecidos
               self.detection_count[name] = 0
               self.save_recognized_face(frame, "Unknown", face)  # Passando a face para desconhecidos


           # Desenhar a bounding box na imagem
           x, y, w, h = face.left(), face.top(), face.width(), face.height()
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
           cv2.putText(frame, name if name != "Unknown" else "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


       self.total_faces_detected = current_faces_detected
       self.get_logger().info(f"Total faces detected on screen: {self.total_faces_detected}")


   def save_recognized_face(self, frame, name, face):
       if frame is None or not isinstance(frame, np.ndarray):
           self.get_logger().error("Invalid frame, cannot save.")
           return


       # Desenhar a bounding box na imagem
       x, y, w, h = face.left(), face.top(), face.height(), face.width()
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


       # Adicionar texto na imagem
       cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


       file_name = os.path.join('predict', f"{name}.jpg")
       count = 1
       while os.path.exists(file_name):
           file_name = os.path.join('predict', f"{name}_{count}.jpg")
           count += 1


       self.get_logger().info(f"Saving image to: {file_name}")


       if cv2.imwrite(file_name, frame):
           self.get_logger().info(f"Saved recognized face: {file_name}")
       else:
           self.get_logger().error(f"Failed to save image: {file_name}")


   def match_face(self, face_descriptor, labels, descriptors):
       if len(descriptors) == 0:
           self.get_logger().warning("No face descriptors available for matching.")
           return "Unknown"
      
       distances = [distance.euclidean(face_descriptor, descriptor) for descriptor in descriptors]
       min_distance = min(distances)


       if min_distance < 0.6:  # Threshold
           index = distances.index(min_distance)
           return labels[index]
      
       return "Unknown"
class App:
   def __init__(self, master, people_recognition_node):
       self.master = master
       self.people_recognition_node = people_recognition_node
       self.frame_counter = 0


       self.camera_label = Label(self.master)
       self.camera_label.pack()


       self.cap = None  # Inicializa como None
       self.update_frame()

  def turned_callback(self, msg):
      self.get_logger().info("turned_callback called.")
      self.get_logger().info("message received: " + msg.data)
  
      if not self.started:
          self.get_logger().info("Preparing training data")
          self.labels, self._descriptors = self.prepare_training_data("images")
          self.started = True  # Preparação completa
          self.recognition_enabled = True
  
          # Inicialize a câmera aqui, apenas se ainda não estiver aberta
          if self.cap is None:
              self.cap = cv2.VideoCapture(0)  # Abre a câmera (ID 0, para câmera padrão)
              if not self.cap.isOpened():
                  self.get_logger().error("Failed to open camera.")
                  return
              else:
                  self.get_logger().info("Camera has been successfully opened.")
          
          self.update_frame()



   def update_frame(self):
       if self.cap is None or not self.cap.isOpened():
           self.people_recognition_node.get_logger().error("Camera is not available.")
           return


       ret, frame = self.cap.read()
       if not ret:
           self.people_recognition_node.get_logger().error("Failed to capture video frame.")
           return


       frame = cv2.flip(frame, 1)
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = self.people_recognition_node.detector(gray)


       self.frame_counter += 1
      
       if self.frame_counter % 15 == 0:  # Capture 15 frames
           if self.people_recognition_node.recognition_enabled == True:
               bridge = CvBridge()
               msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
               self.people_recognition_node.recognition(msg)
           else:
               self.people_recognition_node.get_logger().info("Recognition is not enabled.")


       for face in faces:
           x, y, w, h = face.left(), face.top(), face.height(), face.width()
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
           cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


       if len(faces) == 0:
           height, width, _ = frame.shape
           cv2.putText(frame, "Unknown", (width // 2 - 40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


       self.master.after(10, self.update_frame)


def main(args=None):
   rclpy.init(args=args)
  
   people_recognition_node = PeopleRecognition()


   root = Tk()
   app_instance = App(root, people_recognition_node)
   people_recognition_node.get_logger().info("PeopleRecognition Node is up and running.")
  
   # Spin the node in a separate thread
   import threading
   threading.Thread(target=rclpy.spin, args=(people_recognition_node,), daemon=True).start()
  
   root.mainloop()


   if app_instance.cap is not None:
     app_instance.cap.release()
   rclpy.shutdown()
   cv2.destroyAllWindows()


if __name__ == '__main__':
   main()
