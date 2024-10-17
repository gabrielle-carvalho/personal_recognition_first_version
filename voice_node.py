#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
from gtts import gTTS
import os
from playsound import playsound
import noisereduce as nr
import numpy as np

class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')

        # Publishers
        self.publisher_name = self.create_publisher(String, 'person_name', 10)
        self.publisher_training_status = self.create_publisher(String, 'training_finished', 10)
        self.publisher_next_step = self.create_publisher(String, 'next_capture_step', 10)
        
        # Subscribers
        self.subscription_capture = self.create_subscription(String, '/capture_status', self.capture_status_callback, 10)

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.get_logger().info("Voice Node has started")

        self.keyword_detected = False  # Flag para garantir que a palavra-chave só seja detectada uma vez
        self.current_step_index = 0
        self.capture_status_ok = False  # Flag para aguardar o "OK"
        
        # Define capture steps
        self.capture_steps = [
            "straight_to_camera",
            "straight_to_camera_down",
            "straight_to_camera_up",
            "turn_left_up",
            "turn_left_down",
            "turn_right_up",
            "turn_right_down"
        ]
        
        # Caminhos para os arquivos de áudio
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.audio_file_path_waiting = os.path.join(current_directory, "audios/microphoneOn.mp3")
        self.audio_file_path_detected = os.path.join(current_directory, "audios/keywordDetected.mp3")
        self.audio_file_path_dont_understand = os.path.join(current_directory, "audios/dontUnderstand.mp3")
        self.audio_file_path_error = os.path.join(current_directory, "audios/erro.mp3")
        self.audio_file_path_training_finished = os.path.join(current_directory, "audios/treinamentoFinalizado.mp3")
        self.audio_file_path_whats_your_name = os.path.join(current_directory, "audios/whatsYourName.mp3")
        self.audio_file_path_straight_to_camera = os.path.join(current_directory, "audios/straightToCamera.mp3")
        self.audio_file_path_straight_to_camera_down = os.path.join(current_directory, "audios/straightToCameraDown.mp3")
        self.audio_file_path_straight_to_camera_up = os.path.join(current_directory, "audios/straightToCameraUp.mp3")
        self.audio_file_path_turn_left_down = os.path.join(current_directory, "audios/turnLeftDown.mp3")
        self.audio_file_path_turn_left_up = os.path.join(current_directory, "audios/turnLeftUp.mp3")
        self.audio_file_path_turn_right_up = os.path.join(current_directory, "audios/turnRightUp.mp3")
        self.audio_file_path_turn_right_down = os.path.join(current_directory, "audios/turnRightDown.mp3")

        # Definindo o dicionário de arquivos de áudio
        self.audio_files = {
            "straight_to_camera": self.audio_file_path_straight_to_camera,
            "straight_to_camera_down": self.audio_file_path_straight_to_camera_down,
            "straight_to_camera_up": self.audio_file_path_straight_to_camera_up,
            "turn_left_up": self.audio_file_path_turn_left_up,
            "turn_left_down": self.audio_file_path_turn_left_down,
            "turn_right_up": self.audio_file_path_turn_right_up,
            "turn_right_down": self.audio_file_path_turn_right_down,
        }
        
        self.get_logger().info("Voice node started. Waiting for keyword...")
        self.timer = self.create_timer(2.0, self.listen_for_keyword)  # Timer para repetição do processo

    
    def ask_for_name(self):
        while True:
            self.play_audio(self.audio_file_path_whats_your_name)
            
            with sr.Microphone() as source:
                self.get_logger().info("Waiting for your name...")
                self.play_audio(self.audio_file_path_waiting)
                audio = self.recognizer.listen(source)  

            # Convertendo o áudio capturado em um formato que podemos manipular com NumPy   
            audio_data = np.frombuffer(audio.get_raw_data(), np.int16)

            # Aplicando a redução de ruído
            reduced_noise = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)

            # Reconstruindo o áudio com o ruído reduzido
            audio_reduced = sr.AudioData(reduced_noise.tobytes(), source.SAMPLE_RATE, 2)

            try:
                name = self.recognizer.recognize_google(audio_reduced)
                self.get_logger().info(f"Received name: {name}")
                
                # Pergunta se o nome está correto
                if self.confirm_name(name):
                    self.publish_name(name)
                    self.proceed_to_next_step()  # Chama a função para continuar os passos
                    break  # Sai do loop se o nome foi confirmado
                else:
                    self.get_logger().info("Name not confirmed. Asking again...")

            except sr.UnknownValueError:
                self.get_logger().info("I didn't catch that. Please, repeat your name.")
                self.play_audio(self.audio_file_path_dont_understand)
    
    
    def capture_status_callback(self, msg):
        if msg.data == "ok":
            self.capture_status_ok = True  # Define a flag como True quando o OK for recebido

            # Só avança quando recebe OK
            if self.current_step_index < len(self.capture_steps):
                current_step = self.capture_steps[self.current_step_index]
                self.play_audio(self.audio_files[current_step])
                self.get_logger().info(f"Playing audio for: {current_step}")

                # Avança para o próximo passo
                self.current_step_index += 1

                # Se todos os passos foram concluídos
                if self.current_step_index >= len(self.capture_steps):
                    self.get_logger().info("All capture steps completed.")
                    self.play_audio(self.audio_file_path_training_finished)  # Reproduz o áudio de finalização
                    self.publish_training_finished()  # Publica que o treinamento foi finalizado
            else:
                self.get_logger().info("No more steps to perform.")
                self.get_logger().info("Waiting for new instructions...")
                
                
    def confirm_name(self, name):
        self.speak(f"Did you say your name is {name}? Please, say yes or no")
    
        with sr.Microphone() as source:
            self.get_logger().info("Waiting for confirmation...")
            self.play_audio(self.audio_file_path_waiting)
            audio = self.recognizer.listen(source)
            
        # Convertendo o áudio capturado em um formato que podemos manipular com NumPy
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16)

        # Aplicando a redução de ruído
        reduced_noise = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)

        # Reconstruindo o áudio com o ruído reduzido
        audio_reduced = sr.AudioData(reduced_noise.tobytes(), source.SAMPLE_RATE, 2)
        
        try:
            response = self.recognizer.recognize_google(audio_reduced).lower()
            
            if "yes" in response:
                self.get_logger().info("User confirmed the name")
                return True
            elif "no" in response:
                self.get_logger().info("User did not confirm the name. Asking again.")
                return False
            else:
                self.get_logger().info("I don't understand. Please say yes or no.")
                self.play_audio(self.audio_file_path_dont_understand)  # Reproduz o áudio de não entendimento
                return False
            
        except sr.UnknownValueError:
            self.get_logger().info("I don't understand you. Please, repeat.")
            self.play_audio(self.audio_file_path_dont_understand)  # Reproduz o áudio de não entendimento
            return False
        
    def listen_for_keyword(self):
        if not self.keyword_detected:
            self.get_logger().info("Say the keyword...")
            
            with sr.Microphone() as source:
                self.get_logger().info("Say the keyword...")
                self.play_audio(self.audio_file_path_waiting)  # Reproduz o áudio de espera
                audio = self.recognizer.listen(source)

            # Convertendo o áudio capturado em um formato que podemos manipular com NumPy
            audio_data = np.frombuffer(audio.get_raw_data(), np.int16)

            # Aplicando a redução de ruído
            reduced_noise = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)

            # Reconstruindo o áudio com o ruído reduzido
            audio_reduced = sr.AudioData(reduced_noise.tobytes(), source.SAMPLE_RATE, 2)

            try:
                keyword = self.recognizer.recognize_google(audio_reduced)
                self.get_logger().info(f"Captured keyword: {keyword}")

                if "ok google" in keyword.lower() or "okay view" in keyword.lower() or "ok view" in keyword.lower():
                    keyword = "ok bill"

                if "ok bill" in keyword.lower():
                    self.keyword_detected = True  # Marca a palavra-chave como detectada
                    self.get_logger().info(f"Keyword '{keyword}' detected.")
                    self.play_audio(self.audio_file_path_detected)
                    self.ask_for_name()  # Continua para perguntar o nome
                else:
                    self.get_logger().info("Keyword not detected. Trying again.")
                    self.play_audio(self.audio_file_path_dont_understand)  # Reproduz áudio de não entendimento

            except sr.UnknownValueError:
                self.get_logger().info("I did not understand the keyword.")
                self.play_audio(self.audio_file_path_error)  # Reproduz áudio de erro
    

    def play_audio(self, audio_file_path):
        if os.path.exists(audio_file_path):
            playsound(audio_file_path)
        else:
            self.get_logger().warning(f"Audio file not found: {audio_file_path}")
    
    def play_audio_sequence(self):
        for step in self.capture_steps:
            # Espera receber "OK" do tópico /capture_status antes de prosseguir para o próximo passo
            while not self.capture_status_ok:
                rclpy.spin_once(self)  # Processa as mensagens recebidas para verificar o status de captura
            self.capture_status_ok = False  # Reseta o status para aguardar o próximo "OK"

            # Reproduz o áudio correspondente ao passo atual
            self.play_audio(self.audio_files[step])
            self.get_logger().info(f"Playing audio for: {step}")
            
    def proceed_to_next_step(self):
        self.get_logger().info("Proceeding to the next steps.")
        
        if self.current_step_index == 0:
            # No início, apenas loga que está aguardando a primeira mensagem OK
            self.get_logger().info("Waiting for 'OK' to proceed with the first capture step.")
        # O avanço agora será controlado pelo callback capture_status_callback

    def publish_name(self, name):
        msg = String()
        msg.data = name
        self.publisher_name.publish(msg)
        self.get_logger().info(f"Published name: {name}")
        
    def publish_training_finished(self):
        msg = String()
        msg.data = "Training finished"
        self.publisher_training_status.publish(msg)
        self.get_logger().info("Published training finished status.")
        
    def speak(self, text):
        # Certifique-se de que o diretório "audios" existe
        if not os.path.exists("audios"):
            os.makedirs("audios")

        tts = gTTS(text=text, lang="en")  # Mantenha o idioma como 'en' ou mude para 'pt' para português
        audio_file = "audios/tts_output.mp3"
        tts.save(audio_file)

        playsound(audio_file)

        os.remove(audio_file)
        

def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        voice_node.get_logger().info("Voice node terminated.")
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
