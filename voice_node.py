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
        # Publisher para o nome da pessoa
        self.publisher_name = self.create_publisher(String, 'person_name', 10)
        
        # Publisher para o status do treinamento
        self.publisher_training_status = self.create_publisher(String, 'training_finished', 10)
        
        # Subscrever para o fim do treinamento
        self.subscription = self.create_subscription(String, 'vision_training_status', self.training_callback, 10)
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.get_logger().info("Voice Node has started")
        
        # Caminhos para os arquivos de áudio
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.audio_file_path_waiting = os.path.join(current_directory, "audios/microphoneOn.mp3")
        self.audio_file_path_detected = os.path.join(current_directory, "audios/keywordDetected.mp3")
        self.audio_file_path_dont_understand = os.path.join(current_directory, "audios/dontUnderstand.mp3")
        self.audio_file_path_error = os.path.join(current_directory, "audios/erro.mp3")
        self.audio_file_path_training_finished = os.path.join(current_directory, "audios/treinamentoFinalizado.mp3")
        self.audio_file_path_whats_your_name = os.path.join(current_directory, "audios/whatsYourName.mp3")
        
        self.get_logger().info("Voice node started. Waiting for keyword...")
        self.timer = self.create_timer(2.0, self.listen_for_keyword)  # Timer para repetição do processo

    def play_audio(self, audio_file_path):
        if os.path.exists(audio_file_path):
            playsound(audio_file_path)
        else:
            self.get_logger().warning(f"Audio file not found: {audio_file_path}")
    
    def listen_for_keyword(self):
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
            # Imprimindo a palavra-chave reconhecida
            self.get_logger().info(f"Captured keyword: {keyword}")

            # Verifica se a palavra-chave é "ok bill"
            if "ok bill" in keyword.lower():
                self.play_audio(self.audio_file_path_detected)  # Reproduz o áudio da palavra-chave reconhecida
                self.get_logger().info("Keyword recognized. Asking for the name...")
                self.ask_for_name()
            else:
                self.get_logger().info("Keyword not recognized.")
                self.play_audio(self.audio_file_path_error)

        except sr.UnknownValueError:
            self.get_logger().info("I don't understand you. Please, repeat.")
            self.play_audio(self.audio_file_path_dont_understand)

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
                    break  # Sai do loop se o nome estiver correto

                
            except sr.UnknownValueError:
                self.get_logger().info("I don't understand you. Please, repeat.")
                self.play_audio(self.audio_file_path_dont_understand)

    def confirm_name(self, name):
        self.speak(f"Did you say your name is {name}? Please, sey yes or no")
    
        with sr.Microphone() as source:
            self.get_logger().info("Waiting dor confirmation...")
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
        
    def speak(self, text):
        # Certifique-se de que o diretório "audios" existe
        if not os.path.exists("audios"):
            os.makedirs("audios")

        tts = gTTS(text=text, lang="en")  # Mantenha o idioma como 'en' ou mude para 'pt' para português
        audio_file = "audios/tts_output.mp3"
        tts.save(audio_file)

        playsound(audio_file)

        os.remove(audio_file)

    def publish_name(self, name):
        msg = String()
        msg.data = name
        self.publisher_name.publish(msg)
        self.get_logger().info(f"Name '{name}' published to the topic 'person_name'.")

    def training_callback(self, msg):
        if msg.data == "fim do treinamento":
            self.play_audio(self.audio_file_path_training_finished)
            self.get_logger().info('Fim do treinamento capturado. Reproduzindo áudio e publicando mensagem.')
        
            msg = String()
            msg.data = "treinamento finalizado"
            self.publisher_training_status.publish(msg)    
    
def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
