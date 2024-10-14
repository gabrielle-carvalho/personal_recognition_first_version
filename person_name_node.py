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

class PersonNameNode(Node):
    def __init__(self):
        super().__init__('person_name_node')
        self.publisher_ = self.create_publisher(String, 'person_name', 10)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.get_logger().info("Speech Recognition Node has started")
        
        # Caminhos para os arquivos de áudio
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.audio_file_path_waiting = os.path.join(current_directory, "audios/microphoneOn.mp3")
        self.audio_file_path_detected = os.path.join(current_directory, "audios/keywordDetected.mp3")
        self.audio_file_path_dont_understand = os.path.join(current_directory, "audios/dontUnderstand.mp3")
        self.audio_file_path_error = os.path.join(current_directory, "audios/erro.mp3")
        
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
            audio = self.recognizer.listen(source)  # Corrigido para usar self.recognizer

        # Convertendo o áudio capturado em um formato que podemos manipular com NumPy
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16)

        # Aplicando a redução de ruído
        reduced_noise = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)

        # Reconstruindo o áudio com o ruído reduzido
        audio_reduced = sr.AudioData(reduced_noise.tobytes(), source.SAMPLE_RATE, 2)

        try:
            keyword = self.recognizer.recognize_google(audio_reduced)  # Corrigido para usar self.recognizer

            # Verifica se a palavra-chave é "ok google"
            if keyword.lower() == "ok google":
                keyword = "Ok Bill"  # Altera o texto para "Ok Bill"
                self.get_logger().info(f"You said: {keyword}")  # Imprime a entrada reconhecida

            if "ok bill" in keyword.lower():
                self.play_audio(self.audio_file_path_detected)  # Reproduz o áudio da palavra-chave reconhecida
                self.get_logger().info("Keyword recognized. Asking for the name...")
                self.play_audio(self.audio_file_path_waiting) 
                self.ask_for_name()
            else:
                self.get_logger().info("Keyword not recognized.")
                self.play_audio(self.audio_file_path_error)  # Reproduz o áudio de erro

        except sr.UnknownValueError:
            self.get_logger().info("I don't understand you. Please, repeat.")
            self.play_audio(self.audio_file_path_dont_understand)  # Reproduz o áudio de não entendimento


    def ask_for_name(self):
        self.speak("What's your name?")

        with sr.Microphone() as source:
            self.get_logger().info("Waiting for your name...")
            audio = self.recognizer.listen(source)  # Corrigido para self.recognizer

        # Convertendo o áudio capturado em um formato que podemos manipular com NumPy
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16)

        # Aplicando a redução de ruído
        reduced_noise = nr.reduce_noise(y=audio_data, sr=source.SAMPLE_RATE)

        # Reconstruindo o áudio com o ruído reduzido
        audio_reduced = sr.AudioData(reduced_noise.tobytes(), source.SAMPLE_RATE, 2)

        try:
            name = self.recognizer.recognize_google(audio_reduced)  # Corrigido para self.recognizer
            self.get_logger().info(f"Received name: {name}")
            self.publish_name(name)

        except sr.UnknownValueError:
            self.get_logger().info("I don't understand you. Please, repeat.")
            self.play_audio(self.audio_file_path_dont_understand)  # Reproduz o áudio de não entendimento


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
        self.publisher_.publish(msg)
        self.get_logger().info(f"Name '{name}' published to the topic 'person_name'.")

def main(args=None):
    rclpy.init(args=args)
    voice_node = PersonNameNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()