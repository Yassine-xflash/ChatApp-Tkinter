import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import threading
import tempfile
import wave
import pyaudio
import os
import pyttsx3
import pygame
import soundfile as sf
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import speech_recognition as sr
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Application de Chat Pour les Sourds et Malvoyants")
        self.root.geometry("400x600")

        # Initialize pygame mixer
        pygame.mixer.init()

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()

        # Set TTS properties
        self.set_tts_properties()

        # Initialize login/signup frames
        self.create_login_signup_frames()

        # Initialize chat frame
        self.chat_frame = None
        self.profile_frame = None

    def create_login_signup_frames(self):
        # Login Frame
        self.login_frame = tk.Frame(self.root, bg="#ECE5DD")
        self.login_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.login_frame, text="Connexion", font=("Arial", 24)).pack(pady=20)

        tk.Label(self.login_frame, text="Nom d'utilisateur", font=("Arial", 14)).pack()
        self.login_username_entry = tk.Entry(self.login_frame, font=("Arial", 14))
        self.login_username_entry.pack(pady=5)

        tk.Label(self.login_frame, text="Mot de passe", font=("Arial", 14)).pack()
        self.login_password_entry = tk.Entry(self.login_frame, font=("Arial", 14), show="*")
        self.login_password_entry.pack(pady=5)

        self.login_voice_button = tk.Button(self.login_frame, text="Enregistrer la voix", command=self.record_login_voice)
        self.login_voice_button.pack(pady=10)

        self.login_button = tk.Button(self.login_frame, text="Connexion", font=("Arial", 14), command=self.login)
        self.login_button.pack(pady=20)

        tk.Button(self.login_frame, text="Inscription", font=("Arial", 14), command=self.show_signup).pack(pady=10)

        # Sign Up Frame
        self.signup_frame = tk.Frame(self.root, bg="#ECE5DD")

        tk.Label(self.signup_frame, text="Inscription", font=("Arial", 24)).pack(pady=20)

        tk.Label(self.signup_frame, text="Nom d'utilisateur", font=("Arial", 14)).pack()
        self.signup_username_entry = tk.Entry(self.signup_frame, font=("Arial", 14))
        self.signup_username_entry.pack(pady=5)

        tk.Label(self.signup_frame, text="Mot de passe", font=("Arial", 14)).pack()
        self.signup_password_entry = tk.Entry(self.signup_frame, font=("Arial", 14), show="*")
        self.signup_password_entry.pack(pady=5)

        self.signup_voice_button = tk.Button(self.signup_frame, text="Enregistrer la voix", command=self.record_signup_voice)
        self.signup_voice_button.pack(pady=10)

        self.signup_button = tk.Button(self.signup_frame, text="Inscription", font=("Arial", 14), command=self.signup)
        self.signup_button.pack(pady=20)

        tk.Button(self.signup_frame, text="Retour à la connexion", font=("Arial", 14), command=self.show_login).pack(pady=10)

        self.signup_voice_path = None
        self.login_voice_path = None

    def show_signup(self):
        self.login_frame.pack_forget()
        self.signup_frame.pack(fill=tk.BOTH, expand=True)

    def show_login(self):
        self.signup_frame.pack_forget()
        self.login_frame.pack(fill=tk.BOTH, expand=True)

    def record_signup_voice(self):
        self.signup_voice_path = self.record_voice("signup")
        messagebox.showinfo("Voix enregistrée", "L'enregistrement vocal pour l'inscription est terminé.")

    def record_login_voice(self):
        self.login_voice_path = self.record_voice("login")
        messagebox.showinfo("Voix enregistrée", "L'enregistrement vocal pour la connexion est terminé.")

    def record_voice(self, mode):
        fs = 44100  # Sample rate
        seconds = 10  # Duration of recording
        messagebox.showinfo("Enregistrement", f"Enregistrement de la voix pour {mode} pendant {seconds} secondes. Veuillez parler maintenant.")
        recorded_audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        file_path = temp_file.name
        sf.write(file_path, recorded_audio, fs, format='WAV', subtype='PCM_16')
        temp_file.close()
        return file_path

    def signup(self):
        username = self.signup_username_entry.get().strip()
        password = self.signup_password_entry.get().strip()

        if not username or not password or not self.signup_voice_path:
            messagebox.showerror("Erreur", "Tous les champs sont obligatoires, y compris l'enregistrement vocal.")
            return

        # Check if 'voices' directory exists, if not, create it
        voices_dir = "voices"
        if not os.path.exists(voices_dir):
            os.makedirs(voices_dir)

        # Save user details and voice
        user_data = {'username': username, 'password': password}
        with open(f"{voices_dir}/{username}.txt", 'w') as f:
            f.write(f"{username}\n{password}\n")
        
        # Use shutil to copy the file
        dest_voice_path = f"{voices_dir}/{username}.wav"
        shutil.copy(self.signup_voice_path, dest_voice_path)
        
        # Remove the temporary file
        os.remove(self.signup_voice_path)

        messagebox.showinfo("Succès", "Inscription réussie! Vous pouvez maintenant vous connecter.")
        self.show_login()

    def login(self):
        username = self.login_username_entry.get().strip()
        password = self.login_password_entry.get().strip()

        if not username or not password or not self.login_voice_path:
            messagebox.showerror("Erreur", "Tous les champs sont obligatoires, y compris l'enregistrement vocal.")
            return

        # Check user details and voice
        voices_dir = "voices"
        try:
            with open(f"{voices_dir}/{username}.txt", 'r') as f:
                stored_username, stored_password = f.read().splitlines()
        except FileNotFoundError:
            messagebox.showerror("Erreur", "Utilisateur non trouvé.")
            return

        if stored_username != username or stored_password != password:
            messagebox.showerror("Erreur", "Nom d'utilisateur ou mot de passe invalide.")
            return

        # Compare voice recordings
        if not self.compare_voices(f"{voices_dir}/{username}.wav", self.login_voice_path):
            messagebox.showerror("Erreur", "Échec de l'authentification vocale.")
            return

        messagebox.showinfo("Succès", f"Connexion réussie et bienvenue, {username}!")
        self.show_chat(username)

    def extract_mfcc(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc

    def compare_voices(self, stored_voice_path, login_voice_path):
        # Extract MFCC features from both stored and login voice files
        stored_mfcc = self.extract_mfcc(stored_voice_path)
        login_mfcc = self.extract_mfcc(login_voice_path)

        # Ensure both MFCC feature sets have the same shape
        min_frames = min(stored_mfcc.shape[1], login_mfcc.shape[1])
        stored_mfcc = stored_mfcc[:, :min_frames]
        login_mfcc = login_mfcc[:, :min_frames]

        # Compute the cosine similarity between the two MFCC feature sets
        similarity = cosine_similarity(stored_mfcc.T, login_mfcc.T)
        mean_similarity = np.mean(similarity)

        # Define a threshold for determining if the voices match
        threshold = 0.75  # Adjust this threshold as needed
        return mean_similarity > threshold
    
    def identify_speaker(self, file_path):
        input_features = self.extract_vocal_characteristics(file_path)
        voices_dir = "voices"
        min_distance = float('inf')
        identified_speaker = None
        for user_file in os.listdir(voices_dir):
            if user_file.endswith('.wav'):
                user_voice_path = os.path.join(voices_dir, user_file)
                user_features = self.extract_vocal_characteristics(user_voice_path)
                # Calculate Euclidean distance between input features and stored features
                distance = np.linalg.norm(input_features - user_features)
                if distance < min_distance:
                    min_distance = distance
                    identified_speaker = user_file.replace('.wav', '')
        threshold = 0.5  # Adjust threshold as needed
        if min_distance < threshold:
            return identified_speaker
        else:
            return None

    def extract_vocal_characteristics(self, file_path):
        features = []
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))  # Append mean of MFCCs
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))  # Append mean of chroma
        # Extract mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        features.extend(np.mean(mel_spectrogram, axis=1))  # Append mean of mel spectrogram
        # Extract spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(spectral_contrast, axis=1))  # Append mean of spectral contrast
        # Scale features
        scaler = StandardScaler()
        features = scaler.fit_transform(np.array(features).reshape(1, -1))
        return features

    def show_chat(self, username):
        if self.login_frame:
            self.login_frame.pack_forget()
        if self.signup_frame:
            self.signup_frame.pack_forget()

        self.create_chat_frame(username)

    def create_chat_frame(self, username):
        self.chat_frame = tk.Frame(self.root, bg="#ECE5DD")
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        # Profile Section
        self.profile_frame = tk.Frame(self.root, bg="#075E54")
        self.profile_frame.pack(side=tk.TOP, fill=tk.X)

        # Load and resize profile image
        self.profile_img = Image.open("profile.png")  # Replace with your profile image path
        self.profile_img = self.profile_img.resize((50, 50), Image.LANCZOS)
        self.profile_photo = ImageTk.PhotoImage(self.profile_img)

        self.profile_label = tk.Label(self.profile_frame, image=self.profile_photo, bg="#075E54")
        self.profile_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.profile_name = tk.Label(self.profile_frame, text=username, font=("Arial", 18), bg="#075E54", fg="white")
        self.profile_name.pack(side=tk.LEFT, pady=10)

        # Chat Display Area
        self.chat_frame = tk.Frame(self.root, bg="#ECE5DD")
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_text = tk.Text(self.chat_frame, bg="#ECE5DD", state=tk.DISABLED, wrap=tk.WORD)
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.scrollbar = tk.Scrollbar(self.chat_frame, command=self.chat_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_text['yscrollcommand'] = self.scrollbar.set

        # Message Entry and Send Button
        self.message_frame = tk.Frame(self.root, bg="#ECE5DD")
        self.message_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.message_entry = tk.Entry(self.message_frame, font=("Arial", 14))
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)

        # Load and resize send icon
        self.send_img = Image.open("send_icon.png")  # Replace with your send icon image path
        self.send_img = self.send_img.resize((30, 30), Image.LANCZOS)
        self.send_photo = ImageTk.PhotoImage(self.send_img)

        self.send_button = tk.Button(self.message_frame, image=self.send_photo, command=self.send_message, borderwidth=0)
        self.send_button.pack(side=tk.LEFT, padx=5)

        # Load and resize record icon
        self.record_img = Image.open("record_icon.png")  # Replace with your record icon image path
        self.record_img = self.record_img.resize((30, 30), Image.LANCZOS)
        self.record_photo = ImageTk.PhotoImage(self.record_img)

        self.record_button = tk.Button(self.message_frame, image=self.record_photo, command=self.start_recording, borderwidth=0)
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.is_recording = False
        self.recording_thread = None
        self.recorded_audio = []
        self.audio_file_path = None

    def set_tts_properties(self):
        # Set properties for text-to-speech
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'male' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    def send_message(self):
        message = self.message_entry.get()
        if message.strip():
            self.display_message("Vous", message)
            self.text_to_speech(message)
            self.message_entry.delete(0, tk.END)

    def text_to_speech(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.config(relief=tk.SUNKEN)
            self.recorded_audio = []
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
        else:
            self.is_recording = False
            self.record_button.config(relief=tk.RAISED)
            self.recording_thread.join()
            self.save_audio()

    def record_audio(self):
        fs = 44100  # Sample rate
        while self.is_recording:
            chunk = sd.rec(int(5 * fs), samplerate=fs, channels=2, dtype='float32')  # Record in 5 second chunks
            sd.wait()  # Wait until recording is finished
            self.recorded_audio.append(chunk)

    def save_audio(self):
        if self.recorded_audio:
            audio_data = np.concatenate(self.recorded_audio)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            self.audio_file_path = temp_file.name
            # Convert float32 array to int16
            scaled = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
            scipy.io.wavfile.write(self.audio_file_path, 44100, scaled)
            self.display_audio_message(self.identify_speaker(self.audio_file_path), self.audio_file_path)
            self.speech_to_text(self.audio_file_path)
            temp_file.close()

    def play_audio(self, file_path):
        wf = wave.open(file_path, 'rb')
        p = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            return (data, pyaudio.paContinue)

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        stream_callback=callback)

        stream.start_stream()

        while stream.is_active():
            self.root.update()

        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()

    # Define the Butterworth filter
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def normalize_audio(self, data):
        return np.int16((data / np.max(np.abs(data))) * 32767)

    def preprocess_audio(self, file_path):
        fs, data = wavfile.read(file_path)
        # Apply bandpass filter
        filtered_data = self.bandpass_filter(data, lowcut=300, highcut=3400, fs=fs, order=6)
        # Normalize the audio
        normalized_data = self.normalize_audio(filtered_data)
        # Save the preprocessed audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wavfile.write(temp_file.name, fs, normalized_data)
        return temp_file.name

    def speech_to_text(self, file_path, language='fr-FR'):
        recognizer = sr.Recognizer()
        audio_file_path = self.preprocess_audio(file_path)
        
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data, language=language)
            sender = self.identify_speaker(file_path)
            self.display_message(sender, text)
        except sr.UnknownValueError:
            self.display_message("Système", "Désolé, je n'ai pas compris l'audio.")
        except sr.RequestError:
            self.display_message("Système", "Impossible de demander des résultats au service de reconnaissance vocale.")
        finally:
            os.remove(audio_file_path)


    def display_audio_message(self, sender, file_path):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, f"{sender}: a envoyé un message audio\n")
        play_button = tk.Button(self.chat_text, text="Jouer l'audio", command=lambda: self.play_audio(file_path))
        play_button.pack()
        self.chat_text.window_create(tk.END, window=play_button)
        self.chat_text.insert(tk.END, "\n")
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def display_message(self, sender, message):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, f"{sender}: {message}\n")
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)

root = tk.Tk()
app = ChatApp(root)
root.mainloop()
