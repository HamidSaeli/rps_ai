import os
import time
import random
import numpy as np
import cv2
import tensorflow as tf
import speech_recognition as sr
import pyttsx3
import pygame
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque, Counter
from tqdm import tqdm
from colorama import Fore, Style, init
from tkinter import Tk, Button, Label, PhotoImage

# Initialize colorama
init(autoreset=True)

# Game settings
moves = ["Rock", "Paper", "Scissors"]
move_to_index = {"Rock": 0, "Paper": 1, "Scissors": 2}
index_to_move = {0: "Rock", 1: "Paper", 2: "Scissors"}

# AI settings
sequence_length = 5
hidden_units = 32
learning_rate = 0.01
MODEL_PATH = "rps_ai_model.h5"
HISTORY_FILE = "move_history.npy"

# Game statistics
user_moves = deque(maxlen=sequence_length)
ai_wins, player_wins, draws = 0, 0, 0

# Initialize pygame for sound
pygame.mixer.init()
win_sound = pygame.mixer.Sound("win.wav")
lose_sound = pygame.mixer.Sound("lose.wav")
draw_sound = pygame.mixer.Sound("draw.wav")

# Initialize text-to-speech
engine = pyttsx3.init()

# Load move history if available
def load_moves():
    return list(np.load(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else []

def save_moves():
    np.save(HISTORY_FILE, np.array(user_moves))

# Create or load model
def create_model():
    model = Sequential([
        LSTM(hidden_units, input_shape=(sequence_length, 1), return_sequences=False),
        Dense(3, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print(Fore.GREEN + "✅ Loaded trained AI model!")
    
    return model

model = create_model()

# AI decision logic
def get_winning_move(predicted_index):
    return {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}[index_to_move[predicted_index]]

def adaptive_learning_rate(epoch):
    return max(0.005, learning_rate * (0.95 ** epoch))

def get_reward(user_move, ai_move):
    if ai_move == user_move:
        return -1  # Draw
    elif get_winning_move(move_to_index[user_move]) == ai_move:
        return 1  # AI wins
    else:
        return -2  # AI loses

def analyze_player_moves():
    if len(user_moves) < 10:
        return "Not enough data yet!"
    move_counts = Counter(user_moves)
    most_common = max(move_counts, key=move_counts.get)
    return f"📊 Your most common move: {index_to_move[most_common]}"

# Emotion detection (Webcam)
def detect_emotion():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    cap.release()
    
    if frame is not None:
        return random.choice(["Happy", "Neutral", "Frustrated"])
    return "Unknown"

# Voice recognition input
def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(Fore.BLUE + "🎙️ Say your move (Rock, Paper, Scissors)...")
        try:
            audio = recognizer.listen(source, timeout=3)
            text = recognizer.recognize_google(audio).capitalize()
            if text in moves:
                return text
            else:
                return None
        except sr.UnknownValueError:
            return None

# Graph visualization
def show_graph():
    counts = Counter(user_moves)
    plt.bar(["Rock", "Paper", "Scissors"], [counts[move_to_index[m]] for m in moves])
    plt.xlabel("Move")
    plt.ylabel("Frequency")
    plt.title("Your Playstyle")
    plt.show()

# GUI Interface
def start_gui():
    def play(move):
        global ai_wins, player_wins, draws
        user_move = move
        user_moves.append(move_to_index[user_move])

        # AI Prediction
        if len(user_moves) >= sequence_length:
            X = np.array(user_moves).reshape(1, sequence_length, 1)
            predicted_probs = model.predict(X, verbose=0)
            predicted_index = np.argmax(predicted_probs)
            ai_move = get_winning_move(predicted_index)
        else:
            ai_move = random.choice(moves)

        # Determine game result
        if ai_move == user_move:
            draws += 1
            draw_sound.play()
            result_label.config(text="🔄 Draw!", fg="orange")
        elif get_winning_move(move_to_index[user_move]) == ai_move:
            ai_wins += 1
            lose_sound.play()
            result_label.config(text="😞 You Lost!", fg="red")
        else:
            player_wins += 1
            win_sound.play()
            result_label.config(text="🎉 You Won!", fg="green")

        stats_label.config(text=f"🤖 AI Wins: {ai_wins} | 🧑 Player Wins: {player_wins} | 🔄 Draws: {draws}")

    # GUI setup
    root = Tk()
    root.title("Rock Paper Scissors AI")

    Label(root, text="Choose your move:", font=("Arial", 14)).pack()
    Button(root, text="🪨 Rock", command=lambda: play("Rock")).pack()
    Button(root, text="📜 Paper", command=lambda: play("Paper")).pack()
    Button(root, text="✂ Scissors", command=lambda: play("Scissors")).pack()

    result_label = Label(root, text="", font=("Arial", 16))
    result_label.pack()
    stats_label = Label(root, text="", font=("Arial", 12))
    stats_label.pack()

    Button(root, text="📊 Show Playstyle Graph", command=show_graph).pack()
    root.mainloop()

# Start the game with GUI
start_gui()
