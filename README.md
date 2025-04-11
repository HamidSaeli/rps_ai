# AI-Powered Rock Paper Scissors Game with Voice & Emotion Recognition

📌 **Overview**  
This project is an AI-powered Rock, Paper, Scissors game that uses LSTM neural networks to predict the player's next move and adapt to their playstyle. It features voice recognition, emotion detection, and a graphical user interface (GUI).

🚀 **Features**
- 🎮 **Interactive GUI built with Tkinter**: A user-friendly and interactive graphical interface for playing the game.
- 🧠 **AI-powered predictions using LSTM neural networks**: The AI learns from your moves and predicts your next move based on past patterns.
- 🎤 **Voice recognition for hands-free gameplay**: Play the game using voice commands like "Rock", "Paper", or "Scissors".
- 📷 **Emotion detection via webcam**: The game detects your emotions through webcam analysis and adjusts the AI's behavior accordingly.
- 📊 **Player statistics with data visualization**: Visualize your playstyle over time with a bar graph of your most common moves.
- 🔊 **Sound effects for a more immersive experience**: Enjoy sound effects for win, loss, and draw situations.

🏗️ **Technologies Used**
- Python 🐍
- TensorFlow & Keras 🧠
- OpenCV 🎥
- SpeechRecognition 🎙️
- Pyttsx3 (Text-to-Speech) 🔊
- Pygame 🎵
- Matplotlib 📊
- Tkinter 🖥️

🎯 **How It Works**
- The player selects their move via the GUI or voice command.
- The AI predicts the player's next move based on patterns learned from previous moves using LSTM networks.
- The AI responds strategically to maximize its chances of winning.
- The game tracks statistics, displays insights into the player's behavior, and shows a graph of their most common moves.

🛠️ **Installation**
1️⃣ **Prerequisites**  
Ensure you have Python 3.8+ installed. Then, install dependencies:

```bash
pip install numpy opencv-python tensorflow speechrecognition pyttsx3 pygame matplotlib colorama tqdm
2️⃣ Run the Game
Execute the following command to start the game:


python main.py
🎤 Voice Commands

Say "Rock", "Paper", or "Scissors" to make a move.

If unrecognized, the system will prompt you for another input.

📊 Playstyle Insights

The game analyzes your moves over time and displays which move you favor the most.

Use this insight to break patterns and trick the AI!

🔧 Future Improvements

🤖 More advanced AI for better predictions.

🎭 Emotion-based AI reactions.

🌐 Web-based version using Flask or Streamlit.

📜 License
This project is licensed under the MIT License.

🤝 Contributions
Feel free to fork, modify, and submit PRs to improve the project!

🚀 Enjoy playing against AI and testing your strategy skills!
