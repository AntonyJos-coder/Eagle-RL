# Eagle RL: Teaching an AI to Land a Rocket on the Moon

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-backend-000000?logo=flask&logoColor=white)
![PPO](https://img.shields.io/badge/PPO-Stable--Baselines3-4C9A2A)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-0081A5)
![Pillow](https://img.shields.io/badge/Pillow-JPEG%20streaming-8A2BE2)
![HTML5](https://img.shields.io/badge/HTML5-CSS3-E34F26?logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-dashboard-F7DF1E?logo=javascript&logoColor=black)
![Score](https://img.shields.io/badge/Test%20score-264.8%20%C2%B1%2020-brightgreen)
![Solved](https://img.shields.io/badge/Solved-20%2F20-brightgreen)

A reinforcement-learning agent that learned to land a rocket on the Moon. It is trained with **PPO (Stable-Baselines3)** on Gymnasium's **LunarLander-v3**, and a **Flask** server streams the training live to a browser dashboard built with **HTML, CSS and JavaScript**. The saved model is then tested on fresh attempts to check that the landings are real.

![Landing demo](clips/demo.gif)

## 🎯 Problem It Solves

A rocket lander has to slow down, stay upright and touch down gently on a small pad. Writing rules by hand for this is hard, and training a model often feels like a black box: you start it, wait, and only see a number at the end.

Eagle RL lets the AI learn the landing by trial and error, and lets you watch and measure the learning while it happens.

## 💡 Solution

The agent practises thousands of landings, is rewarded for good behaviour and penalised for crashes, and slowly improves. The dashboard makes that process visible:

1. Trains PPO on 16 simulators at once.
2. Streams real simulator frames to the browser.
3. Shows the eight numbers the AI sees and the engine it fires.
4. Plots the learning curve and the average score.
5. Keeps the best model seen and stops early once the average reaches 250.
6. Tests the saved model on fresh seeded attempts and reports the result.
7. Saves a GIF of a landing.

## 🖥️ Dashboard Features

| Feature | Description |
|---|---|
| 🎥 Flight camera | Live frames from the real simulator, framed like a lunar module window |
| 📡 Lander telemetry | The eight observation numbers (position, speeds, tilt, spin, leg contact) and which engine is firing |
| 📈 Learning curve | Real score per episode with a moving average and the 200 solved line |
| 🏁 Training metrics | Timesteps, episodes, mean score of the last 50, solved rate and steps per second |
| ♻️ Continue training | Resume from the saved model and keep improving it |
| ✅ Honest evaluation | Tests the saved model on 10 to 100 fresh attempts and reports mean, spread, solved, landed, crashed and timed out |
| 🎞️ GIF clips | Saves the first replay as a GIF for your portfolio |

## ⚙️ How It Works

```
Browser dashboard  →  Flask API  →  PPO training (16 simulators)  →  Best model saved
        ▲                                   │
        └────── JPEG frames + live stats ───┘
                                            ▼
                              Evaluation on fresh attempts  →  Report
```

## 🏗️ System Architecture

```
                  User
                    │
                    ▼
        ┌───────────────────────┐
        │  Dashboard            │
        │  HTML, CSS, JS        │
        └───────────┬───────────┘
                    │  REST + video stream
                    ▼
        ┌───────────────────────┐
        │  Flask (Python)       │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 ┌─────────────┐        ┌───────────────┐
 │ PPO         │        │ Pillow        │
 │ Stable-     │        │ frames → JPEG │
 │ Baselines3  │        │ and GIF       │
 └──────┬──────┘        └───────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ Gymnasium           │
 │ LunarLander-v3      │
 │ (Box2D physics)     │
 └─────────────────────┘
```

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python** | The whole backend |
| **Flask** | Web server, status API and video stream |
| **PPO (Stable-Baselines3)** | The learning algorithm; the policy is a small neural network (two layers of 64 units) running on PyTorch |
| **Gymnasium + Box2D** | The LunarLander-v3 game and its physics |
| **Pillow** | Converts simulator frames to JPEG for the browser and builds GIF clips |
| **HTML, CSS, JavaScript** | The dashboard, with no framework; the chart is drawn on a canvas |

## 📊 Results

Evaluated over 20 fresh attempts on the raw game, using fixed seeds and a deterministic policy:

| Mean score | Solved (200 or more) | Landed | Crashed | Timed out |
|---|---|---|---|---|
| **264.8 ± 20.0** | **20 / 20** | 20 | 0 | 0 |

An earlier checkpoint scored 150.2 ± 79.6 (4 of 20 solved). Continuing training from the saved model raised it to the result above.

## 📂 Project Structure

```
Eagle-RL/
│
├── app.py              # training, evaluation and the Flask server
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html      # the dashboard
│
├── models/
│   └── lander.zip      # the trained model
│
└── clips/
    └── demo.gif        # a saved landing
```

## 🚀 Getting Started

1. **Clone the repository**
```
git clone https://github.com/AntonyJos-coder/Eagle-RL.git
cd Eagle-RL
```

2. **Install dependencies**
```
pip install -r requirements.txt
```
If Box2D fails to install on Windows, run `pip install swig` first, then install again.

3. **Start the app**
```
python app.py
```

4. **Open** http://127.0.0.1:5000

5. **Use it**
- Set **1M** timesteps and click **Start training**. It stops early if the average reaches 250.
- Tick **Continue from the saved model** to keep training an existing model.
- Click **Evaluate** (20 attempts) to test it, then **Watch saved model** to see it land.

Training takes roughly 10 to 45 minutes per 1M steps on a CPU, depending on your machine.

## 📚 What I Learned

- Reinforcement learning with PPO and how the reward shapes behaviour
- Why more parallel simulators give PPO better data per update
- Evaluating a model honestly on fresh seeded attempts instead of trusting training scores
- Streaming live video and stats from a Python server to a browser
- Running training in a background thread while keeping a web app responsive
- Debugging real issues: mismatched render modes across environments and NumPy values breaking JSON
- Building a dashboard with plain HTML, CSS and JavaScript
- Git and GitHub project management

## 🔮 Future Improvements

- Compare custom reward designs and show reward hacking on purpose
- Side-by-side comparison of several saved models
- Exportable evaluation reports
- Harder environments such as continuous-control LunarLander
- Cloud deployment

## 📌 Project Purpose

This project explores how reinforcement learning, a web server and a live dashboard can be combined into a tool that makes training visible, and how to test a trained agent honestly.

## 👨‍💻 Author

**Antony Jos**

Computer Engineer | App Development | Web Development | Data Analytics
