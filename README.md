# Eagle RL: teaching an AI to land on the Moon

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-backend-000000?logo=flask&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-dashboard-F7DF1E?logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-CSS3-E34F26?logo=html5&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-0081A5)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-4C9A2A)
![Score](https://img.shields.io/badge/Test%20score-264.8%20%C2%B1%2020-brightgreen)
![Solved](https://img.shields.io/badge/Solved-20%2F20-brightgreen)

A reinforcement-learning agent that learned to land a rocket on the Moon. It is trained with **PPO** on Gymnasium's **LunarLander-v3** and comes with a live browser dashboard that streams real simulator frames, the eight numbers the agent sees, and its learning curve. The saved model is then tested honestly on fresh attempts.

**Result:** mean score **264.8 ± 20** over 20 fresh attempts, with **20 of 20 solved** (200 or more), 0 crashes and 0 timeouts.

![Landing demo](clips/demo.gif)
