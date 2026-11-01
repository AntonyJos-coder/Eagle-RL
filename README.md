# RocketRL Live Dashboard

## Windows PowerShell (from your existing project)

```powershell
cd C:\Users\ADMIN\Downloads\RocketRL_Portfolio_Starter\rocketrl
..\.venv\Scripts\python.exe -m pip install flask opencv-python-headless
..\.venv\Scripts\python.exe app.py
```

Copy `app.py` and the `templates` folder into the existing `rocketrl` directory first.
Then open http://127.0.0.1:5000

This dashboard trains **new** PPO models and saves them as `models/baseline_live.zip` and `models/improved_live.zip`; it does not load or overwrite your old `models/baseline.zip`.

The `improved` shaping in this dashboard is a new example and may differ from the `improved` reward variant in your existing `rocketrl.py`. Reward values for differently shaped variants are not directly comparable; evaluate both on the original environment for a fair comparison.

The video is genuine Gymnasium RGB frames captured during training. Rendering can slow training. Episode score >=200 is a proxy success convention, not a confirmed physical landing test.
