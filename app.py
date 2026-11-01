"""RocketRL Live: real LunarLander-v3 PPO training streamed to a browser, plus honest evaluation.
pip install flask "gymnasium[box2d]" stable-baselines3 pillow
"""
import math
import threading
import time
from io import BytesIO
from pathlib import Path

import gymnasium as gym
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent
MODELS, CLIPS = ROOT / 'models', ROOT / 'clips'
for d in (MODELS, CLIPS):
    d.mkdir(exist_ok=True)
MODEL = MODELS / 'lander.zip'
N_ENVS = 16  # the standard recipe for this game: many simulators feed each PPO update
app = Flask(__name__)
app.json.default = lambda o: o.item() if hasattr(o, 'item') else str(o)  # numpy values never break /api/state
lock = threading.RLock()
stop_event = threading.Event()
state = {
    'running': False, 'mode': 'idle', 'steps': 0, 'target': 0, 'progress': 0, 'episodes': 0,
    'reward': None, 'mean_reward': None, 'solved_rate': 0, 'fps': 0, 'message': 'Ready to train',
    'history': [], 'frame': None, 'obs': [0] * 8, 'action': 0, 'step_reward': 0,
    'elapsed': 0, 't0': 0, 'eval': None,
}
recent = []


def make_env():
    # rgb_array only draws when render() is called, and SB3 needs every env to share one render mode
    return gym.wrappers.RecordEpisodeStatistics(gym.make('LunarLander-v3', render_mode='rgb_array'))


def say(msg):
    with lock:
        state['message'] = msg


def publish(frame, obs=None, action=None, reward=None):
    jpg = None
    if frame is not None:
        buf = BytesIO()
        Image.fromarray(frame).save(buf, 'JPEG', quality=80)
        jpg = buf.getvalue()
    with lock:
        if jpg:
            state['frame'] = jpg
        if obs is not None:
            state.update(obs=[round(float(v), 2) for v in obs], action=int(action), step_reward=round(float(reward), 2))


class LiveCallback(BaseCallback):
    """Streams progress, keeps the best model seen, and stops early once the lander is solved."""

    def __init__(self, target):
        super().__init__()
        self.target, self.t0, self.last = target, time.monotonic(), 0.0
        self.best, self.early = -math.inf, False

    def _on_rollout_end(self):
        with lock:
            m = sum(recent) / len(recent) if len(recent) >= 50 else None
        if m is not None:
            if m > self.best:
                self.best = m
                self.model.save(str(MODEL))
            self.early = m >= 250

    def _on_step(self):
        if stop_event.is_set() or self.early:
            return False
        n = self.num_timesteps
        with lock:
            state.update(steps=n, progress=min(100, 100 * n / self.target),
                         fps=round(n / max(time.monotonic() - self.t0, 0.01)))
            for done, info in zip(self.locals['dones'], self.locals['infos']):
                if done and 'episode' in info:
                    score = float(info['episode']['r'])
                    recent.append(score)
                    del recent[:-50]
                    mean = round(sum(recent) / len(recent), 1)
                    state.update(episodes=state['episodes'] + 1, reward=round(score, 1), mean_reward=mean,
                                 solved_rate=round(100 * sum(s >= 200 for s in recent) / len(recent)))
                    state['history'] = (state['history'] + [{'reward': round(score, 1), 'mean': mean}])[-300:]
        if time.monotonic() - self.last >= 0.06:
            try:  # show simulator 0 of the 16
                publish(self.training_env.envs[0].render(), self.locals['new_obs'][0],
                        self.locals['actions'][0], self.locals['rewards'][0])
            except Exception:
                pass
            self.last = time.monotonic()
        return True


def finish():
    with lock:
        state.update(running=False, mode='idle', elapsed=int(time.monotonic() - state['t0']))


def run_train(target, resume):
    env = None
    try:
        env = DummyVecEnv([make_env] * N_ENVS)
        if resume and MODEL.exists():
            model = PPO.load(str(MODEL), env=env, device='cpu')
        else:
            model = PPO('MlpPolicy', env, verbose=0, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999,
                        gae_lambda=0.98, ent_coef=0.01, seed=42, device='cpu')
        say(f'Training live on {N_ENVS} simulators (showing simulator 1)')
        cb = LiveCallback(target)
        model.learn(total_timesteps=target, callback=cb)
        if cb.best == -math.inf:
            model.save(str(MODEL))
        best = f', best average {cb.best:.0f}' if cb.best > -math.inf else ''
        why = ' (average reached 250, stopped early)' if cb.early else ' (stopped by you)' if stop_event.is_set() else ''
        say(f'Saved models/lander.zip{best}{why}')
    except Exception as exc:
        say(f'Error: {type(exc).__name__}: {exc}')
    finally:
        if env is not None:
            env.close()
        finish()


def summary(res):
    s = [r['score'] for r in res]
    m = sum(s) / len(s)
    count = lambda k: sum(r['end'] == k for r in res)
    return {'results': res, 'n': len(s), 'mean': round(m, 1),
            'std': round(math.sqrt(sum((v - m) ** 2 for v in s) / len(s)), 1),
            'solved': sum(v >= 200 for v in s), 'landed': count('landed'),
            'crashed': count('crashed'), 'timeout': count('timeout')}


def run_play(n, evaluate, record):
    env, res, clip = None, [], []
    try:
        if not MODEL.exists():
            raise FileNotFoundError('No saved model yet. Train one first.')
        env = make_env()
        model = PPO.load(str(MODEL), device='cpu')
        every = 4 if evaluate else 2
        for i in range(n):
            if stop_event.is_set():
                break
            obs, _ = env.reset(seed=1000 + i)
            score, k, end = 0.0, 0, 'timeout'
            while not stop_event.is_set():
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                score, k = score + float(r), k + 1
                if k % every == 0:
                    frame = env.render()
                    publish(frame, obs, action, r)
                    if record and not evaluate and i == 0 and len(clip) < 450:
                        clip.append(Image.fromarray(frame).resize((450, 300)))
                    if not evaluate:
                        time.sleep(0.04)
                if term or trunc:
                    end = ('landed' if r >= 100 else 'crashed') if term else 'timeout'
                    break
            res.append({'score': round(score, 1), 'end': end})
            with lock:
                state.update(episodes=len(res), reward=round(score, 1), progress=100 * len(res) / n)
                if evaluate:
                    state['eval'] = summary(res)
            say(f'{"Evaluating" if evaluate else "Replay"} attempt {len(res)} of {n}: {end}, score {score:.1f}')
            if clip and i == 0:
                pal = [f.convert('P', palette=Image.ADAPTIVE, colors=64) for f in clip]
                pal[0].save(CLIPS / f'lander-{int(time.time())}.gif', save_all=True,
                            append_images=pal[1:], duration=40, loop=0, optimize=True)
                clip = []
        if evaluate and res:
            say(f'Evaluation finished: {len(res)} attempts')
    except Exception as exc:
        say(f'Error: {type(exc).__name__}: {exc}')
    finally:
        if env is not None:
            env.close()
        finish()


def launch(fn, args, **updates):
    with lock:
        if state['running']:
            return jsonify(error='A task is already running. Stop it first.'), 409
        stop_event.clear()
        recent.clear()
        state.update(running=True, steps=0, progress=0, episodes=0, reward=None, mean_reward=None,
                     solved_rate=0, fps=0, history=[], frame=None, eval=None, elapsed=0,
                     t0=time.monotonic(), **updates)
        threading.Thread(target=fn, args=args, daemon=True).start()
    return jsonify(ok=True)


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/state')
def api_state():
    with lock:
        d = {k: v for k, v in state.items() if k not in ('frame', 't0')}
        if state['running']:
            d['elapsed'] = int(time.monotonic() - state['t0'])
    return jsonify(d)


@app.post('/api/start')
def api_start():
    data = request.get_json(silent=True) or {}
    try:
        steps = int(data.get('steps', 1000000))
    except (ValueError, TypeError):
        steps = 0
    if not 16384 <= steps <= 2_000_000:
        return jsonify(error='Training steps must be 16,384 to 2,000,000'), 400
    return launch(run_train, (steps, bool(data.get('resume'))), mode='training', target=steps,
                  message='Starting PPO training...')


@app.post('/api/watch')
def api_watch():
    data = request.get_json(silent=True) or {}
    return launch(run_play, (5, False, bool(data.get('record'))), mode='watching', target=0,
                  message='Loading saved model...')


@app.post('/api/eval')
def api_eval():
    try:
        n = int((request.get_json(silent=True) or {}).get('n', 10))
    except (ValueError, TypeError):
        n = 0
    if not 10 <= n <= 100:
        return jsonify(error='Evaluate 10 to 100 attempts'), 400
    return launch(run_play, (n, True, False), mode='evaluating', target=0, message='Loading saved model...')


@app.post('/api/stop')
def api_stop():
    stop_event.set()
    return jsonify(ok=True)


@app.get('/api/clips')
def api_clips():
    return jsonify([p.name for p in sorted(CLIPS.glob('*.gif'), key=lambda p: p.stat().st_mtime, reverse=True)[:8]])


@app.get('/clips/<path:name>')
def clip(name):
    return send_from_directory(CLIPS, name)


@app.get('/stream')
def stream():
    def frames():
        while True:
            with lock:
                jpg = state['frame']
            if jpg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            time.sleep(0.04)
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)