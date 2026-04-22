```
cello_rl/
│
├── config/
│   └── config.yaml                 # all hyperparameters & settings
│
├── hardware/
│   ├── robot_interface.py          # RTDE communication — sends scalar z-force,
│   │                               # reads TCP position & velocity
│   └── audio_interface.py          # record .wav per interval, return path
│
├── models/
│   ├── gp_gate.py                  # safety gate — unchanged
│   ├── surrogate.py                # (scalar force, tcp) → predicted score
│   └── policy.py                   # act(state) returns scalar z-force
│
├── rl/
│   ├── sac.py                      # SAC update rule
│   ├── ppo.py                      # PPO update rule
│   └── replay_buffer.py            # stores one tuple per interval
│
├── reward/
│   └── reward.py                   # scores each interval's audio window
|   └── classifier.py               # sound classification model
│
├── data/
│   ├── warmup_collector.py         # collects 200 scripted samples
│   └── dataset.py                  # manages growing training dataset
│
├── training/
│   ├── training_loop.py            # two threads: RTDE & RL, shared force queue
│   └── latency_estimator.py        
│
├── evaluation/
│   └── evaluate.py                 # periodic real robot evaluation
│
└── main.py                         # entry point, spins up both threads
```