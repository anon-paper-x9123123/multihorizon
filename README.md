# Revisiting Multi-Horizon Learning: a Generalization Perspective

### Setup
```
conda env create -f requirements.yaml 
```

#### Manual Installation Instructions
The installation may sometime fail for various reasons (e.g. due to cmake or OS version). In such a case, install necessary packages with the following (tested with Python 3.8 and Ubuntu 18.04)

```
sudo apt install zlib1g-dev cmake unrar
conda create -n [env_name] python=3.8
conda activate [env_name]
pip install wandb imageio moviepy tqdm rich procgen gym-retro torch torchsummary stable_baselines3 gym[atari,accept-rom-license]==0.21 opencv-python
```

### How to use

To get started right away, run

```
python train.py --env_name gym:Qbert
python train.py --env_name procgen:bigfish
```

Some other arguments:
```
python train.py --env_name gym:Qbert --use_wandb False --multi_horizon False
python train.py --env_name procgen:caveflyer --use_wandb False --multi_horizon False

python train.py --env_name gym:Qbert --use_wandb False --decorr False --training_steps 100_000
python train.py --env_name procgen:caveflyer --use_wandb False --decorr False --training_steps 100_000
```

### Test cases
The following should be used to test all 3 variants in the repo
1. Single Horizon
```
python train.py --env_name gym:Pong --multi_horizon False
```
2. Multi Horizon (hyperbolic discouting)
```
python train.py --env_name gym:Pong
```
3. Multi Horizon (largest gamma)
```
python train.py --env_name gym:Pong --mh_acting_policy largest_gamma
```


This will train Rainbow on Atari Pong and log all results to "Weights and Biases" and the checkpoints directory.

Please take a look at `common/argp.py` or run `python train.py --help` for more configuration options.
