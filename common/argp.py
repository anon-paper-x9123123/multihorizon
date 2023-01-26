"""
This file handles parsing and validation of the cli arguments to the train_rainbow.py file.
If left unspecified, some argument defaults are set dynamically here.
"""

import argparse
import distutils
import random
import socket
from copy import deepcopy

from common.utils import env_seeding


def read_args():
    parse_bool = lambda b: bool(distutils.util.strtobool(b))
    parser = argparse.ArgumentParser(description='Training framework for Rainbow DQN\n'
                                                 '  - supports environments from the ALE (via gym), gym-retro and procgen\n'
                                                 '  - individial components of Rainbow can be adjusted with cli args (below)\n'
                                                 '  - uses vectorized environments and batches environment steps for best performance\n'
                                                 '  - uses the large IMPALA-CNN (with 2x channels by default)',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # training settings
    parser.add_argument('--training_steps', type=int, default=25_000_000, help='train for n environment interactions ("steps" in the code)')
    parser.add_argument('--record_every', type=int, default=60*50, help='wait at least x seconds between episode recordings (default is to use environment specific presets)')
    parser.add_argument('--seed', type=int, default=0, help='seed for pytorch, numpy, environments, random')
    parser.add_argument('--use_wandb', type=parse_bool, default=True, help='whether use "weights & biases" for tracking metrics, video recordings and model checkpoints')
    parser.add_argument('--use_amp', type=parse_bool, default=True, help='whether to enable automatic mixed precision for the forward passes')
    parser.add_argument('--der', type=parse_bool, default=False, help='enable data-efficient-rainbow profile (overrides some of the settings below)')
    parser.add_argument('--decorr', type=parse_bool, default=True, help='try to decorrelate state/progress in parallel envs')

    # evaluation settings
    parser.add_argument('--eval_episodes', type=int, default=10, help='number of episodes to run evaluation for')
    parser.add_argument('--eval_parallel_envs', type=int, default=8, help='number of vectorized envs to be used during evaluation')
    parser.add_argument('--eval_every', type=int, default=250_000, help='run evaluation every X training steps')
    parser.add_argument('--save_model', type=parse_bool, default=False, help='whether to save a model checkpoint or not')
    parser.add_argument('--checkpoint_every', type=int, default=1_000_000, help='save a model checkpoint every X training steps')

    # multi_horizon settings
    parser.add_argument('--multi_horizon', type=parse_bool, default=True, help='whether to use Multi Horizon Rainbow class')
    parser.add_argument('--mh_num_gammas', type=int, default=5, help='number of gammas to learn over')
    parser.add_argument('--mh_gamma_max', type=float, default=0.99, help='number of gammas to learn over')
    parser.add_argument('--mh_hyp_exp', type=float, default=0.1, help='hyperbolic coefficient k in the equation 1. / (1. + k * t)')
    parser.add_argument('--mh_integral_estimate', type=str, default="lower", help='reimann sum integral estimate, either lower or upper')
    parser.add_argument('--mh_acting_policy', type=str, default="hyperbolic", help='acting policy between hyperbolic or largest_gamma')
    parser.add_argument('--mh_alternate_priorities', type=parse_bool, default=False, help='when true, PER priorities correspond only to largest_gamma TD loss, otherwise averaged across all q_heads')
    
    # environment settings
    parser.add_argument('--env_name', type=str, default='gym:Qbert',
                        help='the gym/procgen/retro environment name, should be either gym:[name], retro:[name] or procgen:[name]\n'
                             'some gym envs:   MsPacman, Phoenix, Breakout, Qbert, Amidar, SpaceInvaders, Assault\n'
                             'some retro envs: SuperMarioWorld-Snes, MortalKombat3-Genesis, SpaceMegaforce-Snes, SmashTV-Nes, AirBuster-Genesis, NewZealandStory-Genesis, Paperboy-Nes\n'
                             'progcen envs:    bigfish, bossfight, caveflyer, chaser, climber, coinrun, dodgeball, fruitbot, heist, jumper, leaper, maze, miner, ninja, plunder, starpilot')
    parser.add_argument('--procgen_distribution_mode', type=str, default='easy',
                        help='what variant of the procgen levels to use, the options are "easy", "hard", "extreme", "memory", "exploration". All'
                             ' games support "easy" and "hard", while other options are game-specific. The default is "hard". Switching to "easy'
                             '" will reduce the number of timesteps required to solve each game and is useful for testing or when working with l'
                             'imited compute resources.')
    parser.add_argument('--retro_state', type=str, default='default', help='initial gym-retro state name or "default" or "randomized" (to randomize on episode reset)')
    parser.add_argument('--time_limit', type=int, default=27_000, help='environment time limit for gym & retro (in env steps, equal to 108k frames with default frame_skip=4)')
    parser.add_argument('--eid', type=int, default=None, help='')
    parser.add_argument('--wandb_tag', type=str, default=None, help='the tag that can be used in wandb to identify the run')

    # env preprocessing settings
    parser.add_argument('--frame_skip', type=int, default=None, help='use only every nth env frame (default is to use environment specific presets)')
    parser.add_argument('--frame_stack', type=int, default=None, help='stack n frames (default is to use environment specific presets)')
    parser.add_argument('--grayscale', type=parse_bool, default=None, help='convert environment to grayscale (default is to use environment specific presets)')
    parser.add_argument('--resolution', type=int, default=None, help='environment resolution (default is to use environment specific presets)')

    # dqn settings
    parser.add_argument('--buffer_size', type=int, default=int(2 ** 20), help='default~1M, capacity of experience replay buffer (must be a power of two)')
    parser.add_argument('--burnin', type=int, default=20_000, help='learning starts. how many transitions should be in the buffer before start of training')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor')
    parser.add_argument('--sync_dqn_target_every', type=int, default=8000, help='target-update. sync Q target net every n steps (32k frames)')

    parser.add_argument('--batch_size', type=int, default=256, help='sample size when sampling from the replay buffer')
    parser.add_argument('--parallel_envs', type=int, default=125, help='number of envs in the vectorized env')
    parser.add_argument('--train_count', type=int, default=1, help='how often to train on a batch_size batch for every step (of the vectorized env)')
    parser.add_argument('--subproc_vecenv', type=parse_bool, default=False, help='whether to run each environment in it\'s own subprocess (always enabled for gym-retro)')

    # rainbow settings
    parser.add_argument('--network_arch', type=str, default='impala_large:2', 
                        help='which model architecture to use for the q-network; one of "nature", "dueling", "impala_small", "impala_large:c" (c is the number of channels in impala large)')
    parser.add_argument('--spectral_norm', type=str, default='all', help='where to use spectral norm in IMPALA-large residual blocks ("none", "last", "all")')
    parser.add_argument('--double_dqn', type=parse_bool, default=True, help='whether to use the double-dqn TD-target')
    parser.add_argument('--prioritized_er', type=parse_bool, default=False, help='whether to use prioritized experience replay')
    parser.add_argument('--prioritized_er_alpha', type=float, default=0.5, help='priority exponent alpha for PER (omega in Rainbow paper, determines how much prioritization is used. α = 0 corresponding to the uniform case') 
    parser.add_argument('--prioritized_er_eps', type=float, default=1e-6, help='minimal priority, prevents zero probabilities. guarantees every transition can be sampled') 
    parser.add_argument('--prioritized_er_beta', type=float, default=0.4, help='initial value of importance sampling exponent Beta for PER, anneals to 1.0. determines the amount of importance-sampling correction. b = 1 fully compensate for the non-uniform probabilities')
    # Note: prioritized_er_beta is set to 0.4 in rainbow paper, Dopamine sets it to 0.5 with no linear increase, Dopamine also does not use alpha)
    parser.add_argument('--prioritized_er_time', type=int, default=None, help='time period over which to increase the IS exponent (+inf for dopamine; default is value of training_steps)')
    parser.add_argument('--n_step', type=int, default=3, help='the n in n-step bootstrapping')
    parser.add_argument('--init_eps', type=float, default=1.0, help='initial dqn exploration epsilon (when not using noisy-nets)')
    parser.add_argument('--final_eps', type=float, default=0.01, help='final dqn exploration epsilon (when not using noisy-nets)')
    parser.add_argument('--eps_decay_steps', type=int, default=62_500, help='exploration epsilon decay frames (when not using noisy-nets)')
    # Note: eps_greedy: Rainbow paper: 250k frames (hence 62.5k steps here), 4M frames for DQN;  250k steps in Dopamine Rainbow, 1M steps in Dopamine DQN
    parser.add_argument('--noisy_dqn', type=parse_bool, default=True, help='whether to use noisy nets dqn')
    parser.add_argument('--noisy_sigma0', type=float, default=0.5, help='sigma_0 parameter for noisy nets dqn')

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate for adam (0.0000625 for rainbow paper/dopamine, 0.00025 for DQN/procgen paper)')
    parser.add_argument('--lr_decay_steps', type=int, default=None, help='learning rate is decayed every n game_steps (disabled by default)')
    parser.add_argument('--lr_decay_factor', type=float, default=None, help='factor by which lr is multiplied (disabled by default)')
    parser.add_argument('--adam_eps', type=float, default=None, help='epsilon for adam (0.00015 for rainbow paper/dopamine, 0.0003125 for DQN/procgen paper); default is to use 0.005/batch_size')
    parser.add_argument('--max_grad_norm', type=float, default=10, help='gradient will be clipped to ensure its l2-norm is less than this')
    parser.add_argument('--loss_fn', type=str, default='huber', help='loss function ("mse" or "huber")')

    # gym-retro specific settings
    parser.add_argument('--retro_stickyprob', type=float, default=0.25, help='sticky-action probability in the StochasticFrameSkip wrapper')
    parser.add_argument('--retro_action_patch', type=str, default='single_buttons',
                        help='defines how to generate the action space from controller buttons, should be either "discrete" '
                             '(each combination of buttons is an action) or "single_buttons" (each button is an action)\n')

    # procgen specific settings (from https://github.com/openai/procgen)
    parser.add_argument('--procgen_num_levels', type=int, default=0, help='the number of unique levels that can be generated. Set to 0 to use unlimited levels. (this does not work correctly when parallel_envs > 1)') # FIXME: this does not work correctly when parallel_envs > 1
    parser.add_argument('--procgen_start_level', type=int, default=0, help="the lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.")
    parser.add_argument('--procgen_paint_vel_info', type=parse_bool, default=False, help='paint player velocity info in the top left corner. Only supported by certain games.')
    parser.add_argument('--procgen_center_agent', type=parse_bool, default=True, help='determines whether observations are centered on the agent or display the full level. Override at your own risk.')
    parser.add_argument('--procgen_use_sequential_levels', type=parse_bool, default=False, help='when you reach the end of a level, the episode is ended and a new level is selected. If use_sequential_levels is set to True, reaching the end of a level does not end the episode, and the seed for the new level is derived from the current level seed. If you combine this with start_level=<some seed> and num_levels=1, you can have a single linear series of levels similar to a gym-retro or ALE game.')
    parser.add_argument('--procgen_use_generated_assets', type=parse_bool, default=False, help='use randomly generated assets in place of human designed assets.')
    parser.add_argument('--procgen_use_backgrounds', type=parse_bool, default=True, help='normally games use human designed backgrounds, if this flag is set to False, games will use pure black backgrounds.')
    parser.add_argument('--procgen_restrict_themes', type=parse_bool, default=False, help='some games select assets from multiple themes, if this flag is set to True, those games will only use a single theme.')
    parser.add_argument('--procgen_use_monochrome_assets', type=parse_bool, default=False, help='if set to True, games will use monochromatic rectangles instead of human designed assets. best used with restrict_themes=True.')
    args = parser.parse_args()

    # some initial checks to ensure all arguments are valid
    assert (args.sync_dqn_target_every % args.parallel_envs) == 0 # otherwise target may not be synced since the main loop iterates in steps of parallel_envs
    assert args.loss_fn in ('mse', 'huber')
    assert (args.lr_decay_steps is None) == (args.lr_decay_factor is None)
    assert args.burnin > args.batch_size
    assert args.spectral_norm == "none" or args.spectral_norm == "last" or args.spectral_norm == "all"
    assert args.mh_integral_estimate == "upper" or args.mh_integral_estimate == "lower"
    assert args.mh_acting_policy == "hyperbolic" or args.mh_acting_policy == "largest_gamma"

    if args.eid is not None:
        envs = ['Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']
        args.env_name = 'gym:' + envs[args.eid]

    args.user_seed = args.seed
    args.seed = env_seeding(args.user_seed, args.env_name)

    # apply default values if user did not specify custom settings
    if args.adam_eps is None: args.adam_eps = 0.005/args.batch_size
    if args.prioritized_er_time is None: args.prioritized_er_time = args.training_steps

    if args.resolution is not None:
        args.resolution = (args.resolution, args.resolution)

    if args.env_name.startswith('gym:'):
        if args.frame_skip is None: args.frame_skip = 4
        if args.frame_stack is None: args.frame_stack = 4
        if args.resolution is None: args.resolution = (84, 84)
        if args.grayscale is None: args.grayscale = True
    elif args.env_name.startswith('retro:'):
        if args.frame_skip is None: args.frame_skip = 4
        if args.frame_stack is None: args.frame_stack = 4
        if args.resolution is None: args.resolution = (80, 80)
        if args.grayscale is None: args.grayscale = False
        if not args.subproc_vecenv: print('[WARNING] subproc_vecenv was forcibly enabled since retro envs need to run in subprocesses anyway!')
        args.subproc_vecenv = True
    elif args.env_name.startswith('procgen:'):
        if args.frame_skip is None: args.frame_skip = 1
        if args.frame_stack is None: args.frame_stack = 4  # although Procgen paper does not use frame stacking, it did show some performance improvement (Appendix H)
        if args.resolution is None: args.resolution = args.resolution = (64, 64)
        if args.grayscale is None: args.grayscale = False
        args.time_limit = None
        args.decorr = None      # no decorrelation necessary in Procgen, unlike ALE

    # hyperparameters for DER are adapted from https://github.com/Kaixhin/Rainbow
    if args.der:
        args.parallel_envs = 1
        args.batch_size = 32
        args.train_count = 1
        args.burnin = 1600
        args.n_step = 20
        args.sync_dqn_target_every = 2000
        args.training_steps = 100_000
        args.buffer_size = 2 ** 17  # 131000 ~100k (DER Paper memory=unbounded, manually setting the memory capacity to be the same as the maximum number of timesteps)
        args.lr = 0.00001
        args.adam_eps = 0.0003125
        args.noisy_sigma0 = 0.1

    # turn off e-greedy exploration if noisy_dqn is enabled, 20k here can be thought of as warm_up steps
    if args.noisy_dqn:
        args.init_eps = 0.002
        args.final_eps = 0.0
        args.eps_decay_steps = 20000

    # clean up the parameters that get logged to wandb
    args.instance = socket.gethostname()
    wandb_log_config = deepcopy(vars(args))
    wandb_log_config['env_type'] = args.env_name[:args.env_name.find(':')]
    del wandb_log_config['record_every']
    del wandb_log_config['use_wandb']
    if not args.env_name.startswith('retro:'):
        for k in list(wandb_log_config.keys()):
            if k.startswith('retro'):
                del wandb_log_config[k]
    if not args.env_name.startswith('procgen:'):
        for k in list(wandb_log_config.keys()):
            if k.startswith('procgen'):
                del wandb_log_config[k]
    if args.multi_horizon is False:
        for k in list(wandb_log_config.keys()):
            if k.startswith('mh'):
                del wandb_log_config[k]
    del wandb_log_config['wandb_tag']

    return args, wandb_log_config
