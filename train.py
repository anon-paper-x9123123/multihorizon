"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import time, random, datetime
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn

import torch, wandb
import numpy as np
from tqdm import trange
from rich import print

from common import argp
from common.rainbow import Rainbow, MH_Rainbow
from common.env_wrappers import create_env, BASE_FPS_ATARI, BASE_FPS_PROCGEN
from common.utils import LinearSchedule, get_mean_ep_length
from test import evaluate

torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

if __name__ == '__main__':
    exp_start_time = time.time()
    args, wandb_log_config = argp.read_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # set up logging & model checkpoints
    # add corresponding project name and entity name for wandb
    wandb.init(project='project_name', save_code=True, config=dict(**wandb_log_config, log_version=100),
               mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=[args.wandb_tag] if args.wandb_tag else [],
               entity="entity_name", 
               #group="hyperbolic"
               )
    wandb.run.name = args.env_name + "_" + str(datetime.datetime.now())
    save_dir = Path("checkpoints") / wandb.run.name
    save_dir.mkdir(parents=True, exist_ok=True) # NOTE: checkpoints will be overwritten
    args.save_dir = str(save_dir)

    # create decay schedules for dqn's exploration epsilon and per's importance sampling (beta) parameter
    eps_schedule = LinearSchedule(burnin=0, initial_value=args.init_eps, final_value=args.final_eps, decay_time=args.eps_decay_steps)
    per_beta_schedule = LinearSchedule(burnin=0, initial_value=args.prioritized_er_beta, final_value=1.0, decay_time=args.prioritized_er_time)

    # When using many (e.g. 64) environments in parallel, having all of them be correlated (all envs starting from the same state) can be an issue.
    # To avoid this, we estimate the mean episode length for this environment and then take i*(mean ep length/parallel envs count)
    # random steps in the i'th environment. Hence each of the vectorized environment's initial state after this decorrelation should be different.
    print(f'Creating', args.parallel_envs, 'environments..', end='')
    decorr_steps = None
    if args.decorr:
        print('Decorrelating environment instances. This may take up to a few minutes..', end='') 
        decorr_steps = get_mean_ep_length(args) // args.parallel_envs
    env = create_env(args, decorr_steps=decorr_steps, eval_mode=False)  # take decorr_steps random steps in env if env wrapped with DecorrEnvWrapper
    states = env.reset()
    print('Done.')

    if args.multi_horizon:
        print('[blue bold]Running Multi-Horizon Rainbow')
        args.network_arch = 'mh_impala_large:2' # enforce this network_arch
        rainbow = MH_Rainbow(env, args)
    else:
        print('[red bold]Running Single-Horizon Rainbow')
        rainbow = Rainbow(env, args) 

    wandb.watch(rainbow.q_policy)

    print('[blue bold]Running environment =', args.env_name,
          '[blue bold]\nwith action space   =', env.action_space,
          '[blue bold]\nobservation space   =', env.observation_space,
          '[blue bold]\nand config:', sn(**wandb_log_config))

    episode_count = 0
    run_avg = 10    # number of episodes over which a running average of the statistics below is calculated in wandb.log
    returns = deque(maxlen=run_avg) 
    losses = deque(maxlen=run_avg)
    q_values = deque(maxlen=run_avg)
    grad_norms = deque(maxlen=run_avg)
    iter_times = deque(maxlen=run_avg)
    reward_density = 0

    returns_all = []
    q_values_all = []

    # main training loop:
    # we will do a total of args.training_steps/args.parallel_envs iterations
    # in each iteration we perform one interaction step in each of the args.parallel_envs environments,
    # and args.train_count training steps on batches of size args.batch_size
    iterations = trange(0, args.training_steps + 1, args.parallel_envs)
    for steps in iterations:    
        iter_start = time.time()
        eps = eps_schedule(steps)
        per_beta = per_beta_schedule(steps)

        # set the online network to train mode
        rainbow.q_policy.train()

        # reset the noisy-nets noise in the policy
        if args.noisy_dqn:
            rainbow.reset_noise(rainbow.q_policy)

        # compute actions to take in all parallel envs, asynchronously start environment step
        actions = rainbow.act(states, eps)  # when using noisy-nets, eps=0 (decays to 0 in 20k steps in args.py)
        env.step_async(actions)

        # if training has started, perform args.train_count training steps, each on a batch of size args.batch_size
        if rainbow.buffer.burnedin:
            for train_iter in range(args.train_count):
                if args.noisy_dqn and train_iter > 0: rainbow.reset_noise(rainbow.q_policy)
                q, loss, grad_norm = rainbow.train(args.batch_size, beta=per_beta)
                losses.append(loss)
                grad_norms.append(grad_norm)
                q_values.append(q)
                q_values_all.append((steps, q))

        # copy the Q-policy weights over to the Q-target net
        # (see also https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/launcher.py#L155)
        if steps % args.sync_dqn_target_every == 0 and rainbow.buffer.burnedin:
            rainbow.sync_Q_target()

        # block until environments are ready, then collect transitions and add them to the replay buffer
        next_states, rewards, dones, infos = env.step_wait()
        for state, action, reward, done, j in zip(states, actions, rewards, dones, range(args.parallel_envs)):
            reward_density = 0.999 * reward_density + (1 - 0.999) * (reward != 0)
            rainbow.buffer.put(state, action, reward, done, j=j)
        states = next_states

        # if any of the envs finished an episode, log stats to wandb
        for info, j in zip(infos, range(args.parallel_envs)):
            if 'episode_metrics' in info.keys():
                episode_metrics = info['episode_metrics']
                returns.append(episode_metrics['return'])
                returns_all.append((steps, episode_metrics['return']))

                train_log = {'x/steps': steps + j, 
                            'x/episode': episode_count,
                            'ep/return': episode_metrics['return'], 
                            'ep/length': episode_metrics['length'], 
                            'ep/time': episode_metrics['time'],
                            'ep/mean_reward_per_step': episode_metrics['return'] / (episode_metrics['length'] + 1), 
                            'grad_norm': np.mean(grad_norms),
                            'mean_q_value': np.mean(q_values),
                            'mean_loss': np.mean(losses), 
                            'env steps per sec': args.parallel_envs / np.mean(iter_times),  
                            'running_avg_return': np.mean(returns), 
                            'lr': rainbow.opt.param_groups[0]['lr'], 
                            'reward_density': reward_density}
                if args.prioritized_er: train_log['per_beta'] = per_beta
                if eps > 0: train_log['epsilon'] = eps

                # log video recordings if available
                if 'emulator_recording' in info: 
                    train_log['emulator_recording'] = wandb.Video(info['emulator_recording'], fps=(
                    BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI), format="mp4")
                    print('Uploading emulator_recording video to wandb at steps: ', steps) 
                if 'preproc_recording' in info: 
                    train_log['preproc_recording'] = wandb.Video(info['preproc_recording'],
                    fps=(BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI) // args.frame_skip, format="mp4")
                    print('Uploading preprocossed_recording video to wandb at steps: ', steps) 

                wandb.log(train_log, step=steps)
                episode_count += 1
        
        # every eval_every steps (default:50k), run evaluation episodes and log results
        if steps % (args.eval_every-(args.eval_every % args.parallel_envs)) == 0 and steps > 0:
            print('Running evaluation at steps: ', steps)         
            eval_rewards = evaluate(args, rainbow)
            print(f'Mean return over last {run_avg} training episodes= {np.mean(returns)}')
            print(f'Mean return over last {args.eval_episodes} evaluation episodes= {np.mean(eval_rewards)}')
            print(f'Std return over last {run_avg} training episodes= {np.std(returns)}')
            print(f'Std return over last {args.eval_episodes} evaluation episodes= {np.std(eval_rewards)}')
            print(f'Episode rewards in last {run_avg} training episodes = {returns}')
            print(f'Episode rewards in last {args.eval_episodes} evaluation episodes = {eval_rewards}')
            eval_log = {"training mean": np.mean(returns), 
                        "training std": np.std(returns), 
                        "training median": np.median(returns),
                        "evaluation mean": np.mean(eval_rewards), 
                        "evaluation std": np.std(eval_rewards), 
                        "evaluation median": np.median(eval_rewards)}
            wandb.log(eval_log, step=steps)
            torch.cuda.empty_cache()

        # every checkpoint_every steps (default:1M), save a model checkpoint to disk and wandb
        if args.save_model and steps % (args.checkpoint_every-(args.checkpoint_every % args.parallel_envs)) == 0 and steps > 0:
            rainbow.save(steps, args=args, run_name=wandb.run.name, run_id=wandb.run.id, target_metric=np.mean(returns), returns_all=returns_all, q_values_all=q_values_all)
            print(f'Model saved at {steps} steps.')

        iter_times.append(time.time() - iter_start)
        iterations.set_description(f' [{steps:>8} steps, {episode_count:>5} episodes]', refresh=False)

    end_log = {'x/steps': steps + args.parallel_envs, 
               'x/episode': episode_count,
               'x/train_step': (steps + args.parallel_envs) // args.parallel_envs * args.train_count,
               'x/emulator_step': (steps + args.parallel_envs) * args.frame_skip}
    wandb.log(end_log, step=steps)

    env.close()
    wandb.finish()
    print('[green]Experiment finished')
    print('[green]Experiment took {:.2f} minutes'.format((time.time() - exp_start_time)/60))
