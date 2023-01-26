from common.env_wrappers import create_env
from collections import deque
import time
from rich import print

def evaluate(args, agent):
    iter_start = time.time()
    env = create_env(args, decorr_steps=None, eval_mode=True)
    agent.q_policy.eval()    # Set online network to evaluation mode
    states = env.reset()

    eval_episodes = args.eval_episodes
    eval_rewards = deque(maxlen=eval_episodes) 

    print('Running evaluation for {} episodes'.format(eval_episodes))
    while len(eval_rewards) < eval_episodes:   
        if args.noisy_dqn:     
            agent.disable_noise(agent.q_policy) # disable exploration during eval by setting noise params to zero

        # compute actions to take in all parallel envs, asynchronously start environment step
        actions = agent.act(states, eps=0) # setting eps to 0 to disable exploration
        env.step_async(actions)       
        states, _, _, infos = env.step_wait() # wait till all parallel envs have stepped
        
        for info in infos:
            if 'episode_metrics' in info.keys():    # if an episode has finished, get episode returns
                episode_metrics = info['episode_metrics']
                eval_rewards.append(episode_metrics['return'])

    env.close()
    iter_end = time.time() - iter_start
    print('Finished evaluation in {:.2f} seconds'.format(iter_end))
    
    return eval_rewards
