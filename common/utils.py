import zlib
from copy import deepcopy
import torch
from tqdm.auto import trange
import math
from common.env_wrappers import create_env


def prep_observation_for_qnet(tensor, use_amp):
    """ Tranfer the tensor the gpu and reshape it into (batch, frame_stack*channels, y, x) """
    assert len(tensor.shape) == 5, tensor.shape # (batch, frame_stack, y, x, channels)
    tensor = tensor.cuda().permute(0, 1, 4, 2, 3) # (batch, frame_stack, channels, y, x)
    # .cuda() needs to be before this ^ so that the tensor is made contiguous on the gpu
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1]*tensor.shape[2], *tensor.shape[3:]))

    return tensor.to(dtype=(torch.float16 if use_amp else torch.float32)) / 255

class LinearSchedule:
    """Set up a linear hyperparameter schedule (e.g. for dqn's epsilon parameter)"""

    def __init__(self, burnin: int, initial_value: float, final_value: float, decay_time: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_time = decay_time
        self.burnin = burnin

    def __call__(self, frame: int) -> float:
        if frame < self.burnin:
            return self.initial_value
        else:
            frame = frame - self.burnin

        slope = (self.final_value - self.initial_value) / self.decay_time
        if self.final_value < self.initial_value:
            return max(slope * frame + self.initial_value, self.final_value)
        else:
            return min(slope * frame + self.initial_value, self.final_value)

def get_mean_ep_length(args):
    """Run a few iterations of the environment and estimate the mean episode length"""
    dc_args = deepcopy(args)
    dc_args.parallel_envs = 12
    dc_args.subproc_vecenv = True
    dc_env = create_env(dc_args)
    dc_env.reset()

    # Decorrelate envs
    ep_lengths = []
    for frame in trange(args.time_limit//4+100):
        _, _, _, infos = dc_env.step([dc_env.action_space.sample() for x in range(dc_args.parallel_envs)])
        for info, j in zip(infos, range(dc_args.parallel_envs)):
            if 'episode_metrics' in info.keys(): ep_lengths.append(info['episode_metrics']['length'])
    dc_env.close()
    mean_length = sum(ep_lengths)/len(ep_lengths)
    return mean_length

def env_seeding(user_seed, env_name):
    return user_seed + zlib.adler32(bytes(env_name, encoding='utf-8')) % 10000


# from https://github.com/google-research/google-research/tree/master/hyperbolic_discount
def compute_eval_gamma_interval(gamma_max, hyp_exponent, number_of_gammas):
  r"""This method computes the eval_gammas intervals.
  These are the non-exponentiated gammas.  We produce this eval_gammas
  sequence in a way that when the largest eval_gamma is exponentiated, the
  largest gamma being learned via Bellman updates is approximately equal
  to self.gamma_max.
  We can see that if we use a power-method for choosing the gamma interval,
  the base b must satisfy the relation,
        (1. - b ^ N) ^ k = \gamma_max
  where N is the number of gammas, k is the hyperbolic exponent.  The base b
  to use is then solved as
        b = exp[ ln(1. - \gamma_max ^ (1 \ k)) / N]
  Args:
    gamma_max:  The maximum gamma to use in Bellman equations.
    hyp_exponent:  The k-coefficient for hyperbolic discounting.
    number_of_gammas:  Number of gammas to simultaneously model.
  Returns:
    eval_gammas:  List of gammas selected for the x-axis of the Riemann
      rectangles.
  """
  if number_of_gammas > 1:
    b = math.exp(
        math.log(1. - math.pow(gamma_max, 1. / hyp_exponent)) /
        number_of_gammas)
    eval_gammas = [1. - math.pow(b, i) for i in range(1, number_of_gammas+1)] # modifying to ignore gamma=0 value

  # If using only a single head, overwrite and set gammas to be the only
  # gamma being used.
  else:
    eval_gammas = [gamma_max]

  return eval_gammas


def integrate_q_values(q_values, integral_estimate, eval_gammas,
                       number_of_gammas, gammas):
  """Estimate the integral for the hyperbolic discount q-values.
  This does a Riemann sum to approximate the integral.  This builds a lower or
  an upper estimate of the integral via a set of rectangles.
  Args:
    q_values:  List of q-values.
    integral_estimate: Type of Riemann integral estimate. One of 'lower',
      'upper'.
    eval_gammas:  List of gammas selected for the x-axis of the Riemann
      rectangles.
    number_of_gammas:  Number of gammas to simultaneously model.
    gammas:  List of exponentiated gammas that are actually used for Bellman
      updates.
  Returns:
    integral:  Scalar estimate of the integral.
  """
  integral = 0.

  if integral_estimate == 'lower':
    gamma_plus_one = eval_gammas + [1.]
    # The weights are derived by which interval of gammas we wish to consider
    # i.e. the self.eval_gammas.  These are the non-exponentiated gammas.
    #
    # Example
    # If we're evaluating at \gamma = 0.5 and our k = 0.1
    # then we still evaluate the integral at \gamma = 0.5, but we are now
    # estimating the Q-values associated with \gamma ^ k = 0.5 ^ 0.1 = 0.933
    weights = [gamma_plus_one[i + 1] - gamma_plus_one[i] \
              for i in range(number_of_gammas)]

    # Estimate the integral, which approximates the hyperbolic discounted
    # Q-values.
    for w, q_val in zip(weights, q_values):
      integral += w * q_val

  elif integral_estimate == 'upper':
    gamma = eval_gammas
    weights = [gamma[i + 1] - gamma[i] \
               for i in range(number_of_gammas - 1)]
    weights = [0.] + weights

    for w, q_val in zip(weights, q_values):
      integral += w * q_val
  else:
    raise NotImplementedError
  return integral