import argparse
import json

import gym
from models.adversarial_sac import AdversarialSAC
from stable_baselines import PPO2, SAC
from stable_baselines.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

from gym_foo import gym_foo
from run_rarl import run_rarl


def run(model_name, total_timesteps=10000, load_path=None, eval=False):
    env = DummyVecEnv([lambda: gym.make('QuadTakeOffHoverEnv-v0')])

    model_class = {'sac': SAC, 'ppo': PPO2}[model_name]
    if load_path:
        model = model_class.load(
            load_path, env, tensorboard_log='./tensorboard')
    else:
        model = model_class('MlpPolicy', env, verbose=1,
                            tensorboard_log='./tensorboard')

    if eval:
        evaluate_policy(model, env)
    else:
        eval_callback = EvalCallback(
            env, best_model_save_path='./logs', eval_freq=1000, verbose=1)
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save('quad_model_'+model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path", help="Path to load trained model", type=str, default=None)
    parser.add_argument(
        "--model_name", help="RL model to use", type=str, default=None)
    parser.add_argument(
        "--eval", help="Whether to evaluate policy", action='store_true', default=False)
    parser.add_argument(
        "--adv_load_path", help="Path to load trained model", type=str, default=None)
    parser.add_argument(
        "--adv_model_name", help="RL model to use for adversary", type=str, default=None)
    parser.add_argument(
        "--adv_bound", help="Maximum adversarial force to apply", type=float, default=None)
    args = parser.parse_args()
    args = vars(args)

    print("arguments: ", json.dumps(args, indent=4))
    if args['adv_model_name'] is not None:
        run_rarl(**args)
    else:
        run(**args)
