import argparse
import json
import os

import numpy as np
import tensorflow as tf
import gym
from models.adversarial_sac import AdversarialSAC
from stable_baselines import PPO2, SAC
from stable_baselines.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy


class RARLCallback(BaseCallback):
    def __init__(self,
                 pro_model,
                 adv_model,
                 total_timesteps,
                 n_eval_episodes=1,
                 save_freq=1000,
                 eval_freq=1000,
                 verbose=0):
        super(RARLCallback, self).__init__(verbose)
        self.pro_model = pro_model
        self.adv_model = adv_model
        self.n_eval_episodes = n_eval_episodes
        self.save_freq = save_freq
        self.eval_freq = eval_freq

        self.adv_model_step = adv_model.learn(total_timesteps=total_timesteps)
        self.eval_env = gym.make('QuadTakeOffHoverEnv-v0', launch_gazebo=False)

    def _on_training_start(self):
        next(self.adv_model_step)

    def _on_step(self):
        try:
            next(self.adv_model_step)
        except StopIteration as e:
            pass

        # save checkpoint
        if self.n_calls % self.save_freq == 0:
            pro_path = os.path.join('./logs/', '{}_{}_steps'.format(
                'protagonist', self.pro_model.num_timesteps))
            self.pro_model.save(pro_path)

            adv_path = os.path.join('./logs/', '{}_{}_steps'.format(
                'adversary', self.adv_model.num_timesteps))
            self.adv_model.save(adv_path)

    def _on_rollout_start(self):
        # evaluate on environment without distubance, and log on tensorboard
        if self.num_timesteps % self.eval_freq != 0:
            return

        episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                           n_eval_episodes=self.n_eval_episodes,
                                                           render=False,
                                                           deterministic=True,
                                                           return_episode_rewards=True)
        mean_reward = np.mean(episode_rewards)
        summary = tf.Summary(value=[tf.Summary.Value(
            tag='eval_episode_reward', simple_value=mean_reward)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        print("evalate episode reward:", mean_reward)


def run_rarl(model_name, adv_model_name=None, total_timesteps=30000, load_path=None,
             adv_load_path=None, eval=False, adv_bound=0.03, adv_rand=False):
    env = gym.make('QuadRARLEnv-v0', adversarial_bound=adv_bound)

    model_class = {'sac': SAC, 'ppo': PPO2}[model_name]
    if load_path:
        pro_model = model_class.load(
            load_path, env, tensorboard_log='./tensorboard')
    else:
        pro_model = model_class('MlpPolicy', env, verbose=1,
                                tensorboard_log='./tensorboard')

    adv_model_class = {'sac': AdversarialSAC}[model_name]
    if adv_load_path:
        adv_model = adv_model_class.load(
            adv_load_path, env, tensorboard_log='./tensorboard')
    else:
        adv_model = adv_model_class('MlpPolicy', env, verbose=1,
                                    tensorboard_log='./tensorboard')

    if eval:
        if adv_rand:
            env.set_adv_predict_func(lambda x: np.random.rand(3)*2-1)
        else:
            env.set_adv_predict_func(lambda x: adv_model.predict(x)[0])

        mean_reward, std_reward = evaluate_policy(
            pro_model, env, n_eval_episodes=100)
        print("mean reward:", mean_reward)
        print("std reward:", std_reward)
    else:
        rarl_callback = RARLCallback(
            pro_model, adv_model, total_timesteps, save_freq=5000)
        pro_model.learn(total_timesteps=total_timesteps,
                        callback=rarl_callback)
        pro_model.save('protagonist_model_'+model_name)
        adv_model.save('adversary_model_'+adv_model_name)
