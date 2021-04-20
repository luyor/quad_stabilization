import argparse
import json
import os

import gym
from models.adversarial_sac import AdversarialSAC
from stable_baselines import PPO2, SAC
from stable_baselines.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy


class RARLCallback(BaseCallback):
    def __init__(self, pro_model, adv_model, total_timesteps, save_freq=1000, verbose=0):
        super(RARLCallback, self).__init__(verbose)
        self.pro_model = pro_model
        self.adv_model = adv_model
        self.save_freq = save_freq

        self.adv_model_step = adv_model.learn(total_timesteps=total_timesteps)

    def on_training_start(self, locals, globals):
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


def run_rarl(model_name, adv_model_name, total_timesteps=30000, load_path=None, adv_load_path=None, eval=False, adv_bound=0.1):
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
        evaluate_policy(model, env)
    else:
        rarl_callback = RARLCallback(
            pro_model, adv_model, total_timesteps, save_freq=1000)
        callbacks = CallbackList([rarl_callback])
        pro_model.learn(total_timesteps=total_timesteps, callback=callbacks)
        pro_model.save('protagonist_model_'+model_name)
        adv_model.save('adversary_model_'+adv_model_name)
