# Required general libraries
import pandas
import numpy as np
import rospy
import time
import gym
from gym import spaces
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import gazebo_env
from stable_baselines import logger

# Required ROS msgs
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose, Wrench
from sensor_msgs.msg import Imu
from mav_msgs.msg import Actuators
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

# Required ROS services
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState, ApplyBodyWrench, BodyRequest

from gym_foo.gym_foo.envs.QuadTakeOffHoverEnv_v0 import QuadTakeOffHoverEnv_v0

# ttr Engine for the use of TTR reward
# from ttr_engine.ttr_helper import ttr_helper


class QuadRARLEnv_v0(QuadTakeOffHoverEnv_v0):
    def __init__(self, adversarial_bound, **kwargs):
        super(QuadRARLEnv_v0, self).__init__(**kwargs)
        self.apply_body_wrench = rospy.ServiceProxy(
            '/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.clear_body_wrenches = rospy.ServiceProxy(
            '/gazebo/clear_body_wrenches', BodyRequest)

        self.adversarial_bound = adversarial_bound
        self.adv_predict_func = None

    def set_adv_predict_func(self, adv_predict_func):
        self.adv_predict_func = adv_predict_func

    def reset(self, **kwargs):
        obsrv = super(QuadRARLEnv_v0, self).reset(**kwargs)
        self.reset_adv_obs = obsrv
        return obsrv

    def do_adversarial_action(self, action):
        # --- apply body wrench to crazyflie ---
        body_name = 'crazyflie2::crazyflie2/base_link'
        wrench = Wrench()
        wrench.force.x = self.adversarial_bound * action[0]
        wrench.force.y = self.adversarial_bound * action[1]
        wrench.force.z = self.adversarial_bound * action[2]
        self.clear_body_wrenches(body_name)
        self.apply_body_wrench(body_name, "", None, wrench, rospy.Time.from_sec(
            0), rospy.Duration.from_sec(10000.0))

    def step(self, action):
        prev_position = np.array([self.x, self.y, self.pre_obsrv[0]])

        self.do_quadrotor_action(action)

        if self.adv_predict_func:
            self.adv_action = self.adv_predict_func(self.pre_obsrv)
        self.do_adversarial_action(self.adv_action)

        obsrv = self.run_simulation()

        done = False
        suc = False
        self.step_counter += 1

        cost = self.compute_cost(obsrv, prev_position)
        reward = 10-cost

        if self.in_collision(self.pre_obsrv):
            done = True

        if self.step_counter >= self.max_steps:
            done = True

        info = {'is_success': suc,
                'adv_obs': obsrv,
                'adv_reward': cost}
        self.last_step = np.array(np.copy(obsrv)), reward, done, info
        return self.last_step

    # adversarial observation space: [z,vx,vy,vz,roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate]
    @property
    def adv_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

    @property
    def adv_action_space(self):
        return spaces.Box(low=-1, high=1, shape=(3,))


if __name__ == "__main__":
    env = QuadRARLEnv_v0()
    obs = env.reset()
    while True:
        print("obs:", obs)
        if obs[0] > 1:
            obs = env.reset()
        obs, _, _, _ = env.step([0.9, 0.9, 0.9, 0.9])
