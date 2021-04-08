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
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Imu
from mav_msgs.msg import Actuators
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

# Required ROS services
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState


# ttr Engine for the use of TTR reward
# from ttr_engine.ttr_helper import ttr_helper


class QuadTakeOffHoverEnv_v0(gazebo_env.GazeboEnv):
    def __init__(self, **kwargs):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(
            self, "crazyflie2_without_controller.launch")

        self.is_evaluate = kwargs['is_evaluate']

        # --- Max episode steps ---
        self.max_steps = 50
        self.step_counter = 0

        # --- Take off and Hover identifier --
        self.isTakeoff = True
        self.isHover = False

        self.min_motor_speed = 0
        # --- Specification of maximum motor speed, from crazyflie manual ---
        self.max_motor_speed = 2618

        # --- Specification of target ---
        self.target_height = 1

        # rospy.init_node("QuadTakeOffHover", anonymous=True, log_level=rospy.INFO)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_model_state = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)

        self.enable_motor = rospy.Publisher(
            '/crazyflie2/command/motor_speed', Actuators, queue_size=1)

    def reset(self, reset_args=None):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
            # initialize pose position
            pose = Pose()
            pose.position.x = 0
            pose.position.y = 0
            pose.position.z = np.random.uniform(0.9, 1.1)

            # initialize pose orientation
            # Note: gazebo rotation order: roll, pitch, yaw
            # the angle w.r.t x-axis: roll in gazebo, [-np.pi/2, np.pi/2]
            # the angle w.r.t y-axis: pitch in gazebo, [-np.pi/2, np.pi/2]
            # the angle w.r.t z-axis: yaw in gazebo, [0, 2*np.pi]

            # roll = np.random.uniform(-np.pi/2, np.pi/2)
            # pitch = np.random.uniform(-np.pi/2, np.pi/2)
            # yaw = np.random.uniform(0, 2*np.pi)
            roll = np.random.uniform(-np.pi/2, np.pi/2)
            pitch = np.random.uniform(-np.pi/2, np.pi/2)
            yaw = np.random.uniform(-np.pi/2, np.pi/2)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_euler(
                roll, pitch, yaw)

            # initialize twist
            # the reset value should be referring to some empirical value
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = np.random.uniform(
                -1, 1, 3)
            twist.angular.x, twist.angular.y, twist.angular.z = np.random.uniform(
                -1, 1, 3)

            reset_state = ModelState()
            reset_state.model_name = "crazyflie2"
            reset_state.pose = pose
            reset_state.twist = twist
            self.set_model_state(reset_state)
        except rospy.ServiceException as e:
            print("# /gazebo/reset_simulation call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # read observation
        Pose_data = None
        Imu_data = None
        Odom_data = None
        while Pose_data is None or Imu_data is None or Odom_data is None:
            Pose_data = rospy.wait_for_message(
                "/crazyflie2/pose_with_covariance", PoseWithCovarianceStamped, timeout=5)
            # time.sleep(0.01)
            Imu_data = rospy.wait_for_message(
                '/crazyflie2/ground_truth/imu', Imu, timeout=5)
            # time.sleep(0.01)
            # Odom_data = rospy.wait_for_message('/crazyflie2/ground_truth/odometry', Odometry, timeout=5)
            Odom_data = self.get_model_state(model_name="crazyflie2")
            # time.sleep(0.01)

        # Pause simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed!")

        obsrv = self.get_obsrv(Pose_data, Imu_data, Odom_data)
        self.pre_obsrv = obsrv

        # --- reset these ---
        self.isTakeoff = True
        self.isHover = False
        self.step_counter = 0

        return obsrv

    def get_obsrv(self, Pose_data, Imu_data, Odom_data):
        # we don't include any state variables w.r.t 2D positions (x,y)
        # valid state variables: [z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w]

        # Height to ground
        self.x = Pose_data.pose.pose.position.x
        self.y = Pose_data.pose.pose.position.y
        z = Pose_data.pose.pose.position.z

        # linear velocities
        # vx = Odom_data.twist.twist.linear.x
        # vy = Odom_data.twist.twist.linear.y
        # vz = Odom_data.twist.twist.linear.z
        vx = Odom_data.twist.linear.x
        vy = Odom_data.twist.linear.y
        vz = Odom_data.twist.linear.z

        # roll, pitch, yaw
        roll, pitch, yaw = euler_from_quaternion(
            [Imu_data.orientation.x, Imu_data.orientation.y, Imu_data.orientation.z, Imu_data.orientation.w])

        # angular velocities of roll, pitch, yaw
        roll_w, pitch_w, yaw_w = Imu_data.angular_velocity.x, Imu_data.angular_velocity.y, Imu_data.angular_velocity.z

        # print(np.array([z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w]))

        return np.array([z, vx, vy, vz, roll, pitch, yaw, roll_w, pitch_w, yaw_w])

    def compute_reward(self, state, prev_position):
        target = np.array([self.target_height, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        diff = state - target
        pos = np.array([self.x, self.y, state[0]])
        pos_diff = pos - prev_position
        pos_cost = np.linalg.norm(pos_diff) * 20
        twist_cost = np.linalg.norm(diff[1:4], 2) * 0.05
        angle_cost = np.linalg.norm(diff[4:7], 2) * 0.5
        angular_vel_cost = np.linalg.norm(diff[7:10], 2) * 0.1
        costs = [pos_cost, twist_cost, angle_cost, angular_vel_cost]

        # if self.is_evaluate:
        #     headers = ['position', 'twist', 'angle', 'angular vel']
        #     print(pandas.DataFrame([costs], ['cost'], headers))
        #     print('total cost:', sum(costs))

        return 10-sum(costs)

    def step(self, action):
        prev_position = np.array([self.x, self.y, self.pre_obsrv[0]])
        # action is 4-dims representing drone's four motor speeds
        action = np.asarray(action)

        # --- check if the output of policy network is nan ---
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # --- transform action from network output into environment limit, i.e. [-1,1] to [self.min_motor_speed, self.max_motor_speed]---
        real_action = spaces.Box(
            low=self.min_motor_speed, high=self.max_motor_speed, shape=(4,))
        env_action = self.min_motor_speed + 0.5 * \
            (action+1)*(self.max_motor_speed-self.min_motor_speed)
        clipped_env_ac = np.clip(
            env_action.copy(), self.min_motor_speed, self.max_motor_speed)
        # print("real action:", clipped_env_ac)

        # --- apply motor speed to crazyflie ---
        cmd_msg = Actuators()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.angular_velocities = clipped_env_ac
        self.enable_motor.publish(cmd_msg)

        # --- run simulator to collect data ---
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        Pose_data = None
        Imu_data = None
        Odom_data = None
        while Pose_data is None or Imu_data is None or Odom_data is None:
            Pose_data = rospy.wait_for_message(
                "/crazyflie2/pose_with_covariance", PoseWithCovarianceStamped)
            # time.sleep(0.01)
            Imu_data = rospy.wait_for_message(
                '/crazyflie2/ground_truth/imu', Imu)
            # time.sleep(0.01)
            # Odom_data = rospy.wait_for_message('/crazyflie2/ground_truth/odometry', Odometry)
            Odom_data = self.get_model_state(model_name='crazyflie2')
            # time.sleep(0.01)

        # --- pause simulator to process data ---
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # --- deal with obsrv and reward ---
        obsrv = self.get_obsrv(Pose_data, Imu_data, Odom_data)
        self.pre_obsrv = obsrv

        reward = 0
        done = False
        suc = False
        self.step_counter += 1

        reward = self.compute_reward(obsrv, prev_position)

        if self.in_collision(self.pre_obsrv):
            done = True

        if self.step_counter >= self.max_steps:
            done = True

        return np.array(np.copy(obsrv)), reward, done, {'suc': suc}

    def in_collision(self, obsrv):
        if self.x >= 2 or self.x <= -2 \
                or self.y >= 2 or self.y <= -2 \
                or obsrv[0] <= 0.5 or obsrv[0] >= 1.5:
            # print("in collision, x and y out of range!")
            return True
        else:
            return False

    # observation space: [z,vx,vy,vz,roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate]
    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(4,))


if __name__ == "__main__":
    env = QuadTakeOffHoverEnv_v0()
    obs = env.reset()
    while True:
        print("obs:", obs)
        if obs[0] > 1:
            obs = env.reset()
        obs, _, _, _ = env.step([0.9, 0.9, 0.9, 0.9])
