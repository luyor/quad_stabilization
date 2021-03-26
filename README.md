This branch is called "dev_from_202004", which means it is a new start from April 2020. 

(1) Complie ROS package CrazyS
(2) Customize launch files in ./launchs, or world files in ./worlds


## Run with VSCode remote container
1. install VcXsrv in the host machine, and run X-Launch server
2. install vscode remote container plugin, `ctl-shift-p` then input `Reopen in container`
3. setup environment variable `DISPLAY=172.26.96.1:0`, replace `172.26.96.1` with your X-Launch server ip, to enable gazebo UI
4. `python run.py --model_name sac` to start training. model_name can be `sac` or `ppo`.
5. `python run.py --model_name sac --load_path logs/best_model` to continue training.
6. `python run.py --model_name sac --load_path logs/best_model --eval` to evaluate model.