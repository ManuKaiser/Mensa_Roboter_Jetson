# Mensa_Roboter_Jetson

This is the code running on the jetson. The code for ROS is here: https://github.com/FabCode288/Mobile_Robot_PI5

This is an adaptation of https://github.com/quancore/social-lstm, the goal is to use it to predict the trajectories of pedestrians in front of the Robot and to later use them in navigation.



## Usage

```
cd /Mensa-Roboter
```

If not already set up:

```
./setup_system.sh
```
---
Start the docker container first:

View all containers:
```
docker ps -a
```

Start the docker container:
```
docker start -ai <container ID>
```

Inside the container:
```
cd ~/Mobile_Robot_PI5/
source install/setup.bash
ros2 run camera_publisher camera_publisher
```
---

When the publisher is ready (outside the container):
```
source social_lstm/.venv/bin/activate

python -m social_lstm.scripts.video_main
```
---
To update the documentation:

```
cd /docs

make html
```

To add other folders to the documentation:

```
cd docs/source

sphinx-apidoc -o api path/to/your/folder
```
