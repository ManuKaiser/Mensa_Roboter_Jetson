# Mensa-Roboter

This is an adaptation of https://github.com/quancore/social-lstm, the goal is to use it to predict the trajectories of pedestrians in front of the Robot and to later use them in navigation.

## Usage

```
cd /Mensa-Roboter
```

If not already set up:

```
./setup_system.sh
```

```
source social_lstm/.venv/bin/activate

python -m social_lstm.scripts.video_main
```

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