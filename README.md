# CS 188 Final Project: Square Assmebly using Imitation Learning

This repository contains the final project for CS 188: Introduction to Robotics. The goal of the project is use imitation learning to train a robot in completeing Robo-Suite's "Square Assmebl" task.

## Background

The implementation uses a DMP-policy and nearest neighbor algorithm to generate a trajectory for the robot's end effector to follow using multiple demonstrations. It handles orientation by adjusting the robot's end effector before picking up and dropping off the square nut.

## Project Structure
```
- dmp.py                # DMP and canonical system implementation
- dmp_policy.py         # DMP-policy class 
- pid.py                # 3-DoF PID controller 
- load_data.py          # Loads demonstration data from .npz
- nearest_neighbor.py   # Nearest neighbor selection based on EEF position
- graph.py              # Evaluation + success rate plotting for multiple demo sizes
- test_ca3.py           # Test script 
- demos.npz             # Demo file containing demonstration data
- README.md             
```

## Installation

1. Install [Robosuite] 

2. Clone the repository
```
git clone https://github.com/justinktruong/cs-188-final-project.git
```

3. Run test_ca3.py to run a single test
```
python test_ca3.py
```

4. Run graph.py to evaluate different sample sizes
```
python graph.py
```

 [Robosuite]: https://robosuite.ai/docs/installation.html