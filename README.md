## Disclaimer: Not an official Google product.

## About

RoboDesk is a benchmark designed for multi-task reinforcement learning on pixels. The benchmark features 9 core tasks that have been carefully chosen to represent a wide variety of robotic skills. The benchmark is built on top of a desk environment which features multiple interactive objects (buttons, drawers, blocks of various sizes) and a Franka Panda robotic arm. 

Our environment is built to allow transfer and sharing of data across tasks, enabling multi-task learning. We chose 9 tasks in order to minimize compute resources required and make our benchmark accessible to as many researchers as possible. Our benchmark features visual complexity and diversity of object shapes, sizes, and textures. We randomize the starting position of the objects on our desk, the drawer, and the moving slide to avoid memorization by the agent.

We also carefully developed each component of our environment (step(), reset(), action space, etc.) by keeping in mind common physics errors that arise. We extensively tested the physics of the environment to ensure its robustness and to avoid errors that can happen with complicated multi-object environments (e.g. objects passing through each other, simulation instability). Furthermore, the tasks have been tested and calibrated with RL agents to ensure that they are of appropriate difficulty. Similar to dm_control, we include both dense and sparse reward functions with human-interpretable fixed ranges (between 0 and 1). 

Finally, we designed our Python environment to be a self-contained, simple implementation that can serve as a blueprint for new Mujoco environments. Our environment does not rely on large frameworks and has as little code as possible. In addition to serving as a blueprint, we hope that the environment's simplicity will help with in-depth understanding and debugging.


## Installation

Dependencies: gym, Mujoco, dm_control

pip3 install --user git+git://github.com/deepmind/dm_control.git

pip3 install --user gym python3

python3 robodesk_example.py
