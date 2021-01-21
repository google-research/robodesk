## About

RoboDesk is a multi-task benchmark for fine-grained robotic manipulation tasks
in simulation. The benchmark features 12 core tasks that have been carefully
chosen to represent a wide variety of robotic skills. The benchmark is built on
top of a desk environment which features multiple interactive objects (buttons,
drawers, blocks of various sizes) and a Franka Panda robotic arm.

Our environment is built to allow transfer and sharing of data across tasks,
enabling multi-task learning. In addition to our 12-task benchmark, we include
reward functions for 45 total tasks. However, we encourage users to only report
numbers on our pre-defined 12-task benchmark in order to keep results
standardized. We deliberately chose only 12 tasks in order to minimize compute
resources required and make our benchmark accessible to as many researchers as
possible. These 12 manipulation tasks test a wide variety of skills (listed
below), and our benchmark features visual complexity and diversity of object
shapes, sizes, and textures.

We have also carefully developed each component of our environment (step(),
reset(), action space, etc.) by testing each component for possible physics
issues and ensuring that they do not arise. Furthermore, the tasks have been
carefully tested and calibrated with RL agents to ensure that they are of
appropriate difficulty. Similar to dm_control, we include both dense and sparse
reward functions with human-interpretable fixed ranges (between 0 and 1).

Finally, we designed our Python environment to have as little code as possible
and to be a self-contained, simple implementation that does not rely on large
frameworks. We hope that this simplicity will help with in-depth understanding
and debugging and will also serve as a blueprint for new environments.

## Installation

Dependencies: gym, Mujoco, dm_control

pip3 install --user git+git://github.com/deepmind/dm_control.git

pip3 install --user gym python3

python3 robodesk_example.py
