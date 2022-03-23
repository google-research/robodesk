"""Desk environment with Franka Panda arm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

from dm_control import mujoco
from dm_control.utils import inverse_kinematics
import gym
import numpy as np
from PIL import Image

from .utils import CameraSpec, NumPyRNGWrapper, EnvLightManager, TVManager, ButtonManager


class RoboDeskBase(gym.Env):
  r"""
  Multi-task manipulation environment.

  Common Arguments::

    task (str):   Task of the environment, defining the reward function.
                  Default: "open_slide".
    reward (str): Type of the reward, also affecting the reward function.
                  Choices: "dense", "sparse", "success". Default: "dense".
    action_repeat (int): Default: 1.
    episode_length (int): Default: 500.
    image_size (int): Default: 64.

  Following arguments are useful for advanced use cases and customization of the
  environment, including settings for various distractors and noises. If you are
  looking for a simple-to-use API for noisy environment with distractors, see
  RoboDeskNoisy or RoboDeskNoisyWithTV.

  Distractors / Noises Arguments::

    [ Environment Lighting ]
    env_light_noise_strength (float): Controlling level of environment light
                                      jittering and flickering. This tends to
                                      lead to dimmer scene, so adjusting
                                      `headlight_brightness` might be needed.
                                      Should be within [0, 1]. Default: 0.
    headlight_brightness (float): Adjusting the headlight level. Useful when the
                                  rendering is darker than ideal, e.g., due to
                                  noisy lighting. Default: 0.4.

    [ Button Sensor ]
    button_sensor_noise_strength (float): Controlling level of noise in button
                                          sensor reading. Many elements depend
                                          on the amount the button pressed (e.g.,
                                          hue of TV). Turning this on makes the
                                          button reading noisy. Notably this
                                          affects the brightness of the lights,
                                          and the TV hue that is controlled by
                                          button. With this turned on (value > 0)
                                          their brightness will reflect the
                                          noisy sensor reading (rather than
                                          having binary states on vs. off).
                                          Should be within [0, 1]. Default: 0.

    [ Camera Location ]
    camera_noise_strength (float): Controlling level of camera shaking.
                                   Should be within [0, 1]. Default: 0.

    [ TV ]
    tv_video_file_pattern (str): A glob pattern that selects videos to be played
                                 on a TV in the environment. If `None`, TV will
                                 not show any video. Default: None.
  """

  MODEL_PATH: str
  CAMERA_SPEC: CameraSpec

  def __init__(self, task='open_slide', reward='dense', action_repeat=1,
               episode_length=500, image_size=64,
               headlight_brightness=0.4,
               button_sensor_noise_strength=0,
               env_light_noise_strength=0,
               camera_noise_strength=0,
               tv_video_file_pattern=None):
    assert reward in ('dense', 'sparse', 'success'), reward

    # model = os.path.join(os.path.dirname(__file__), 'assets/desks_with_tv.xml')
    self.physics = mujoco.Physics.from_xml_path(self.MODEL_PATH)

    # Adjust headlight
    self.physics.model.vis.headlight.ambient[:] = headlight_brightness * 0.25  # default 0.4 -> 0.1
    self.physics.model.vis.headlight.diffuse[:] = headlight_brightness  # default 0.4 -> 0.4
    self.physics.model.vis.headlight.specular[:] = headlight_brightness * 0.8 + 0.18  # default 0.4 -> 0.5

    # Create a copy for IK
    self.physics_copy = self.physics.copy(share_model=True)
    self.physics_copy.data.qpos[:] = self.physics.data.qpos[:]

    # Robot constants
    self.num_joints = 9
    self.joint_bounds = self.physics.model.actuator_ctrlrange.copy()

    # Environment params
    self.image_size = image_size
    self.action_dim = 5
    self.reward = reward
    self.success = None

    # RNG
    # [env, button, camera, env_light, tv_init, tv_run]
    seed, button_seed, cam_seed, env_light_seed, tv_seed = NumPyRNGWrapper.split_seed(seed=None, n=5)
    self.np_rng = NumPyRNGWrapper(seed)

    # Managers of specific elements
    button_manager = ButtonManager(self.physics, button_sensor_noise_strength, button_seed)
    self.elem_managers = dict(
      camera=self.CAMERA_SPEC.get_camera_manager(self.physics, camera_noise_strength, cam_seed),
      button=button_manager,
      env_light=EnvLightManager(self.physics, swing_scale=env_light_noise_strength,
        flicker_scale=env_light_noise_strength, seed=env_light_seed),
      tv=TVManager(self.physics, tv_video_file_pattern, button_manager, tv_seed)
    )

    # Noises
    self.button_sensor_noise_strength = button_sensor_noise_strength
    self.tv_video_file_pattern = tv_video_file_pattern

    # Action space
    self.end_effector_scale = 0.01
    self.wrist_scale = 0.02
    self.joint_scale = 0.02

    # Episode length
    self.action_repeat = action_repeat
    self.num_steps = 0
    self.episode_length = episode_length
    assert episode_length % action_repeat == 0, "episode_length must be divisible by action_repeat"

    self.original_pos = {}
    self.previous_z_angle = None
    self.total_rotation = 0

    # pylint: disable=g-long-lambda
    self.reward_functions = {
        # Core tasks
        'open_slide': self._slide_reward,
        'open_drawer': self._drawer_reward,
        'push_green': (lambda reward_type: self._button_reward(
            'green', reward_type)),
        'stack': self._stack_reward,
        'upright_block_off_table': (lambda reward_type: self._push_off_table(
            'upright_block', reward_type)),
        'flat_block_in_bin': (lambda reward_type: self._put_in_bin(
            'flat_block', reward_type)),
        'flat_block_in_shelf': (lambda reward_type: self._put_in_shelf(
            'flat_block', reward_type)),
        'lift_upright_block': (lambda reward_type: self._lift_block(
            'upright_block', reward_type)),
        'lift_ball': (lambda reward_type: self._lift_block(
            'ball', reward_type)),

        # Extra tasks
        'push_blue': (lambda reward_type: self._button_reward(
            'blue', reward_type)),
        'push_red': (lambda reward_type: self._button_reward(
            'red', reward_type)),
        'flat_block_off_table': (lambda reward_type: self._push_off_table(
            'flat_block', reward_type)),
        'ball_off_table': (lambda reward_type: self._push_off_table(
            'ball', reward_type)),
        'upright_block_in_bin': (lambda reward_type: self._put_in_bin(
            'upright_block', reward_type)),
        'ball_in_bin': (lambda reward_type: self._put_in_bin(
            'ball', reward_type)),
        'upright_block_in_shelf': (lambda reward_type: self._put_in_shelf(
            'upright_block', reward_type)),
        'ball_in_shelf': (lambda reward_type: self._put_in_shelf(
            'ball', reward_type)),
        'lift_flat_block': (lambda reward_type: self._lift_block(
            'flat_block', reward_type)),
        'tv_green_hue': (lambda reward_type: self._tv_hue('green', reward_type))
    }

    self.core_tasks = list(self.reward_functions)[0:12]
    self.all_tasks = list(self.reward_functions)
    self.task = task
    # pylint: enable=g-long-lambda

  def seed(self, seed=None):
    seed, button_seed, cam_seed, env_light_seed, tv_seed = NumPyRNGWrapper.split_seed(seed, 5)
    self.np_rng.seed(seed)
    self.elem_managers['button'].seed(button_seed)
    self.elem_managers['camera'].seed(cam_seed)
    self.elem_managers['env_light'].seed(env_light_seed)
    self.elem_managers['tv'].seed(tv_seed)

  def get_random_state(self):
    return (self.np_rng.get_random_state(), {k: m.get_random_state() for k, m in self.elem_managers.items()})

  def set_random_state(self, random_state):
    self.np_rng.set_random_state(random_state[0])
    for k, m in self.elem_managers.items():
        m.set_random_state(random_state[1][k])

  @property
  def action_space(self):
    return gym.spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim))

  @property
  def observation_space(self):
    spaces = {
        'image': gym.spaces.Box(
            0, 255, (self.image_size, self.image_size, 3), np.uint8),
        'qpos_robot': gym.spaces.Box(self.joint_bounds[:, 0],
                                     self.joint_bounds[:, 1]),
        'qvel_robot': gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
        'end_effector': gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        'qpos_objects': gym.spaces.Box(-np.inf, np.inf, (26,), np.float32),
        'qvel_objects': gym.spaces.Box(-np.inf, np.inf, (26,), np.float32)}
    return gym.spaces.Dict(spaces)

  def render(self, mode='rgb_array', resize=True):
    for m in self.elem_managers.values():
      m.pre_render()
    return self.elem_managers['camera'].render(
      render_size=min(480, int(self.image_size / 64 * 120)),
      image_size=self.image_size if resize else None,
    )

  def _ik(self, pos):
    out = inverse_kinematics.qpos_from_site_pose(
        self.physics_copy, 'end_effector', pos,
        joint_names=('panda0_joint1', 'panda0_joint2', 'panda0_joint3',
                     'panda0_joint4', 'panda0_joint5', 'panda0_joint6'),
        inplace=True)
    return out.qpos[:]

  def _action_to_delta_joint(self, unscaled_value, joint_bounds):
    """Convert actions from [-1, 1] range to joint bounds."""
    joint_range = joint_bounds[1] - joint_bounds[0]
    return (((unscaled_value + 1) * joint_range) / 2) + joint_bounds[0]

  def _convert_action(self, full_action):
    """Converts action from [-1, 1] space to desired joint position."""
    full_action = np.array(full_action)

    delta_action = full_action[0:3] * self.end_effector_scale
    position = (
        self.physics.named.data.site_xpos['end_effector'] + delta_action)

    joint = self._ik(position)
    delta_wrist = self._action_to_delta_joint(full_action[3],
                                              self.joint_bounds[6])
    joint[6] = ((self.wrist_scale * delta_wrist) +
                self.physics.named.data.qpos[6])
    joint[6] = np.clip(joint[6], self.joint_bounds[6][0],
                       self.joint_bounds[6][1])
    joint[7] = self._action_to_delta_joint(full_action[4],
                                           self.joint_bounds[7])
    joint[8] = joint[7]
    return joint

  def step(self, action):
    total_reward = 0
    for _ in range(self.action_repeat):
      joint_position = self._convert_action(action)
      for _ in range(10):
        self.physics.data.ctrl[0:9] = joint_position[0:9]
        # Ensure gravity compensation stays enabled.
        self.physics.data.qfrc_applied[0:9] = self.physics.data.qfrc_bias[0:9]
        self.physics.step()
      self.physics_copy.data.qpos[:] = self.physics.data.qpos[:]

      for m in self.elem_managers.values():
        m.step()

      if self.reward == 'dense':
        total_reward += self._get_task_reward(self.task, 'dense_reward')
      elif self.reward == 'sparse':
        total_reward += float(self._get_task_reward(self.task, 'success'))
      elif self.reward == 'success':
        if self.success:
          total_reward += 0  # Only give reward once in case episode continues.
        else:
          self.success = self._get_task_reward(self.task, 'success')
          total_reward += float(self.success)
      else:
        raise ValueError(self.reward)

    self.num_steps += self.action_repeat
    if self.episode_length and self.num_steps >= self.episode_length:
      done = True
    else:
      done = False

    return self._get_obs(), total_reward, done, {'discount': 1.0}

  def _get_init_robot_pos(self):
    init_joint_pose = np.array(
        [-0.30, -0.4, 0.28, -2.5, 0.13, 1.87, 0.91, 0.01, 0.01])
    init_joint_pose += 0.15 * self.np_rng.uniform(
        low=self.physics.model.actuator_ctrlrange[:self.num_joints, 0],
        high=self.physics.model.actuator_ctrlrange[:self.num_joints, 1])
    return init_joint_pose

  def reset(self):
    """Resets environment."""
    self.success = False
    self.num_steps = 0

    self.physics.reset()

    # Randomize object positions.
    self.physics.named.data.qpos['drawer_joint'] -= 0.10 * self.np_rng.random()
    self.physics.named.data.qpos['slide_joint'] += 0.20 * self.np_rng.random()

    self.physics.named.data.qpos['flat_block'][0] += 0.3 * self.np_rng.random()
    self.physics.named.data.qpos['flat_block'][1] += 0.07 * self.np_rng.random()
    self.physics.named.data.qpos['ball'][0] += 0.48 * self.np_rng.random()
    self.physics.named.data.qpos['ball'][1] += 0.08 * self.np_rng.random()
    self.physics.named.data.qpos['upright_block'][0] += (
        0.3 * self.np_rng.random() + 0.05)
    self.physics.named.data.qpos['upright_block'][1] += (
        0.05 * self.np_rng.random())

    # Set robot position.
    self.physics.data.qpos[:self.num_joints] = self._get_init_robot_pos()
    self.physics.data.qvel[:self.num_joints] = np.zeros(9)

    # Reset managers
    for m in self.elem_managers.values():
      m.reset()

    # Relax object intersections.
    self.physics.forward()

    # Copy physics state into IK simulation.
    self.physics_copy.data.qpos[:] = self.physics.data.qpos[:]
    self.original_pos['ball'] = self.physics.named.data.xpos['ball']
    self.original_pos['upright_block'] = self.physics.named.data.xpos[
        'upright_block']
    self.original_pos['flat_block'] = self.physics.named.data.xpos['flat_block']

    self.drawer_opened = False

    return self._get_obs()

  def _did_not_move(self, block_name):
    current_pos = self.physics.named.data.xpos[block_name]
    dist = np.linalg.norm(current_pos - self.original_pos[block_name])
    return dist < 0.01

  def _total_movement(self, block_name, max_dist=5.0):
    current_pos = self.physics.named.data.xpos[block_name]
    dist = np.linalg.norm(current_pos - self.original_pos[block_name])
    return dist / max_dist

  def _get_dist_reward(self, object_pos, max_dist=1.0):
    eepos = self.physics.named.data.site_xpos['end_effector']
    dist = np.linalg.norm(eepos - object_pos)
    reward = 1 - (dist / max_dist)
    return max(0, min(1, reward))

  def _slide_reward(self, reward_type='dense_reward'):
    blocks = ['flat_block', 'upright_block', 'ball']
    if reward_type == 'dense_reward':
      door_pos = self.physics.named.data.qpos['slide_joint'][0] / 0.6
      target_pos = (self.physics.named.data.site_xpos['slide_handle'] -
                    np.array([0.15, 0, 0]))
      dist_reward = self._get_dist_reward(target_pos)
      did_not_move_reward = (0.33 * self._did_not_move(blocks[0]) +
                             0.33 * self._did_not_move(blocks[1]) +
                             0.34 * self._did_not_move(blocks[2]))
      task_reward = (0.75 * door_pos) + (0.25 * dist_reward)
      return (0.9 * task_reward) + (0.1 * did_not_move_reward)
    elif reward_type == 'success':
      return 1 * (self.physics.named.data.qpos['slide_joint'] > 0.55)

  def _drawer_reward(self, reward_type='dense_reward'):
    if reward_type == 'dense_reward':
      drawer_pos = abs(self.physics.named.data.qpos['drawer_joint'][0]) / 0.3
      dist_reward = self._get_dist_reward(
          self.physics.named.data.geom_xpos['drawer_handle'])
      return (0.75 * drawer_pos) + (0.25 * dist_reward)
    elif reward_type == 'success':
      return 1 * (self.physics.named.data.qpos['drawer_joint'] < -0.2)

  def _button_reward(self, color, reward_type='dense_reward'):
    press_button = (
        self.physics.named.data.qpos[color + '_light'][0] < -0.00453)
    if reward_type == 'dense_reward':
      dist_reward = self._get_dist_reward(
          self.physics.named.data.xpos[color + '_button'])
      return (0.25 * press_button) + (0.75 * dist_reward)
    elif reward_type == 'success':
      return 1.0 * press_button

  def _stack_reward(self, reward_type='dense_reward'):
    target_offset = [0, 0, 0.0377804]
    current_offset = (self.physics.named.data.xpos['upright_block'] -
                      self.physics.named.data.xpos['flat_block'])

    offset_difference = np.linalg.norm(target_offset - current_offset)

    dist_reward = self._get_dist_reward(
        self.physics.named.data.xpos['upright_block'])

    if reward_type == 'dense_reward':
      return -offset_difference + dist_reward
    elif reward_type == 'success':
      return offset_difference < 0.04

  def _push_off_table(self, block_name, reward_type='dense_reward'):
    blocks = ['flat_block', 'upright_block', 'ball']
    blocks.remove(block_name)
    if reward_type == 'dense_reward':
      block_pushed = (1 - (self.physics.named.data.xpos[block_name][2] /
                           self.original_pos[block_name][2]))
      block_0_stay_put = (1 - self._total_movement(blocks[0]))
      block_1_stay_put = (1 - self._total_movement(blocks[1]))
      reward = ((0.8 * block_pushed) + (0.1 * block_0_stay_put) +
                (0.1 * block_1_stay_put))
      reward = max(0, min(1, reward))
      dist_reward = self._get_dist_reward(
          self.physics.named.data.xpos[block_name])
      return (0.75 * reward) + (0.25 * dist_reward)
    elif reward_type == 'success':
      return 1 * ((self.physics.named.data.qpos[block_name][2] < 0.6) and
                  self._did_not_move(blocks[0]) and
                  self._did_not_move(blocks[1]))

  def _put_in_bin(self, block_name, reward_type='dense_reward'):
    pos = self.physics.named.data.xpos[block_name]
    success = (pos[0] > 0.28) and (pos[0] < 0.52) and (pos[1] > 0.38) and (
        pos[1] < 0.62) and (pos[2] > 0) and (pos[2] < 0.4)
    if reward_type == 'dense_reward':
      dist_reward = self._get_dist_reward(
          self.physics.named.data.xpos[block_name])
      return (0.5 * dist_reward) + (0.5 * float(success))
    elif reward_type == 'success':
      return 1 * success

  def _put_in_shelf(self, block_name, reward_type='dense_reward'):
    x_success = (self.physics.named.data.xpos[block_name][0] > 0.2)
    y_success = (self.physics.named.data.xpos[block_name][1] > 1.0)
    success = x_success and y_success
    blocks = ['flat_block', 'upright_block', 'ball']
    blocks.remove(block_name)
    if reward_type == 'dense_reward':
      target_x_y = np.array([0.4, 1.1])
      block_dist_reward = 1 - (np.linalg.norm(
          target_x_y - self.physics.named.data.xpos[block_name][0:2]))
      dist_reward = self._get_dist_reward(
          self.physics.named.data.xpos[block_name])
      block_0_stay_put = (1 - self._total_movement(blocks[0]))
      block_1_stay_put = (1 - self._total_movement(blocks[1]))
      block_in_shelf = ((0.33 * dist_reward) + (0.33 * block_dist_reward) +
                        (0.34 * float(success)))
      reward = ((0.5 * block_in_shelf) + (0.25 * block_0_stay_put) +
                (0.25 * block_1_stay_put))
      return reward
    elif reward_type == 'success':
      return 1 * success

  def _lift_block(self, block_name, reward_type='dense_reward'):
    if reward_type == 'dense_reward':
      dist_reward = self._get_dist_reward(
          self.physics.named.data.xpos[block_name])
      block_reward = (self.physics.named.data.xpos[block_name][2] -
                      self.original_pos[block_name][2]) * 10
      block_reward = max(0, min(1, block_reward))
      return (0.85 * block_reward) + (0.15 * dist_reward)
    elif reward_type == 'success':
      success_criteria = {'upright_block': 0.86, 'ball': 0.81,
                          'flat_block': 0.78}
      threshold = success_criteria[block_name]
      return 1 * (self.physics.named.data.xpos[block_name][2] > threshold)

  def _tv_hue(self, color, reward_type='dense_reward'):
        tv_manager: TVManager = self.elem_managers['tv']
        assert tv_manager.tv_enabled, "TV-based reward can only be used when TV is enabled"
        cidx = {'red': 0, 'green': 1, 'blue': 2}[color]

        self.elem_managers['tv'].ensure_texure_updated()
        tv_reward = self.elem_managers['tv'].tv_tex[..., cidx].mean() / 255
        if reward_type == 'success':
          return tv_reward
        elif reward_type == 'dense_reward':
          dist_reward = self._get_dist_reward(
              self.physics.named.data.xpos[color + '_button'])
          press_button = (
              self.physics.named.data.qpos[color + '_light'][0] < -0.00453)
          return 0.5 * tv_reward + 0.25 * dist_reward + 0.25 * press_button

  def _get_task_reward(self, task, reward_type):
    reward = self.reward_functions[task](reward_type)
    reward = max(0, min(1, reward))
    return reward

  def _get_obs(self):
    return {'image': self.render(resize=True),
            'qpos_robot': self.physics.data.qpos[:self.num_joints].copy(),
            'qvel_robot': self.physics.data.qvel[:self.num_joints].copy(),
            'end_effector': self.physics.named.data.site_xpos['end_effector'],
            'qpos_objects': self.physics.data.qvel[self.num_joints:].copy(),
            'qvel_objects': self.physics.data.qvel[self.num_joints:].copy()}


class RoboDesk(RoboDeskBase):
  r"""
  Multi-task manipulation environment.

  Arguments::

    task (str):   Task of the environment, defining the reward function.
                  Default: "open_slide".
    reward (str): Type of the reward, also affecting the reward function.
                  Choices: "dense", "sparse", "success". Default: "dense".
    action_repeat (int): Default: 1.
    episode_length (int): Default: 500.
    image_size (int): Default: 64.
    distractors (str or set): Subset of ``{'button', 'env_light', 'camera'}``,
                              specifying which noise distractors are enabled.
                              String "all" means all of them, and string "none"
                              means the empty set. Default: "none".
  """


  MODEL_PATH = os.path.join(os.path.dirname(__file__), 'assets/desk.xml')
  CAMERA_SPEC = CameraSpec()
  AVAILABLE_DISTRACTORS = {'button', 'env_light', 'camera'}

  def __init__(self, task='open_slide', reward='dense', action_repeat=1,
               episode_length=500, image_size=64, distractors='none'):
    if distractors == 'all':
      distractors = self.AVAILABLE_DISTRACTORS
    elif distractors == 'none':
      distractors = set()
    assert isinstance(distractors, set) and self.AVAILABLE_DISTRACTORS.issuperset(distractors), \
      ('RoboDesk supports `distractor` argument being "all", "none", or a set '
       f'containing elements of {self.AVAILABLE_DISTRACTORS}')
    super().__init__(task, reward, action_repeat, episode_length, image_size,
                     button_sensor_noise_strength=1 if 'button' in distractors else 0,
                     env_light_noise_strength=0.7 if 'env_light' in distractors else 0,
                     headlight_brightness=0.9 if 'env_light' in distractors else 0.4,
                     camera_noise_strength=1 if 'camera' in distractors else 0)


class RoboDeskWithTV(RoboDeskBase):
  r"""
  Multi-task manipulation environment with TV. Different from `RoboDesk`, this
  scene contains a TV in addition to the desk. The TV can sequentially play
  videos from disk (see ``tv_video_file_pattern`` argument).

  Arguments::

    task (str):   Task of the environment, defining the reward function.
                  Default: "open_slide".
    reward (str): Type of the reward, also affecting the reward function.
                  Choices: "dense", "sparse", "success". Default: "dense".
    action_repeat (int): Default: 1.
    episode_length (int): Default: 500.
    image_size (int): Default: 96 (higher than default RoboDesk due to further camera view).
    distractors (str or set): Subset of ``{'button', 'env_light', 'camera', 'tv'}``,
                              specifying which noise distractors are enabled.
                              String "all" means all of them, and string "none"
                              means the empty set. Default: "none".
    tv_video_file_pattern (str): A glob pattern that selects videos to be played
                                 on a TV in the environment. If `None`, TV will
                                 not show any video. Default: None.
  """

  MODEL_PATH = os.path.join(os.path.dirname(__file__), 'assets/desks_with_tv.xml')
  CAMERA_SPEC = CameraSpec(
    elevation_offset=13,
    distance_offset=1.15,
    lookat_offset=np.array([0.875, 0.4, 0.875]),
    cropbox_for_render_size_120=np.array([0, 25.892, 120, 120]),
  )
  AVAILABLE_DISTRACTORS = {'button', 'env_light', 'camera', 'tv'}

  def __init__(self, task='open_slide', reward='dense', action_repeat=1,
               episode_length=500, image_size=96, distractors='none',
               tv_video_file_pattern=None):
    if distractors == 'all':
      distractors = self.AVAILABLE_DISTRACTORS
    elif distractors == 'none':
      distractors = set()
    assert isinstance(distractors, set) and self.AVAILABLE_DISTRACTORS.issuperset(distractors), \
      ('RoboDesk supports `distractor` argument being "all", "none", or a set '
       f'containing elements of {self.AVAILABLE_DISTRACTORS}')
    super().__init__(task, reward, action_repeat, episode_length, image_size,
                     tv_video_file_pattern=tv_video_file_pattern if 'tv' in distractors else None,
                     button_sensor_noise_strength=1 if 'button' in distractors else 0,
                     env_light_noise_strength=1 if 'env_light' in distractors else 0,
                     headlight_brightness=0.9 if 'env_light' in distractors else 0.4,
                     camera_noise_strength=1 if 'camera' in distractors else 0)
