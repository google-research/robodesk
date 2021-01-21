"""Tests for google3.experimental.users.hkannan.world_models.franka_desk.robodesk."""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from google3.experimental.users.hkannan.world_models.franka_desk import robodesk
from google3.testing.pybase import googletest


class RobodeskTest(googletest.TestCase):

  def test_environment(self):
    env = robodesk.RoboDesk()
    img = env.render(resize=True)
    action = env.action_space.sample()
    obs = env.reset()
    self.assertEqual(obs['image'].shape, (64, 64, 3))
    obs, _, _, _ = env.step(action)
    self.assertEqual(obs['image'].shape, (64, 64, 3))
    self.assertEqual(action.shape, (5,))
    self.assertEqual(img.shape, (64, 64, 3))


if __name__ == '__main__':
  googletest.main()
