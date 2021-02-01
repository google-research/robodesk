"""
Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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
