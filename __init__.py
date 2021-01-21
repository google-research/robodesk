from gym.envs import registration
from google3.experimental.users.hkannan.world_models.franka_desk.franka import Franka

registration.register(id='MultiTaskDesk-v0', entry_point=Franka)
