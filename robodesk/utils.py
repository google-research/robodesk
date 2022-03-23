import glob

import cv2
import numpy as np
from PIL import Image

from dm_control import mujoco
from dm_control.mujoco.engine import Pose


class NumPyRNGWrapper(object):
    r"""
    A wrapper for managing NumPy old and new random APIs.
    """

    def __init__(self, seed):
        self.seed(seed)

    def seed(self, seed=None):
        # seed: Union[int, np.random.SeedSequence, None]
        if hasattr(np.random, 'Generator'):
            self.rng = np.random.Generator(np.random.PCG64(seed))
        else:
            self.np_rng = np.random.RandomState(seed=seed)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self.rng.uniform(low=low, high=high, size=size)

    def random(self):
        return self.rng.random()

    def integers(self, *args, **kwargs):
        if hasattr(np.random, 'Generator'):
            return self.rng.integers(*args, **kwargs)
        else:
            return self.rng.randint(*args, **kwargs)

    def normal(self, loc=0, scale=1, size=None):
        return self.rng.normal(loc=loc, scale=scale, size=size)

    def permutation(self, list):
        return self.rng.permutation(list)

    def get_random_state(self):
        if hasattr(np.random, 'Generator'):
            return self.rng.bit_generator.state
        else:
            return self.rng.get_state()

    def set_random_state(self, random_state):
        if hasattr(np.random, 'Generator'):
            self.rng.bit_generator.state = random_state
        else:
            self.rng.set_state(random_state)

    @staticmethod
    def split_seed(seed, n: int):
        # Splits a seed into n seeds
        if hasattr(np.random, 'Generator'):
            if not isinstance(seed, np.random.SeedSequence):
                seed = np.random.SeedSequence(seed)
            seeds = seed.spawn(n)
        else:
            seeds = np.random.RandomState(seed=seed).randint(2 ** 32, size=n).tolist()
        return seeds


# Get camera loc from a dm_control.mujoco.engine.Pose
def pose2from(pose):
    x, y, z = pose.lookat
    flat_distance = pose.distance * np.cos(pose.elevation * np.pi / 180)
    lookfrom = np.array([
        x - flat_distance * np.cos(pose.azimuth * np.pi / 180),
        y - flat_distance * np.sin(pose.azimuth * np.pi / 180),
        z - pose.distance * np.sin(pose.elevation * np.pi / 180),
    ])
    return lookfrom


# Construct a dm_control.mujoco.engine.Pose from camera loc and target loc
def from2pose(lookfrom, lookat):
    at2from = lookat - lookfrom
    distance = np.linalg.norm(at2from)
    elevation = np.arcsin(at2from[-1] / distance) * 180 / np.pi
    azimuth = np.arctan2(at2from[1], at2from[0]) * 180 / np.pi
    return Pose(
        lookat=lookat,
        distance=distance,
        azimuth=azimuth,
        elevation=elevation,
    )


class SmoothRandomWalker(object):
    r"""
    A simple implementation of a smooth random N-d walk. We only really just
    choose the `pullr` argument, which roughly says how large the range is. But
    below is the entire process description for completeness.


    The random walk tracks (location, velocity, acceleration). At each step,

     + Acceration update:

       A random N-d acceleration change (with random direction) is sampled and
       applied, simulating random forces.

       If the location is `> pullr` distance away from origin, a pulling
       acceleration `-pull * location` is used instead.

     + Velocity update:

       Additively affected by the acceration, and then decayed by `vdecay`.

     + Location update:

       Additively affected by the velocity.

    Finally, `loc_scale` optionally scales the location before yielding it.

    `time_scale` roughly tunes the simulation speed.
    """
    def __init__(self, dim=3, pull=0.1, pullr=0.075, randastddev=0.005,
                 vdecay=0.9, loc_scale=1, time_scale=0.5, *, np_rng: NumPyRNGWrapper):
        self.dim = dim
        self.pull = pull
        self.pullr = pullr
        self.randastddev = randastddev
        self.vdecay = vdecay
        self.loc_scale = loc_scale
        self.time_scale = time_scale
        self.np_rng = np_rng

    def __iter__(self):
        if self.loc_scale == 0:
            while True:
                yield 0
        else:
            loc = self.np_rng.normal(size=[self.dim], scale=self.pullr / np.sqrt(self.dim))
            v = self.np_rng.normal(size=[self.dim], scale=self.randastddev)
            while True:
                yield loc * self.loc_scale
                a = self.np_rng.normal(size=[self.dim], scale=self.randastddev)
                if np.linalg.norm(loc) >= self.pullr:
                    a = -self.pull * loc
                v += a * self.time_scale
                v = v * (self.vdecay ** self.time_scale)
                loc = loc + v * self.time_scale


class CameraSpec(object):
    r"""
    Represents a camera at a specific location aimed at a specific direction.
    This is specified via offsets with the default camera location (`default_pose`).

    The camera can be noisily jittered, controlled via `jitter_scale` argument.

    NB: Jittering is applied at the default camera location, before applying
        offsets so that the same `jitter_scale` roughly translates to similar
        amounts of noise regardless of the offset.
    """
    default_pose: Pose = Pose(
        elevation=-60,
        distance=1.8,
        azimuth=90,
        lookat=np.array([0, 0.535, 1.1]),
    )
    default_lookfrom: np.ndarray = pose2from(default_pose)

    offset_pose: Pose
    cropbox_for_render_size_120: np.ndarray

    def __init__(self, elevation_offset=0, distance_offset=0, azimuth_offset=0,
                 lookat_offset=np.array((0, 0, 0)),
                 cropbox_for_render_size_120=np.array((16.75, 25.0, 105.0, 88.75))):
        self.offset_pose = Pose(
            elevation=elevation_offset,
            distance=distance_offset,
            azimuth=azimuth_offset,
            lookat=lookat_offset,
        )
        self.cropbox_for_render_size_120 = cropbox_for_render_size_120

    def get_camera_manager(self, physics, jitter_scale, np_rng):
        return CameraManager(self, physics, jitter_scale, np_rng)


class EnvElementManager(object):
    r"""
    Managing a single element of the environment.
    """

    def reset(self):
        r"""
        Called in env.reset()
        """
        pass

    def step(self):
        r"""
        Called after each physics step. I.e., if action_repeat = N, this gets
        called N times in env.step(...), *before* any rendering or reward
        computation.
        """
        pass

    def pre_render(self):
        r"""
        Called in env.render(...) right before rendering.
        """
        pass

    def seed(self, seed):
        pass

    def get_random_state(self):
        pass

    def set_random_state(self, random_state):
        pass



class CameraManager(EnvElementManager):
    r"""
    Handles noisy camera.
    """

    def __init__(self, camera_spec: 'CameraSpec', physics, jitter_scale, seed):
        self.camera_spec = camera_spec
        self.physics = physics
        self.np_rng = NumPyRNGWrapper(seed)
        self.jitter_scale = jitter_scale
        self.reset()

    def seed(self, seed):
        self.np_rng.seed(seed)

    def get_random_state(self):
        return self.np_rng.get_random_state()

    def set_random_state(self, random_state):
        self.np_rng.set_random_state(random_state)

    def reset(self):
        self.jitter_iter = dict(
            lookfrom=iter(SmoothRandomWalker(loc_scale=0.3 * self.jitter_scale,
                                             np_rng=self.np_rng)),
            lookat=iter(SmoothRandomWalker(loc_scale=0.65 * self.jitter_scale,
                                           np_rng=self.np_rng)),
        )
        self.step()

    def step(self):
        self.jitter_amount = {k: next(v) for k, v in self.jitter_iter.items()}

    def render(self, render_size, image_size=None):
        # Apply jittering
        pose = from2pose(
            self.camera_spec.default_lookfrom + self.jitter_amount['lookfrom'],
            self.camera_spec.default_pose.lookat + self.jitter_amount['lookat'],
        )
        # Apply offset
        pose = Pose(
            elevation=pose.elevation + self.camera_spec.offset_pose.elevation,
            distance=pose.distance + self.camera_spec.offset_pose.distance,
            azimuth=pose.azimuth + self.camera_spec.offset_pose.azimuth,
            lookat=pose.lookat + self.camera_spec.offset_pose.lookat,
        )
        # Apply pose to camera
        camera = mujoco.MovableCamera(
            physics=self.physics, height=render_size, width=render_size)
        camera.set_pose(*pose)
        # Render
        image = camera.render(depth=False, segmentation=False)
        camera._scene.free()
        if image_size is not None:
            cropbox = self.camera_spec.cropbox_for_render_size_120 * render_size / 120
            image = Image.fromarray(image).crop(box=cropbox)
            image = image.resize([image_size, image_size], resample=Image.LANCZOS)
            image = np.asarray(image)
        return image


class EnvLightManager(EnvElementManager):
    r"""
    Helps adjusting env lights (total 3) according to flickering and jittering/swinging.
    """

    # Radius of light swing
    SWING_RADIUS = np.array([0.5, 1 / np.sqrt(2), 1])
    # Each time step, apply an angular velocity sampled from N(mu, sigma).clamp(0, 0.8)
    SWING_ANGULAR_SPEED_MEAN = np.array([np.sqrt(5), 1, 1 / np.sqrt(3)]) / 30
    SWING_ANGULAR_SPEED_STDDEV = np.array([np.sqrt(5), 1, 1 / np.sqrt(3)]) / 60
    SWING_ANGULAR_SPEED_MAX = 0.8

    # Flicking is brightness adjustment, modelling as a sine curve + noise:
    #   brightness <- sin( T * period + t0 ) * magnitude + offset + N(0, sigma)
    #   brightness <- clamp(brightness, [0, 1])
    FLICKER_PERIOD = np.array([1, np.sqrt(8), np.sqrt(28)]) / 15
    FLICKER_SINE_MAGNITUDE = np.array([0.3, 0.3, 0.3])
    FLICKER_SINE_OFFSET = -0.15
    FLICKER_NOISE_STDDEV = np.array([0.15, 0.15, 0.15])
    FLICKER_BRIGHTNESS_MAX = 1

    def __init__(self, physics, swing_scale, flicker_scale, seed):
        self.physics = physics
        self.swing_scale = swing_scale
        self.flicker_scale = flicker_scale
        self.np_rng = NumPyRNGWrapper(seed)
        self.initial_diffuse = self.physics.model.light_diffuse[:, 0].copy()
        self.initial_pos = self.physics.model.light_pos.copy()
        self.initial_sq_dist = (self.initial_pos ** 2).sum(-1)
        self.reset()

    def seed(self, seed):
        self.np_rng.seed(seed)

    def get_random_state(self):
        return self.np_rng.get_random_state()

    def set_random_state(self, random_state):
        self.np_rng.set_random_state(random_state)

    def reset(self):
        self.swing_angle = self.np_rng.normal(size=3)
        self.flicker_period_offset = self.np_rng.normal(size=3)
        self.num_time_steps = 0
        self.mujoco_outdated = True

    def step(self):
        self.mujoco_outdated = True
        # Track timestep for flicker computation
        self.num_time_steps += 1
        # Swing
        swing_angular_speed = self.np_rng.normal(
            loc=self.SWING_ANGULAR_SPEED_MEAN,
            scale=self.SWING_ANGULAR_SPEED_STDDEV,
        )
        swing_angular_speed = np.clip(swing_angular_speed, 0, self.SWING_ANGULAR_SPEED_MAX)
        self.swing_angle += swing_angular_speed
        self.swing_angle %= 2 * np.pi

    def pre_render(self):
        if not self.mujoco_outdated:
            return
        # Emit to mujoco
        t = self.num_time_steps

        # Swing
        theta = self.swing_angle
        radius = self.SWING_RADIUS * self.swing_scale  # scale here
        loc_delta = np.stack([np.sin(theta), np.cos(theta)], axis=-1) * radius[:, None]
        self.physics.model.light_pos[:, :2] = loc_delta + self.initial_pos[:, :2]
        curr_sq_dist = (self.physics.model.light_pos ** 2).sum(-1)
        light_dir = -self.physics.model.light_pos / (curr_sq_dist ** 0.5)[:, None]
        self.physics.model.light_dir[:] = light_dir

        # Flicker, i.e., brightness
        br_mean = np.sin(
            self.flicker_period_offset + t * self.FLICKER_PERIOD
        ) * self.FLICKER_SINE_MAGNITUDE + self.FLICKER_SINE_OFFSET
        br = self.np_rng.normal(loc=br_mean, scale=self.FLICKER_NOISE_STDDEV)
        br = np.clip(br, 0, self.FLICKER_BRIGHTNESS_MAX)
        # Swing further -> dimmer
        distance_sq_ratio = self.initial_sq_dist / curr_sq_dist
        br = br * distance_sq_ratio
        # Apply flicker scaling here
        br = self.flicker_scale * br + (1 - self.flicker_scale) * self.initial_diffuse
        for ii in range(3):
            self.physics.model.light_diffuse[ii, :] = br[ii]

        self.mujoco_outdated = False


class ButtonManager(EnvElementManager):
    r"""
    Button noise is an additive quantity that gets applied to normal button
    readings. The noise follows a diffusion-like process like the following:

        # A hidden diffusion process that spreads to N(0, 1)
        corr_noise_0 ~ N(0, 1)
        corr_noise_t = alpha * corr_noise_{t-1} + (1 - alpha) * N(0, 1)

        # Real additive noise is just that plus a small noise
        noise_t = corr_noise_scale * corr_noise_t + iid_noise_scale * N(0, 1)
    """
    CORR_NOISE_DIFFUSE_ALPHA = 0.96 ** 0.5

    def __init__(self, physics, noise_scale, seed):
        self.physics = physics
        self.noise_scale = noise_scale
        if self.noise_scale != 0:
            # Turn off original noiseless light mechanism.
            # We will use the althernative one that supports noise.
            # See NOTE [ Button to Light Mechanism ] in desk.xml.
            for c in ['red', 'green', 'blue']:
                physics.named.model.geom_rgba[f'{c}_light_rise_cylinder'][-1] = 0
        self.corr_noise_scale = noise_scale * 0.295
        self.iid_noise_scale = noise_scale * 0.0125
        self.np_rng = NumPyRNGWrapper(seed)
        self.reset()
                    # button_to_light_offset=0.78,

    def seed(self, seed):
        self.np_rng.seed(seed)

    def get_random_state(self):
        return self.np_rng.get_random_state()

    def set_random_state(self, random_state):
        self.np_rng.set_random_state(random_state)

    def get_normalized_button(self):
        r"""
        Returns normalized "how much is it pressed" for each button.

        0 means not pressed at all.
        1 means fully pressed.

        In the noiseless setting (based on the initial RoboDesk xml), a
        reading value > 0.9 is interpreted as "button being pressed", which
        turns on the corresponding light which has alpha 0.4. This logic is
        entirely handled by mujoco (and specified in xml).

        For noisy environments, the lights are more than just being binary, and
        have brightness (alpha) levels. The linear between button and brightness
        is coded in `_update_noises_and_lights()`.

        See also NOTE [ Button to Light Mechanism ] in desk.xml.
        """
        if self.button_reading_outdated:
            # get button joint from negated light joints
            button_jnt = -np.asarray([
                self.physics.named.data.qpos['red_light'][0],
                self.physics.named.data.qpos['green_light'][0],
                self.physics.named.data.qpos['blue_light'][0],
            ], dtype=np.float32)
            # Convert JNT via linear tsfm
            self.button_reading = button_jnt / 0.005 + self.noise
            self.button_reading_outdated = False
        return self.button_reading

    def _update_button_noises(self, corr_noise):
        assert self.noise_scale != 0
        self.corr_noise = corr_noise
        self.noise = self.corr_noise * self.corr_noise_scale + self.np_rng.normal(size=(3,)) * self.iid_noise_scale
        self.button_reading_outdated = True

    def _update_lights_based_noisy_button(self):
        assert self.noise_scale != 0
        # Update lights based on (noisy) reading: a linear transform from button reading to light alpha
        light_alpha = (self.get_normalized_button() - 0.12) * 6 + 0.4  # 0.9 leads to very dim lights, so use an adjusted bias
        light_alpha = np.clip(light_alpha, -1, 1)
        for ci, c in enumerate(['red', 'green', 'blue']):
            a = light_alpha[ci].item()
            self.physics.named.model.geom_rgba[f'{c}_light_overlay'][-1] = max(a, 0)
            self.physics.named.model.geom_rgba[f'{c}_light_neg_overlay'][-1] = max(-a, 0)

    def reset(self):
        self.button_reading_outdated = True
        if self.noise_scale == 0:
            self.noise = 0
        else:
            self._update_button_noises(self.np_rng.normal(size=(3,)))
            self._update_lights_based_noisy_button()

    def step(self):
        if self.noise_scale != 0:
            self._update_button_noises(
                self.corr_noise * self.CORR_NOISE_DIFFUSE_ALPHA +
                self.np_rng.normal(size=(3,)) * (1 - self.CORR_NOISE_DIFFUSE_ALPHA)
            )  # diffuse
            self._update_lights_based_noisy_button()


from .video_source import RandomVideoSource, ConcatRollingImageSource


class TVManager(EnvElementManager):
    r"""
    NOTE [ TV Frame Sampling ]

    Unlike other elements of the environment, the TV (when enabled) consumes RNG at initialization, randomly
    sampling the video files to load. As the consequence, this means that after initialization, this specific
    environment's TV can only display the loaded files (regardless of `.seed/reset()` calls), and that different
    seeds at initialization causes different set of videos being plays (regardless of `.seed/reset()` calls).

    While this is a bit contrary to the common assumption that calling `env.seed(s); env.reset()` should always make
    `env` behave the same after wards, this is done for three reasons:
    + Natural video noise is meant to be ignored. Having training and testing env using different sets of videos
      better test the generalization of removal (i.e., whether the agent just overfit to the training videos).
    + This follows the behavior of many RL codebases for learning with distractors. Notably, [1] introduces noisy
      video backgrounds and utlizes this strategy (probably for the above reason).
    + Loading videos is expensive. Doing so at every `.seed/reset()` is not practical.

    [1] Zhang, Amy, et al. "Learning invariant representations for reinforcement learning without reconstruction."
    ICLR 2021.
    """

    def __init__(self, physics, video_file_pattern, button_manager: ButtonManager, seed,
                 button_controls_hue=np.array([0., 1., 0.])):  # by default, only let green button control TV green hue
        self.physics = physics
        self.video_file_pattern = video_file_pattern
        self.tv_enabled = video_file_pattern is not None
        if self.video_file_pattern is None:
            return

        # Get the mujoco texture for TV
        assert 'tv_texture' in physics.named.model.tex_adr.axes.row.names, 'Model should have tv_texture'
        tex_adr = physics.named.model.tex_adr['tv_texture']
        tex_w = physics.named.model.tex_width['tv_texture']
        tex_h = physics.named.model.tex_height['tv_texture']
        size = tex_w * tex_h * 3
        full_tex = physics.named.model.tex_rgb[tex_adr:tex_adr + size].reshape(tex_h, tex_w, 3)
        front_h0 = tex_h * 3 // 6
        front_ht = tex_h * 4 // 6
        full_tex[:front_h0] = 0
        full_tex[front_ht:] = 0
        self.tv_tex = full_tex[front_h0:front_ht]
        self.tv_texid = physics.named.model.mat_texid['tv_material']

        # Fetch the TV video files
        tv_video_files = sorted(glob.glob(video_file_pattern))
        self.tv_source = ConcatRollingImageSource(
            [RandomVideoSource((60, 60), tv_video_files) for _ in range(2)])
        self.tv_source.seed(seed)

        # Button effects
        self.button_manager = button_manager
        self.button_controls_hue = button_controls_hue

        self.reset()

    def seed(self, seed):
        if self.video_file_pattern is None:
            return
        self.tv_source.seed(seed)

    def get_random_state(self):
        if self.video_file_pattern is None:
            return None
        return self.tv_source.get_random_state()

    def set_random_state(self, random_state):
        if self.video_file_pattern is None:
            assert random_state is None
            return
        self.tv_source.set_random_state(random_state)

    def reset(self):
        if self.video_file_pattern is None:
            return
        self.tv_source.reset()
        self.texure_outdated = True  # whether the CPU mujoco model needs updating
        self.mujoco_outdated = True  # whether the GPU mujoco model needs updating

    def step(self):
        if self.video_file_pattern is None:
            return
        self.tv_source.step()
        self.texure_outdated = True

    def pre_render(self):
        if self.video_file_pattern is None:
            return
        self.ensure_mujoco_updated()

    def ensure_texure_updated(self):
        if self.video_file_pattern is None:
            return
        if self.texure_outdated:
            img = self.tv_source.get_image()

            # adjust based on button:
            #   not pressed -> 0 hue change
            #   fully pressed -> 0.75 hue change
            lerp_w = self.button_manager.get_normalized_button() * self.button_controls_hue * 0.75
            lerp_w = np.clip(lerp_w, 0, 1)
            img = np.round(img * (1 - lerp_w) + 255 * lerp_w).astype(np.uint8)

            cv2.resize(
                img,
                self.tv_tex.shape[:2][::-1],
                dst=self.tv_tex,
                interpolation=cv2.INTER_NEAREST,
            )
            self.mujoco_outdated = True
            self.texure_outdated = False

    def ensure_mujoco_updated(self):
        if self.video_file_pattern is None:
            return
        self.ensure_texure_updated()
        if self.mujoco_outdated:
            from dm_control.mujoco.wrapper.mjbindings import mjlib
            # push updated tex to GPU
            with self.physics.contexts.gl.make_current() as ctx:
                ctx.call(
                    mjlib.mjr_uploadTexture,
                    self.physics.model.ptr,
                    self.physics.contexts.mujoco.ptr,
                    self.tv_texid,
                )
            self.mujoco_outdated = False

