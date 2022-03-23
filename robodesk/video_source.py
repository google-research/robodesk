import numpy as np
import cv2

from .utils import NumPyRNGWrapper


class RandomVideoSource(object):
    r"""
    Class that manages a list of video frames collected from video files. The
    default contrast change and sharpening values are chosen to make video more
    legible when rendered in RoboDesk environment.

    attr:`contrast_change` and :attr:`sharpen` processes the loaded frames,
    usually for the purpose of better visual clarity.
    """
    def __init__(self, shape, video_files, num_frames=1000,
                 interpolation=cv2.INTER_NEAREST, contrast_change=20, sharpen=0.5):
        self.height, self.width = shape
        self.init_np_rng, self.np_rng = [NumPyRNGWrapper(s) for s in NumPyRNGWrapper.split_seed(None, 2)]
        self.video_files = video_files
        self.num_frames = num_frames
        self.interpolation = interpolation
        self.contrast_change = contrast_change
        self.sharpen = sharpen
        self.frames_loaded = False
        self.reset()

    def seed(self, seed):
        init_np_rng, np_rng = [NumPyRNGWrapper(s) for s in NumPyRNGWrapper.split_seed(seed, 2)]
        return self.set_random_state((init_np_rng.get_random_state(), np_rng.get_random_state()))  # go through the checks

    def get_random_state(self):
        return (self.init_random_state_before_loading, self.np_rng.get_random_state())

    def set_random_state(self, random_state):
        if self.frames_loaded:
            # If frames have been loaded, we have consumed `init_np_rng`. Since
            # the set of loaded frames depends on the `init_np_rng` state,
            # future loaded random state must contain the same `init_np_rng`
            # state that was used to load the frames, otherwise it would be
            # inconsistent as the different `init_np_rng` state would have
            # loaded a different set of frames.
            assert np.array_equiv(
                np.asarray(self.init_random_state_before_loading, dtype=object),
                np.asarray(random_state[0], dtype=object),
            ), (
                "Incompatible random state for RandomVideoSource that have already loaded frames. "
                "Either set seed / random state before loading frames (i.e., first rendering) or "
                "use the same seed / random state that RandomVideoSource uses when loading random frames."
            )
        else:
            # If not loaded yet, okay to use any rng
            self.init_np_rng.set_random_state(random_state[0])
            self.init_random_state_before_loading = self.init_np_rng.get_random_state()
        self.np_rng.set_random_state(random_state[1])

    def load_frames_if_needed(self):
        if self.frames_loaded:
            return
        self.init_random_state_before_loading = self.init_np_rng.get_random_state()
        self.frames = np.empty(
            (self.num_frames, self.height, self.width, 3), dtype=np.uint8)
        frame_ii = 0
        try:
            import skvideo.io
        except ImportError:
            raise RuntimeError('TV videos requires scikit-video')
        while True:
            for video_f in self.init_np_rng.permutation(self.video_files):
                video = skvideo.io.vread(video_f)
                for video_ii in range(video.shape[0]):
                    self.frames[frame_ii] = self.process_frame(video[video_ii])
                    frame_ii += 1
                    if frame_ii == self.num_frames:
                        self.frames_loaded = True
                        return  # always exit here
            if frame_ii == 0:
                raise RuntimeError('Given videos do not have any frame')

    def process_frame(self, frame):
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=self.interpolation)
        if self.contrast_change != 0:
            contrast_delta = max(-127, min(127, float(self.contrast_change)))
            f = 131 * (contrast_delta + 127) / (127 * (131 - contrast_delta))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            frame = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)
        if self.sharpen != 0:
            s = float(self.sharpen)
            kernel = np.array([
                [0, -s, 0],
                [-s, 1 + 4 * s, -s],
                [0, -s, 0],
            ])
            frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
        return frame

    def reset(self):
        self.frame_ii = self.np_rng.integers(0, self.num_frames)

    def get_image(self):
        self.load_frames_if_needed()
        return self.frames[self.frame_ii % self.num_frames]

    def step(self):
        self.frame_ii += 1



class ConcatRollingImageSource(object):
    def __init__(self, sources, axis=1):
        assert len(sources) > 0
        self.sources = sources
        self.axis = axis
        self.roll_offset = 0

    def seed(self, seed):
        for source, s in zip(self.sources, NumPyRNGWrapper.split_seed(seed, len(self.sources))):
            source.seed(s)

    def get_random_state(self):
        return [source.get_random_state() for source in self.sources]

    def set_random_state(self, random_state):
        for source, source_random_state in zip(self.sources, random_state):
            source.set_random_state(source_random_state)

    def get_image(self):
        image: np.ndarray = np.concatenate(
            [s.get_image() for s in self.sources],
            axis=self.axis,
        )
        if self.roll_offset != 0:
            image = np.roll(image, axis=self.axis, shift=self.roll_offset)
        return image

    def step(self):
        self.roll_offset += 1
        for s in self.sources:
            s.step()

    def reset(self):
        self.roll_offset = 0
        for s in self.sources:
            s.reset()
