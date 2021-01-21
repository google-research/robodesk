from PIL import Image
import robodesk

env = robodesk.RoboDesk()
env.reset()
img = env.render(resize=True)
Image.fromarray(img).save('/tmp/desk.png')
env.step([1, 0, 0, 0, 0])
