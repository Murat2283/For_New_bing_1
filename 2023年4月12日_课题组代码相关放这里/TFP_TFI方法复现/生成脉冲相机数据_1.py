
import numpy as np
import scipy.io

# set the parameters
height = 640 # the height of the pulse array
width = 480 # the width of the pulse array
frames = 1000 # the number of frames
threshold = 100 # the threshold for generating a pulse
scene = 'outdoor' # the scene type
target = 'red balloon' # the target type

# create an empty pulse array
pulse = np.zeros((height, width, frames), dtype=np.uint8)

# create a function to simulate the light intensity of a pixel
def light_intensity(x, y, t):
    # use different formulas for different scenes and targets
    if scene == 'outdoor' and target == 'red balloon':
        # simulate a bright and sunny day
        intensity = 200 + np.random.randint(-10, 10)
        # simulate a red balloon flying from left to right
        cx = t * 0.5 + 50 # the x coordinate of the balloon center
        cy = height / 2 + np.sin(t * 0.01) * 50 # the y coordinate of the balloon center
        r = 20 # the radius of the balloon
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) # the distance from the pixel to the balloon center
        if d < r:
            # inside the balloon, reduce the intensity by a factor of 0.8
            intensity *= 0.8
        return intensity
    else:
        # for other scenes and targets, return a random intensity
        return np.random.randint(0, 255)

# loop through the pixels and frames
for x in range(height):
    for y in range(width):
        for t in range(frames):
            # get the light intensity of the pixel at time t
            intensity = light_intensity(x, y, t)
            # compare it with the threshold
            if intensity > threshold:
                # generate a pulse and reset the intensity to zero
                pulse[x, y, t] = 1
                intensity = 0
            else:
                # accumulate the intensity for the next frame
                intensity += light_intensity(x, y, t + 1)

# save the pulse array as a .mat file
scipy.io.savemat('./data/train/pulse_camera_data.mat', {'pulse': pulse})