import numpy as np
import scipy.io
import torch

# set the parameters
height = 480 # the height of the pulse array
width = 480 # the width of the pulse array
frames = 1000 # the number of frames
threshold = 100 # the threshold for generating a pulse
scene = 'outdoor' # the scene type
target = 'red balloon' # the target type
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" #murat
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# create an empty pulse array on GPU
pulse = torch.zeros((height, width, frames), dtype=torch.uint8, device='cuda')
# create an empty time array on GPU
time = torch.zeros((height, width, frames), dtype=torch.float32, device='cuda')

# create a function to simulate the light intensity of a pixel on GPU

#---------------------------------------------------------------

# create a vectorized version of the light_intensity function
def vectorize(func):
    # a decorator to vectorize a function
    def wrapper(*args):
        # get the shape of the arguments
        shape = args[0].shape
        # flatten the arguments
        args = [arg.flatten() for arg in args]
        # apply the function to each element
        result = [func(*arg) for arg in zip(*args)]
        # convert the result to a tensor
        result = torch.tensor(result, device='cuda')
        # reshape the result to the original shape
        result = result.reshape(shape)
        return result
    return wrapper
@vectorize
def light_intensity(x, y, t):
    # use different formulas for different scenes and targets
    if scene == 'outdoor' and target == 'red balloon':
        # simulate a bright and sunny day
        intensity = 200 + torch.randint(-10, 10, (1,), device='cuda')
        t = t.clone().detach().to(device='cuda') # convert t to a Tensor
        # simulate a red balloon flying from left to right
        cx = t * 0.5 + 50 # the x coordinate of the balloon center
        cy = height / 2 + torch.sin(t * 0.01) * 50 # the y coordinate of the balloon center
        r = 20 # the radius of the balloon
        d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) # the distance from the pixel to the balloon center
        if d < r:
            # inside the balloon, reduce the intensity by a factor of 0.8
            intensity *= 0.8
        return intensity
    else:
        # for other scenes and targets, return a random intensity
        return torch.randint(0, 255, (1,), device='cuda')



# create arrays of x, y, and t coordinates on GPU
x = torch.arange(height, device='cuda').reshape(height, 1, 1)
y = torch.arange(width, device='cuda').reshape(1, width, 1)
t = torch.arange(frames, device='cuda').reshape(1, 1, frames)

x = x.transpose(0, 1) # swap the first and second dimensions of x
y = y.transpose(0, 1) # swap the first and second dimensions of y
# get the light intensity of all pixels and frames at once

intensity = light_intensity(x, y, t)

# compare it with the threshold and generate pulses
pulse = intensity > threshold

# reset the intensity to zero where pulses are generated
intensity[pulse] = 0

# record the timestamp of the pulses
time[pulse] = t[pulse] / frames # normalize the timestamp to [0, 1]

# accumulate the intensity for the next frame
intensity += light_intensity(x, y, t + 1)
#---------------------------------------------------------------



# move the pulse and time arrays to CPU and convert them to numpy arrays
pulse = pulse.cpu().numpy()
time = time.cpu().numpy()

# save the pulse and time arrays as a .mat file
scipy.io.savemat('./data/pulse_camera_data.mat', {'spike': pulse, 'time': time})