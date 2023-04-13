import scipy.io
import matplotlib.pyplot as plt

# load the .mat file
data = scipy.io.loadmat('./data/train/pulse_camera_data.mat')
# get the pulse array
pulse = data['pulse']
# get the shape of the array
height, width, frames = pulse.shape
# create a figure to show the frames
fig = plt.figure()
# loop through the frames
for i in range(frames):
    # get the i-th frame
    frame = pulse[:, :, i]
    # plot the frame as an image
    plt.imshow(frame, cmap='gray')
    # set the title as the frame number
    plt.title(f'Frame {i}')
    # show the image
    plt.show()