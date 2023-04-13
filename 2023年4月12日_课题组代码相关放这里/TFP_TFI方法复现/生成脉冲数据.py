# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Define the pulse parameters
amplitude = 1 # Pulse amplitude
width = 0.1 # Pulse width
rise_time = 0.01 # Pulse rise time
fall_time = 0.01 # Pulse fall time
period = 0.2 # Pulse period
frequency = 5 # Pulse frequency
num_pulses = 10 # Number of pulses

# Create a time array
time = np.linspace(0, num_pulses * period, 1000)

# Create a pulse array
pulse = np.zeros_like(time)
for i in range(num_pulses):
    start = i * period # Start time of the pulse
    end = start + width # End time of the pulse
    rise_start = start - rise_time / 2 # Start time of the rise edge
    rise_end = start + rise_time / 2 # End time of the rise edge
    fall_start = end - fall_time / 2 # Start time of the fall edge
    fall_end = end + fall_time / 2 # End time of the fall edge
    pulse[(time >= rise_start) & (time < rise_end)] = amplitude * (time[(time >= rise_start) & (time < rise_end)] - rise_start) / rise_time # Rise edge
    pulse[(time >= rise_end) & (time < fall_start)] = amplitude # Constant amplitude
    pulse[(time >= fall_start) & (time < fall_end)] = amplitude * (1 - (time[(time >= fall_start) & (time < fall_end)] - fall_start) / fall_time) # Fall edge

# Plot the pulse data
plt.figure(figsize=(10, 6))
plt.plot(time, pulse)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Pulse Data")
plt.grid()
plt.show()