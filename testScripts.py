from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/23-2/blinks.csv')
waveform_noisy = df['channel_1']
t = df['timestamp']

# Apply Savitzky-Golay filter
waveform_smooth = savgol_filter(waveform_noisy, window_length=11, polyorder=2)

# Plot
plt.figure(figsize=(10,4))
plt.plot(t, waveform_noisy, label="Noisy Signal", alpha=0.5)
plt.plot(t, waveform_smooth, label="Savitzky-Golay Filtered", linestyle='--', color='red')
plt.legend()
plt.show()
