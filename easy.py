import numpy as np

data = np.array([115.3, 195.5, 120.5, 110.2, 90.4, 105.6, 110.9, 116.3, 122.3, 125.4])

mean_val = np.mean(data)

median_val = np.median(data)

# to calculate mode for the data
unique_val , counts = np.unique(data, return_counts=True)
max_count = np.argmax(counts)

mode_val = unique_val[max_count]

#standard dev
std_dev = np.std(data)

variance = np.var(data)

min_max_normalized = (data - np.min(data))/(np.max(data)-np.min(data))


standardized_data = (data - mean_val) / std_dev

print("Mean: ", mean_val)
print("Median: ", median_val)
print("Mode: ", mode_val)
print("Standard Deviation : ", std_dev )
print("Variance : ", variance)
print("Min Max Normalised: ", min_max_normalized)
print("Standardized: ", standardized_data)
