from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# File paths
log_file1 = "summary/events.out.tfevents.1733523218.sumukh"
log_file2 = "summary/events.out.tfevents.1733613946.sumukh"

# Function to extract scalar data
def get_scalar_data(log_file, tag):
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()  # Load events
    if tag in ea.Tags()['scalars']:
        return [(scalar_event.step, scalar_event.value) for scalar_event in ea.Scalars(tag)]
    else:
        raise ValueError(f"Tag '{tag}' not found in {log_file}")

# Extract data
tag = 'avg_ep_rews/train'
data1 = get_scalar_data(log_file1, tag)
data2 = get_scalar_data(log_file2, tag)

# Offset steps in the second file
offset = data1[-1][0] + 1 if data1 else 0  # Last step of file1 + 1
data2 = [(step + offset, value) for step, value in data2]

# Combine data
combined_steps = [x[0] for x in data1] + [x[0] for x in data2]
combined_values = [x[1] for x in data1] + [x[1] for x in data2]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(combined_steps, combined_values, label='avg_ep_rews/train')
plt.ylabel('Reward', fontsize=24)
plt.xlabel('Iterations', fontsize=24)
plt.title('Reward vs Iterations', fontsize=28)

# Increase font size of tick labels
plt.tick_params(axis='both', labelsize=24)

# plt.legend()
plt.grid(False)
plt.show()
