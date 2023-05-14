import pandas as pd
import matplotlib.pyplot as plt

pastelBlue = "#0072B2"
pastelRed = "#F5615C"
pastelGreen = "#009E73"
pastelPurple = "#8770FE"

# Load data
llama_df = pd.read_csv('llama_highway_data.csv')
planner_df = pd.read_csv('planner_highway_data.csv')

# Create a figure
fig, ax = plt.subplots(tight_layout=True)

# Plot histograms
ax.hist(llama_df['total_rewards'], bins=20, alpha=0.5, color=pastelRed, label='LLaMA')
ax.hist(planner_df['total_rewards'], bins=20, alpha=0.5, color=pastelBlue, label='OPD')

# Set labels
ax.set_xlabel('Episode Rewards')
ax.set_ylabel('Frequency')

# Show legend
ax.legend()

# Save the figure
plt.savefig('plots/rewards_histogram.pdf')

# Create a figure for min-max comparison
fig1, ax1 = plt.subplots(tight_layout=True)

# Plot histograms
ax1.hist(llama_df['min_inference_times'], bins=20, alpha=0.5, color=pastelGreen, label='First Inference Times')
ax1.hist(llama_df['max_inference_times'], bins=20, alpha=0.5, color=pastelPurple, label='Last Inference Times')

# Set labels
ax1.set_xlabel('Inference Times')
ax1.set_ylabel('Frequency')

# Show legend
ax1.legend()

# Save the figure
plt.savefig('plots/min_max_inference_times_histogram.pdf')

# Create a figure for avg comparison
fig2, ax2 = plt.subplots(tight_layout=True)

# Plot histograms
ax2.hist(llama_df['avg_inference_times'], bins=20, alpha=0.5, color=pastelRed, label='LLaMA Average Inference Times')
ax2.hist(planner_df['avg_inference_times'], bins=20, alpha=0.5, color=pastelBlue, label='OPD Average Planning Times')

# Set labels
ax2.set_xlabel('Average Inference Times')
ax2.set_ylabel('Frequency')

# Show legend
ax2.legend()

# Save the figure
plt.savefig('plots/avg_inference_times_histogram.pdf')


