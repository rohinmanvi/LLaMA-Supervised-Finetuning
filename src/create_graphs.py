import pandas as pd
import matplotlib.pyplot as plt

pastelBlue = "#0072B2"
pastelRed = "#F5615C"

# Load data
llama_df = pd.read_csv('llama_highway_data.csv')
planner_df = pd.read_csv('planner_highway_data.csv')

# Create a figure
fig, ax = plt.subplots(tight_layout=True)

# Plot histograms
ax.hist(llama_df['total_rewards'], bins=20, alpha=0.5, color=pastelBlue, label='LLaMA')
ax.hist(planner_df['total_rewards'], bins=20, alpha=0.5, color=pastelRed, label='OPD')

# Set labels
ax.set_xlabel('Total Rewards')
ax.set_ylabel('Frequency')

# Show legend
ax.legend()

# Save the figure
plt.savefig('histogram.pdf')
