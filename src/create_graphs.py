import pandas as pd
import matplotlib.pyplot as plt

# Load data
llama_df = pd.read_csv('llama_highway_data.csv')
planner_df = pd.read_csv('planner_highway_data.csv')

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# Plot histograms
axs[0].hist(llama_df['total_rewards'], bins=20, alpha=0.5, color='blue', label='Llama')
axs[1].hist(planner_df['total_rewards'], bins=20, alpha=0.5, color='red', label='Planner')

# Set labels
axs[0].set_xlabel('LLaMA Episode Rewards')
axs[0].set_ylabel('Frequency')
axs[1].set_xlabel('OPD Episode Rewards')

# Show legend
axs[0].legend()
axs[1].legend()

# Save the figure
plt.savefig('histogram.pdf')
