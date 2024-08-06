import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

df_path = '/path/to/results/csv'
df = pd.read_csv(df_path)

# Group by experiment and calculate mean and SEM (standard error of the mean)
grouped = df.groupby('experiment').agg({
    ' AUC': ['mean', 'std', 'count'],
    ' FNR gap': ['mean', 'std', 'count']
}).reset_index()

# Calculate SEM
grouped['auc_sem'] = grouped[(' AUC', 'std')] / np.sqrt(grouped[(' AUC', 'count')])
grouped['fnr_gap_sem'] = grouped[(' FNR gap', 'std')] / np.sqrt(grouped[(' FNR gap', 'count')])

# Flatten the MultiIndex columns
grouped.columns = ['experiment', 'auc_mean', 'auc_std', 'auc_count', 'fnr_gap_mean', 'fnr_gap_std', 'fnr_gap_count', 'auc_sem', 'fnr_gap_sem']

# List of experiments to plot
experiments_to_plot = [
    # ex: 'regular_none_0.0_none'
]

# Filter the grouped DataFrame to include only the specified experiments
filtered_grouped = grouped[grouped['experiment'].isin(experiments_to_plot)]

# Define labels
labels = {
    # ex: 'regular_none_0.0_none': 'BASELINE',
    # ex: 'regular_minority_0.25_dark_positive': 'Disease Augment x 1.25',
}

# Different markers for each experiment
markers = ['o', 's', '^', 'P', 'D', 'v', '*', '>', '<', '>', 'x']
plt.figure(figsize=(10, 6))

# Plot each experiment with a different marker
for i, experiment in enumerate(filtered_grouped['experiment']):
    label = labels.get(experiment, experiment)
    plt.errorbar(filtered_grouped['auc_mean'].iloc[i], filtered_grouped['fnr_gap_mean'].iloc[i],
                 xerr=filtered_grouped['auc_sem'].iloc[i], yerr=filtered_grouped['fnr_gap_sem'].iloc[i],
                 fmt=markers[i % len(markers)], capthick=2, label=label)

# Add labels and title
plt.xlabel('AUC')
plt.ylabel('FNR Gap')
plt.title('AUC vs FNR Gap')

# Add legend
plt.legend(title='Legend', bbox_to_anchor=(1.10, 1.15), loc='upper right')

# Save the plot
save_dir = '/path/to/save/dir'
save_path = os.path.join(save_dir, 'final_dark_positive_augment_auc_vs_fnr_gap.png')
plt.savefig(save_path)
plt.show()