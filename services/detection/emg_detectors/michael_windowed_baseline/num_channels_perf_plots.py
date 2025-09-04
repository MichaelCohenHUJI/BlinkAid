import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Weighted F1 results (collapsed duplicates to keep one score per (channel, pca))
results = {
    # 2 symmetric channels
    (2, 2): 0.96, (2, 1): 0.78,
    # 2 asymmetric channels
    (1.5, 2): 0.87, (1.5, 1): 0.84,
    # 4 channels
    (4, 1): 0.77, (4, 2): 0.96, (4, 3): 0.97,
    # baseline
    (16, 3): 0.95
}

baseline_score = results[(16, 3)]

# Plot setup
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

channel_configs = [
    (2, '2 Channels (Symmetric)', [1, 2]),
    (4, '4 Channels', [1, 2, 3]),
    (1.5, '2 Channels (Asymmetric)', [1, 2])
]

bar_width = 0.6  # consistent bar width

for ax, (ch_key, title, pca_list) in zip(axes, channel_configs):
    scores = [results.get((ch_key, pca), 0) for pca in pca_list]
    x = range(len(pca_list))

    bars = ax.bar(x, scores, width=bar_width, color='skyblue', edgecolor='black')

    # Annotate bar values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.2f}",
                ha='center', va='bottom', fontsize=10)

    # Add baseline
    ax.axhline(y=baseline_score, color='red', linestyle='--', linewidth=1)
    ax.text(len(pca_list) - 0.5, baseline_score + 0.01, f"Baseline: {baseline_score:.2f}",
            color='red', fontsize=9, ha='right', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pca_list])
    ax.set_title(title)
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Weighted Avg F1-Score')
    ax.set_ylim(0, 1)

plt.suptitle('Weighted Avg F1-Score vs PCA Components\nBaseline: 16ch, 3PCs = 0.95', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
