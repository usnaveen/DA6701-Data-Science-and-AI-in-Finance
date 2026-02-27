import seaborn as sns
import matplotlib.pyplot as plt

def plot_covariance_heatmap(S, title, filename):
    plt.figure(figsize=(10,8))
    sns.heatmap(S, cmap="coolwarm")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()