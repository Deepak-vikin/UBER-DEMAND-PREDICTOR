import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for premium look
plt.style.use('dark_background')
sns.set_context("talk")

def plot_scatter():
    file_path = 'taxi.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Define features
    features = ['Priceperweek', 'Population', 'Monthlyincome', 'Averageparkingpermonth']
    target = 'Numberofweeklyriders'
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scatter Plots: Features vs Weekly Riders', fontsize=24, color='#FDB813') # Taxi Yellow title
    axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C'] # Vibrant heatmap colors

    for i, col in enumerate(features):
        ax = axes[i]
        # Scatter plot
        sns.scatterplot(data=df, x=col, y=target, ax=ax, s=100, color=colors[i], alpha=0.8, edgecolor='w')
        
        # Styling
        ax.set_title(f'{col} vs Riders', fontsize=16, color='white')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Weekly Riders', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == "__main__":
    plot_scatter()
