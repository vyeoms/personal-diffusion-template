# Utilities for visualization and debugging. Created with some help from Claude

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
import wandb

def bucket_and_average(x, y, num_buckets=10, method='equal_width'):
    """
    Bucket x values and calculate average y values for each bucket.
    
    Parameters:
    x: list or array of x values
    y: list or array of y values (same length as x)
    num_buckets: number of buckets to create
    method: 'equal_width' for equal-width bins, 'equal_count' for equal-count bins
    
    Returns:
    bucket_centers: x-coordinates for plotting (center of each bucket)
    avg_y_values: average y value for each bucket
    bucket_edges: edges of the buckets
    """
    
    x = x.numpy()
    y = y.numpy()
    
    if method == 'equal_width':
        # Create equal-width buckets
        bucket_edges = np.linspace(x.min(), x.max(), num_buckets + 1)
    elif method == 'equal_count':
        # Create buckets with roughly equal number of points
        bucket_edges = np.percentile(x, np.linspace(0, 100, num_buckets + 1))
    
    # Assign each point to a bucket
    bucket_indices = np.digitize(x, bucket_edges) - 1
    
    # Handle edge case where points exactly equal the maximum value
    bucket_indices = np.clip(bucket_indices, 0, num_buckets - 1)
    
    # Calculate bucket centers and average y values
    bucket_centers = []
    avg_y_values = []
    std_y_values = []
    
    for i in range(num_buckets):
        # Find points in this bucket
        mask = bucket_indices == i
        
        if np.any(mask):
            # Calculate bucket center
            bucket_center = (bucket_edges[i] + bucket_edges[i + 1]) / 2
            bucket_centers.append(bucket_center)
            
            # Calculate average y value for this bucket
            avg_y = np.mean(y[mask])
            avg_y_values.append(avg_y)
            std_y_values.append(np.std(y[mask]))
        else:
            # Empty bucket - you might want to handle this differently
            bucket_center = (bucket_edges[i] + bucket_edges[i + 1]) / 2
            bucket_centers.append(bucket_center)
            avg_y_values.append(np.nan)  # or 0, depending on your preference
            std_y_values.append(np.nan)
    
    return np.array(bucket_centers), np.array(avg_y_values), np.array(std_y_values), bucket_edges

# Example usage and plotting
def plot_bucketed_data(x, y, loss_mean, loss_std, log_fn, num_buckets=10, method='equal_width', step=None):
    """
    Create and plot bucketed data.
    """
    bucket_centers, avg_y_values, std_y_values, bucket_edges = bucket_and_average(x, y, num_buckets, method)
    
    # Create the plot
    fig, ax2 = plt.subplots(1, 1)
    
    # Plot 2: Bucketed averages
    # Remove NaN values for plotting
    valid_mask = ~np.isnan(avg_y_values)
    ax2.plot(bucket_centers[valid_mask], avg_y_values[valid_mask], 'o-', linewidth=2, markersize=8, label='Bucketed averages')
    ax2.plot(bucket_centers[valid_mask], avg_y_values[valid_mask] - std_y_values[valid_mask], '--', c="red", linewidth=2, markersize=8)
    ax2.plot(bucket_centers[valid_mask], avg_y_values[valid_mask] + std_y_values[valid_mask], '--', c="red", linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('X bucket centers')
    ax2.set_ylabel('Average Y values')
    ax2.set_title(f'Bucketed Data ({num_buckets} buckets, {method})')
    ax2.grid(True, alpha=0.3)

    x = np.logspace(x.min().log10(), x.max().log10(), 1000)
    p = lognorm.pdf(x = x, scale = np.exp(loss_mean), s = loss_std)
    ax2.plot(x, p, c="orange", linewidth=2, label="Noise distribution")

    ax2.legend()
    plt.tight_layout()
    fig = ax2.get_figure()
    if log_fn is not None:
        if step is None:
            raise ValueError("step must be provided when logging")
        log_fn({ f'Buckets' : wandb.Image(fig) }, step=step)
    plt.clf()
    # plt.show()
    
    return bucket_centers, avg_y_values, bucket_edges

def spaghetti_plot_paths(paths, sigma_schedule, measured, log_fn=None, step=None):
    # Convert to numpy for plotting
    paths_np = paths.cpu().numpy()

    # Calculate mean path
    mean_path = paths_np.mean(axis=0)

    # Create the plot
    # plt.figure(figsize=(10, 6))

    # Plot all individual paths in grey with transparency
    for i in range(paths.shape[0]):
        plt.plot(sigma_schedule, paths_np[i, :], color='grey', alpha=0.3, linewidth=0.5)

    # Plot the mean path in bold color
    plt.plot(sigma_schedule, mean_path, color='red', linewidth=3, label='Batch mean')

    plt.xlabel('sigma')
    plt.ylabel('Value')
    # plt.title('Spaghetti Plot: Individual Paths and Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    fig = plt.gca().get_figure()
    if log_fn is not None:
        if step is None:
            raise ValueError("step must be provided when logging")
        log_fn({ f'{measured} over sampling' : wandb.Image(fig) }, step=step)
    plt.close(fig)
