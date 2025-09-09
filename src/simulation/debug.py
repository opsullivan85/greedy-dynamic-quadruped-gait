import numpy as np
import matplotlib.pyplot as plt
from nptyping import Float32, NDArray, Shape

def view_footstep_cost_map(cost_map: NDArray[Shape["4, H, W"], Float32], selected_idx: tuple|None = None) -> None:
    """Visualize the footstep cost map.

    Args:
        cost_map (NDArray[Shape["4, H, W"], Float32]): The footstep cost map to visualize.
        selected_idx (tuple): The (foot, row, col) index of the selected footstep in the cost map.
            if present, will highlight the selected footstep on the heatmap.
    """
    plt.clf()
    
    # Compute global min and max for consistent colorbar
    vmin = float(np.min(cost_map))
    vmax = float(np.max(cost_map))
    
    # Store subplot references
    axes = []
    titles = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
    
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        axes.append(ax)
        ax.set_title(titles[i])
        im = ax.imshow(cost_map[i], cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax)
    
    # Add a single colorbar for all subplots
    fig = plt.gcf()
    fig.colorbar(im, ax=axes)
    
    # Plot the selected point on the correct subplot
    if selected_idx is not None:
        foot, row, col = selected_idx
        axes[foot].scatter([col], [row], color="blue", s=100, marker="x")
    
    plt.show(block=True)