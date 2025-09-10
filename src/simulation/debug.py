from ast import Raise
import numpy as np
import matplotlib.pyplot as plt
from nptyping import Float32, NDArray, Shape


def view_footstep_cost_map(
    cost_map: NDArray[Shape["4, H, W"], Float32],
    selected_idx: tuple | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Visualize the footstep cost map.

    Args:
        cost_map (NDArray[Shape["4, H, W"], Float32]): The footstep cost map to visualize.
            Input expects leg ordering: [FR, FL, RR, RL] (Front Right, Front Left, Rear Right, Rear Left)
            Display shows in FR FL RR RL order with each map rotated 180 degrees.
        selected_idx (tuple): The (foot, row, col) index of the selected footstep in the cost map.
            if present, will highlight the selected footstep on the heatmap.
        title (str, optional): Title for the entire figure. Defaults to None.
        vmin (float, optional): Minimum value for color scale. Defaults to None.
        vmax (float, optional): Maximum value for color scale. Defaults to None.
    """
    plt.clf()

    # Compute global min and max for consistent colorbar
    vmin = float(np.min(cost_map)) if vmin is None else vmin
    vmax = float(np.max(cost_map)) if vmax is None else vmax

    # Reorder cost_map from [FR, FL, RR, RL] to [FR, FL, RR, RL] for display
    # This keeps the original order but rotates each map 180 degrees
    display_order = [0, 1, 2, 3]  # FR, FL, RR, RL (no reordering, just rotation)
    cost_map_display = np.array([np.flipud(cost_map[i]) for i in display_order])

    # Store subplot references
    axes = []
    titles = ["Front Right", "Front Left", "Rear Right", "Rear Left"]

    im = None
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        axes.append(ax)
        ax.set_title(titles[i])
        im = ax.imshow(
            cost_map_display[i], cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        # Overlay the flat index value for each pixel
        H, W = cost_map.shape[1:]
        for row in range(H):
            for col in range(W):
                # Compute the original flat index for this display position
                # Since we rotated 180 degrees, we need to flip the row and col
                original_row = H - 1 - row
                original_col = W - 1 - col
                original_leg_idx = display_order[i]
                flat_idx = np.ravel_multi_index((original_leg_idx, original_row, original_col), cost_map.shape)
                ax.text(
                    col,
                    row,
                    str(flat_idx),
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

    # Add a single colorbar for all subplots, using the last imshow result (im is guaranteed to be set)
    fig = plt.gcf()
    if im is None:
        raise RuntimeError("No data to display in cost map.")

    cbar = fig.colorbar(im, ax=axes)
    cbar.set_label("Action Cost (Lower is better)")

    # Plot the selected point on the correct subplot (adjust for display reordering and rotation)
    if selected_idx is not None:
        foot, row, col = selected_idx
        # Find which display position this leg maps to
        display_idx = display_order.index(foot)
        # Since we rotated 180 degrees, flip the row and col coordinates
        H, W = cost_map.shape[1:]
        rotated_row = H - 1 - row
        rotated_col = W - 1 - col
        axes[display_idx].scatter([rotated_col], [rotated_row], color="blue", s=100, marker="x")

    if title is not None:
        plt.suptitle(title)

    plt.show(block=True)
