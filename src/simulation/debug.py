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
            Display shows in FL FR RL RR order for intuitive robot-centered orientation.
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

    # Reorder cost_map from [FR, FL, RR, RL] to [FL, FR, RL, RR] for display
    # This puts left legs on the left side of the visualization, right legs on the right
    display_order = [1, 0, 3, 2]  # FL, FR, RL, RR indices from original [FR, FL, RR, RL]
    cost_map_display = cost_map[display_order]

    # Store subplot references
    axes = []
    titles = ["Front Left", "Front Right", "Rear Left", "Rear Right"]

    im = None
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        axes.append(ax)
        ax.set_title(titles[i])
        im = ax.imshow(
            cost_map_display[i, ::-1], cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        # Overlay the flat index value for each pixel
        H, W = cost_map.shape[1:]
        for row in range(H):
            for col in range(W):
                # Compute the original flat index for this display position
                original_leg_idx = display_order[i]
                flat_idx = np.ravel_multi_index((original_leg_idx, row, col), cost_map.shape)
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

    # Plot the selected point on the correct subplot (adjust for display reordering)
    if selected_idx is not None:
        foot, row, col = selected_idx
        # Find which display position this leg maps to
        display_idx = display_order.index(foot)
        axes[display_idx].scatter([col], [row], color="blue", s=100, marker="x")

    if title is not None:
        plt.suptitle(title)

    plt.show(block=True)
