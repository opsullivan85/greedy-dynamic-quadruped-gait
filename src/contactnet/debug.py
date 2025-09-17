from ast import Raise
from calendar import c
import numpy as np
import matplotlib.pyplot as plt
from nptyping import Float32, NDArray, Shape
from mpl_toolkits.axes_grid1 import AxesGrid


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
        selected_idx (tuple): The (foot, row, col) index of the selected footstep in the cost map.
            if present, will highlight the selected footstep on the heatmap.
        title (str, optional): Title for the entire figure. Defaults to None.
        vmin (float, optional): Minimum value for color scale. Defaults to None.
        vmax (float, optional): Maximum value for color scale. Defaults to None.
    """
    plt.clf()

    # Compute global min and max for consistent colorbar
    vmin = float(np.min(cost_map[~np.isnan(cost_map)])) if vmin is None else vmin
    vmax = float(np.max(cost_map[~np.isnan(cost_map)])) if vmax is None else vmax

    # Store subplot references
    axes = []
    titles = ["Front Left", "Front Right", "Rear Left", "Rear Right"]

    im = None
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        axes.append(ax)
        ax.set_title(titles[i])
        im = ax.imshow(
            # cost_map[i, ::-1, ::-1], cmap="hot", interpolation="nearest"
            cost_map[i, ::-1, ::-1], cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        # Overlay the flat index value for each pixel
        H, W = cost_map.shape[1:]
        for row in range(H):
            for col in range(W):
                # Compute the original flat index for this display position
                # Since we rotated 180 degrees, we need to flip the row and col
                original_row = H - 1 - row
                original_col = W - 1 - col
                flat_idx = np.ravel_multi_index((i, original_row, original_col), cost_map.shape)
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
        # Since we rotated 180 degrees, flip the row and col coordinates
        H, W = cost_map.shape[1:]
        rotated_row = H - 1 - row
        rotated_col = W - 1 - col
        axes[foot].scatter([rotated_col], [rotated_row], color="blue", s=100, marker="x")

    if title is not None:
        plt.suptitle(title)

    plt.show(block=True)

def view_multiple_footstep_cost_maps(
    cost_maps: list[NDArray[Shape["4, H, W"], Float32]],
    titles: list[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Visualize multiple footstep cost maps side by side.

    Args:
        cost_maps (list[NDArray[Shape["4, H, W"], Float32]]): The footstep cost maps to visualize.
        titles (list[str], optional): Titles for each subplot. Defaults to None.
        vmin (float, optional): Minimum value for color scale. Defaults to None.
        vmax (float, optional): Maximum value for color scale. Defaults to None.
    """
    M = len(cost_maps)
    if M == 0:
        return

    # Figure

    nrows_groups = 2
    ncols_groups = (M + nrows_groups - 1) // nrows_groups  # ceil(M / 2)

    f = plt.figure(figsize=(4*ncols_groups, 3*nrows_groups))

    margin = 0.05
    gap = 0.05
    plot_width = 1 - 2 * margin - (ncols_groups - 1) * gap
    plot_height = 1 - 2 * margin - (nrows_groups - 1) * gap
    width_per_group = plot_width / ncols_groups
    height_per_group = plot_height / nrows_groups

    axes_groups = []
    im = None
    for i in range(M):
        row = i // ncols_groups
        col = i % ncols_groups
        left = margin + col * (width_per_group + gap)
        bottom = margin + (nrows_groups - 1 - row) * (height_per_group + gap)
        width = width_per_group
        height = height_per_group
        ag = AxesGrid(f, [left, bottom, width, height], nrows_ncols=(2, 2), axes_pad=0.1, cbar_mode='single', cbar_location='right')
        axes_groups.append(ag)
        cost_map = cost_maps[i]
        # titles_leg = ["Front Left", "Front Right", "Rear Right", "Rear Left"]

        # Compute per cost_map min and max
        vmin_cm = float(np.min(cost_map[~np.isnan(cost_map)])) if vmin is None else vmin
        vmax_cm = float(np.max(cost_map[~np.isnan(cost_map)])) if vmax is None else vmax
        spread = vmax_cm - vmin_cm

        for j in range(4):
            ax = ag[j]
            # ax.set_title(titles_leg[j])
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(
                cost_map[j, ::-1, ::-1], cmap="hot", interpolation="nearest", vmin=vmin_cm, vmax=vmax_cm
            )

        # Add colorbar for this AxesGrid
        cbar = ag.cbar_axes[0].colorbar(im)
        # cbar.set_label("Action Cost (Lower is better)")

        # Add title for the group if provided
        if titles is not None:
            center_x = left + width / 2
            center_y = bottom + height
            f.text(center_x, center_y, f"{titles[i]} ({spread:.3f})", ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.show()
