"""
lattice_visualization.py
--------------------------------
All visualization helpers for the Lattice class.  Nothing here modifies
Lattice state — functions only read from the instance they receive.

Registered plot names (callable via ``lattice.plot(kind=...)``)
---------------------------------------------------------------
  "lattice_vectors"   - image + lattice vectors/grid (after define_lattice_vectors)
  "atoms"             - image + detected/refined atom overlays (after add_atoms / refine_atoms)
  "polarization"      - image + polarization vectors, color wheel and optional
                        reference-neighbour legend (after measure_polarization)
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from quantem.core.visualization import show_2d

# ---------------------------------------------------------------------------
# Registry used by Lattice.plot()
# ---------------------------------------------------------------------------
PLOT_REGISTRY: dict[str, callable] = {}  # type:ignore


def _register(name: str):
    def decorator(fn):
        PLOT_REGISTRY[name] = fn
        return fn

    return decorator


### --- Plotting Functions ---
@_register("lattice_vectors")
def plot_lattice_vectors(
    lattice, *, returnfig: bool = False, bound_num_vectors: int | None = None, **kwargs
) -> None | tuple:
    """
    Overlay fitted lattice vectors and grid lines on the image.
    Call after define_lattice_vectors().

    Parameters
    ----------
    bound_num_vectors : int | None
        Number of lattice vectors to draw in each direction.
        None clips lines to image edges.
    **kwargs forwarded to show_2d (e.g. cmap, title).
    """
    if not hasattr(lattice, "_lat"):
        raise ValueError(
            "Must define lattice vectors first. Call `Lattice.define_lattice_vectors()`"
        )
    if "figsize" not in kwargs:
        kwargs["figsize"] = (10, 10)
    fig, ax = show_2d(lattice._image.array, returnfig=True, **kwargs)
    if ax.images:
        ax.images[-1].set_zorder(0)
    H, W = lattice._image.shape
    r0, u, v = (np.asarray(x, dtype=float) for x in lattice._lat)

    ax.scatter(
        r0[1], r0[0], s=60, edgecolor=(0, 0, 0), facecolor=(0, 0.5, 0), marker="s", zorder=5
    )

    n_vec = int(np.ceil(bound_num_vectors)) if bound_num_vectors is not None else 1
    for k in range(1, n_vec + 1):
        tip = r0 + k * u
        ax.arrow(
            r0[1],
            r0[0],
            (tip - r0)[1],
            (tip - r0)[0],
            length_includes_head=True,
            head_width=4.0,
            head_length=6.0,
            linewidth=2.0,
            color="red",
            zorder=4,
        )
    for k in range(1, n_vec + 1):
        tip = r0 + k * v
        ax.arrow(
            r0[1],
            r0[0],
            (tip - r0)[1],
            (tip - r0)[0],
            length_includes_head=True,
            head_width=4.0,
            head_length=6.0,
            linewidth=2.0,
            color=(0.0, 0.7, 1.0),
            zorder=4,
        )

    if bound_num_vectors is None:
        corners = np.array([[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]])
        x_lo, x_hi, y_lo, y_hi = 0.0, float(H), 0.0, float(W)
    else:
        n = float(bound_num_vectors)
        corners = np.array([r0 - n * u, r0 - n * v, r0 + n * u, r0 + n * v], dtype=float)
        x_lo = float(np.min(corners[:, 0]))
        x_hi = float(np.max(corners[:, 0]))
        y_lo = float(np.min(corners[:, 1]))
        y_hi = float(np.max(corners[:, 1]))

    A = np.column_stack((u, v))
    ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
    a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
    b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

    def clipped_segment(base, direction):
        x0, y0 = base
        dx, dy = direction
        t0, t1 = -np.inf, np.inf
        eps = 1e-12
        if abs(dx) < eps:
            if not (x_lo <= x0 <= x_hi):
                return None
        else:
            ts = sorted([(x_lo - x0) / dx, (x_hi - x0) / dx])
            t0, t1 = max(t0, ts[0]), min(t1, ts[1])
        if abs(dy) < eps:
            if not (y_lo <= y0 <= y_hi):
                return None
        else:
            ts = sorted([(y_lo - y0) / dy, (y_hi - y0) / dy])
            t0, t1 = max(t0, ts[0]), min(t1, ts[1])
        if t0 > t1:
            return None
        return base + t0 * direction, base + t1 * direction

    for a in range(a_min, a_max + 1):
        seg = clipped_segment(r0 + a * u, v)
        if seg:
            ax.plot(
                [seg[0][1], seg[1][1]],
                [seg[0][0], seg[1][0]],
                color=(0.0, 0.7, 1.0),
                lw=1,
                zorder=10,
            )
    for b in range(b_min, b_max + 1):
        seg = clipped_segment(r0 + b * v, u)
        if seg:
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="red", lw=1, zorder=10)

    ax.set_xlim(y_lo, y_hi)
    ax.set_ylim(x_hi, x_lo)
    if returnfig:
        return fig, ax
    else:
        return None


@_register("atoms")
def plot_atoms(lattice, *, returnfig: bool = False, **kwargs) -> None | tuple:
    """
    Overlay detected (and optionally refined) atom positions on the image.
    Call after add_atoms() or refine_atoms().

    Parameters
    ----------
    returnfig : bool, default False
        If True, return (fig, ax) instead of displaying.
    **kwargs forwarded to show_2d (e.g. cmap, title, figsize).
    """
    if not hasattr(lattice, "atoms"):
        raise ValueError("No atoms to plot. Call `Lattice.add_atoms()` first.")
    if "figsize" not in kwargs:
        kwargs["figsize"] = (10, 10)

    fig, ax = show_2d(lattice._image.array, returnfig=True, **kwargs)
    if ax.images:
        ax.images[-1].set_zorder(0)

    H, W = lattice._image.shape

    for s in range(lattice._num_sites):
        cell = lattice.atoms[s].array
        if isinstance(cell, list) or cell is None or cell.size == 0:
            continue
        x = lattice.atoms[s].select_fields("x").array[:, 0]
        y = lattice.atoms[s].select_fields("y").array[:, 0]
        rgb = site_colors(int(lattice._numbers[s]))
        ax.scatter(
            y,
            x,
            s=18,
            facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
            edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
            linewidths=0.75,
            marker="o",
            zorder=18,
        )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    if returnfig:
        return fig, ax
    else:
        return None


@_register("polarization")
def plot_polarization(
    lattice,
    *,
    returnfig: bool = False,
    show_legend: bool = False,
    figax: tuple | None = None,
    length_scale: float = 1.0,
    show_image: bool = True,
    subtract_median: bool = False,
    linewidth: float = 1.0,
    tail_width: float = 1.0,
    headwidth: float = 4.0,
    headlength: float = 4.0,
    outline: bool = True,
    outline_width: float = 2.0,
    outline_color: str = "black",
    alpha: float = 1.0,
    show_ref_points: bool = False,
    ref_marker: str = "o",
    ref_size: float = 20.0,
    ref_edge: str = "k",
    ref_face: str = "none",
    chroma_boost: float = 2.0,
    use_magnitude_lightness: bool = True,
    show_colorbar: bool = True,
    disp_color_max: float | None = None,
    phase_offset_deg: float = 180.0,
    phase_dir_flip: bool = False,
    adaptive_head: bool = True,
    max_head_ratio: float = 0.5,
    width_ratios: tuple[float, float] = (8, 2),
    height_ratios: tuple[float, float] = (1, 1),
    wspace: float = 0.1,
    hspace: float = 0.0,
    legend_kwargs: dict | None = None,
    **kwargs,
) -> None | tuple:
    """
    Overlay polarization vectors of the measured site on the image.
    Call after measure_polarization().

    Arrows start at each measured atom and point along its displacement from the expected
    position. Arrow color encodes direction (hue) and magnitude (lightness), as shown in the
    color wheel legend.

    Parameters
    ----------
    returnfig : bool, default False
        If True, return (fig, ax) instead of displaying.
        With show_legend=True, ax is [ax_main, ax_color_wheel, ax_legend].
    show_legend : bool, default False
        If True, lay out the figure with the vector map on the left, and the color wheel and the
        reference-neighbour legend stacked on the right. Cannot be combined with figax.
    figax : tuple | None
        (fig, ax) to draw the vector map on. If None, a new figure is created.
    length_scale : float, default 1.0
        Multiplier applied to the pixel displacements when drawing arrows.
    show_image : bool, default True
        If True, draw the lattice image behind the arrows.
    subtract_median : bool, default False
        If True, subtract the median displacement before drawing and coloring.
    linewidth, tail_width, headwidth, headlength : float
        Arrow edge width, tail width, and maximum head width / length (in points).
    outline, outline_width, outline_color
        Draw a stroke of outline_width in outline_color around each arrow.
    alpha : float, default 1.0
        Arrow transparency.
    show_ref_points : bool, default False
        If True, mark the expected (reference) positions with ref_marker, ref_size,
        ref_edge and ref_face.
    chroma_boost : float, default 2.0
        Color saturation boost for arrows and the color wheel.
    use_magnitude_lightness : bool, default True
        If True, lightness scales with displacement magnitude. Otherwise lightness is constant.
    show_colorbar : bool, default True
        If True, draw the color wheel legend.
    disp_color_max : float | None
        Displacement (in pixels) at which the lightness saturates.
        If None, the 95th percentile of non-zero displacements is used.
    phase_offset_deg : float, default 180.0
        Hue rotation in degrees (the default makes down red).
    phase_dir_flip : bool, default False
        If True, reverse the direction of the hue mapping.
    adaptive_head : bool, default True
        If True, shrink arrow heads to at most max_head_ratio of the arrow length.
    max_head_ratio : float, default 0.5
    width_ratios, height_ratios, wspace, hspace
        GridSpec layout parameters used when show_legend=True.
    legend_kwargs : dict | None
        Styling forwarded to the reference-neighbour legend: atom_size, linewidth,
        measured_color, reference_color, other_color, alpha.
    **kwargs forwarded to show_2d (e.g. cmap, title, figsize).
    """
    import matplotlib.patheffects as pe
    from matplotlib.patches import ArrowStyle, FancyArrowPatch

    from quantem.core.visualization.visualization_utils import array_to_rgba

    if not hasattr(lattice, "polarization"):
        raise ValueError("No polarization to plot. Call `Lattice.measure_polarization()` first.")

    figsize = kwargs.pop("figsize", (12, 10) if show_legend else (10, 10))

    if show_legend:
        if figax is not None:
            raise ValueError("figax cannot be used with show_legend=True.")
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2,
            2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=wspace,
            hspace=hspace,
        )
        figax = (fig, fig.add_subplot(gs[:, 0]))
        ax_wheel = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, 1])

    # Background
    if show_image:
        fig, ax = show_2d(
            lattice._image.array, returnfig=True, figax=figax, figsize=figsize, **kwargs
        )
        if ax.images:
            ax.images[-1].set_zorder(0)
    elif figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    H, W = lattice._image.shape
    pol = lattice.polarization[0]
    has_data = pol.array.size > 0

    if has_data:
        x = pol.select_fields("x").array[:, 0]
        y = pol.select_fields("y").array[:, 0]
        da = pol.select_fields("da").array[:, 0]
        db = pol.select_fields("db").array[:, 0]

        # Fractional polarization -> pixel displacements (rows, cols)
        _, u, v = (np.asarray(vec, dtype=float) for vec in lattice._lat)
        dr_raw, dc_raw = np.column_stack((u, v)) @ np.vstack((da, db))

        dr, dc, amp, disp_cap_px = _compute_polar_color_mapping(
            dr_raw,
            dc_raw,
            subtract_median=subtract_median,
            use_magnitude_lightness=use_magnitude_lightness,
            disp_color_max=disp_color_max,
        )

        # Angle mapping consistent with the color wheel (down=0°, right=+90°, up=180°, left=-90°)
        ang = np.arctan2(dc, dr)
        if phase_dir_flip:
            ang = -ang
        ang += np.deg2rad(phase_offset_deg)
        # array_to_rgba expects 2D inputs, so colors are computed as a (1, N) row
        colors = array_to_rgba(amp[None, :], ang[None, :], chroma_boost=chroma_boost)
        colors = colors.reshape(-1, 4)[:, :3]

        # Arrow head sizes: optionally capped at max_head_ratio of the arrow length
        if adaptive_head:
            arrow_lengths = np.hypot(dr, dc) * length_scale
            head_lengths = np.maximum(np.minimum(headlength, arrow_lengths * max_head_ratio), 0.5)
            head_widths = np.maximum(np.minimum(headwidth, arrow_lengths * max_head_ratio), 0.5)
        else:
            head_lengths = np.full(x.size, headlength)
            head_widths = np.full(x.size, headwidth)

        for i in range(x.size):
            x0, y0 = float(x[i]), float(y[i])
            x1 = x0 + float(dr[i]) * length_scale
            y1 = y0 + float(dc[i]) * length_scale

            arrow = FancyArrowPatch(
                (y0, x0),
                (y1, x1),
                arrowstyle=ArrowStyle.Simple(
                    head_length=float(head_lengths[i]),
                    head_width=float(head_widths[i]),
                    tail_width=tail_width,
                ),
                mutation_scale=1.0,
                linewidth=linewidth,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=alpha,
                zorder=11,
                capstyle="round",
                joinstyle="round",
                shrinkA=0.0,
                shrinkB=0.0,
            )
            if outline:
                arrow.set_path_effects(
                    [
                        pe.Stroke(linewidth=linewidth + outline_width, foreground=outline_color),
                        pe.Normal(),
                    ]
                )
            ax.add_patch(arrow)

        if show_ref_points:
            ax.scatter(
                y - dc_raw,
                x - dr_raw,
                s=ref_size,
                marker=ref_marker,
                facecolors=ref_face,
                edgecolors=ref_edge,
                linewidths=1.0,
                zorder=12,
            )

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    if "title" not in kwargs:
        ax.set_title("polarization" + (" (median subtracted)" if subtract_median else ""))

    if show_legend:
        if has_data and show_colorbar:
            _draw_polarization_colorwheel(
                ax_wheel,
                disp_cap_px,
                chroma_boost=chroma_boost,
                phase_offset_deg=phase_offset_deg,
                phase_dir_flip=phase_dir_flip,
            )
        else:
            ax_wheel.axis("off")
        if has_data and lattice._most_common_neighbours.size > 0:
            _plot_polarization_legend(lattice, figax=(fig, ax_legend), **(legend_kwargs or {}))
        else:
            ax_legend.axis("off")
        ax = [ax, ax_wheel, ax_legend]
    elif has_data and show_colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax_wheel = make_axes_locatable(ax).append_axes("right", size="28%", pad="6%")
        _draw_polarization_colorwheel(
            ax_wheel,
            disp_cap_px,
            chroma_boost=chroma_boost,
            phase_offset_deg=phase_offset_deg,
            phase_dir_flip=phase_dir_flip,
        )

    if returnfig:
        return fig, ax
    else:
        return None


# --- Plotting Helper Functions ---
def site_colors(number):
    """
    Map an integer 'number' to an RGB triple in [0,1].
    If 'number' is a list, array, or tuple, returns an array of RGB triples.
    Starts with the requested seed palette and cycles thereafter.
    """
    palette = [
        (1.00, 0.00, 0.00),  # 0: red
        (0.00, 0.70, 1.00),  # 1: lighter blue
        (0.00, 0.70, 0.00),  # 2: green with lower perceptual brightness
        (1.00, 0.00, 1.00),  # 3: magenta
        (1.00, 0.70, 0.00),  # 4: orange
        (0.00, 0.00, 1.00),  # 5: full blue
        (0.60, 0.20, 0.80),
        (0.30, 0.75, 0.75),
        (0.80, 0.40, 0.00),
        (0.20, 0.60, 0.20),
        (0.70, 0.70, 0.00),
        (1.00, 1.00, 1.00),  # -2: white
        (0.00, 0.00, 0.00),  # -1: black
    ]

    if isinstance(number, int):
        idx = int(number) % len(palette)
        return palette[idx]
    else:
        numbers = np.asarray(number, dtype=int)
        indices = numbers % len(palette)
        return np.array([palette[idx] for idx in indices.flat]).reshape(numbers.shape + (3,))


def _compute_polar_color_mapping(
    dr: np.ndarray,
    dc: np.ndarray,
    *,
    subtract_median: bool,
    use_magnitude_lightness: bool,
    disp_color_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (dr_adj, dc_adj, amp, disp_cap_px):
      dr_adj, dc_adj  -> components after optional median subtraction
      amp             -> [0,1] lightness (or constant if not using magnitude lightness)
      disp_cap_px     -> saturation cap (px): user value or 95th percentile
    """
    dr = np.asarray(dr, float).copy()
    dc = np.asarray(dc, float).copy()

    if subtract_median and dr.size:
        dr -= np.median(dr)
        dc -= np.median(dc)

    mag = np.hypot(dr, dc)

    if use_magnitude_lightness:
        if disp_color_max is None:
            nz = mag[mag > 0]
            disp_cap_px = float(np.percentile(nz, 95)) if nz.size else 1.0
        else:
            disp_cap_px = max(float(disp_color_max), 1e-9)
        amp = np.clip(mag / disp_cap_px, 0.0, 1.0)
    else:
        disp_cap_px = float(disp_color_max) if disp_color_max is not None else 1.0
        amp = np.full_like(mag, 0.85, dtype=float)

    return dr, dc, amp, disp_cap_px


def _draw_polarization_colorwheel(
    ax,
    disp_cap_px: float,
    *,
    chroma_boost: float = 2.0,
    phase_offset_deg: float = 180.0,
    phase_dir_flip: bool = False,
) -> None:
    """
    Draw the circular color legend for polarization vectors on ax.
    Hue encodes direction and lightness encodes magnitude, using the same mapping as
    plot_polarization. The scale arrow is labelled with disp_cap_px.
    """
    from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch

    from quantem.core.visualization.visualization_utils import array_to_rgba

    N = 256
    yy = np.linspace(-1, 1, N)
    xx = np.linspace(-1, 1, N)
    YY, XX = np.meshgrid(yy, xx, indexing="ij")
    rr = np.sqrt(XX**2 + YY**2)
    disk = rr <= 1.0

    ang_grid = np.arctan2(XX, -YY)
    if phase_dir_flip:
        ang_grid = -ang_grid
    ang_grid += np.deg2rad(phase_offset_deg)

    amp_grid = np.clip(rr, 0, 1)
    rgba_grid = array_to_rgba(amp_grid, ang_grid, chroma_boost=chroma_boost)
    rgba_grid[~disk] = 0.0

    ax.imshow(rgba_grid, origin="lower", extent=(-1, 1, -1, 1), interpolation="nearest", zorder=0)
    ax.set_aspect("equal")
    ax.axis("off")

    ring = Circle((0, 0), 0.98, facecolor="none", edgecolor="k", linewidth=1.2, zorder=3)
    ring.set_clip_on(False)
    ax.add_patch(ring)

    # Cardinal labels (down/right/up/left)
    ax.text(0.00, -1.12, "0°", ha="center", va="top", fontsize=9, color="k")
    ax.text(1.12, 0.00, "90°", ha="left", va="center", fontsize=9, color="k")
    ax.text(0.00, 1.12, "180°", ha="center", va="bottom", fontsize=9, color="k")
    ax.text(-1.12, 0.00, "270°", ha="right", va="center", fontsize=9, color="k")

    # Scale arrow along +x, label centered above midpoint (white)
    scale_len = 0.85
    arrow_scale = FancyArrowPatch(
        (0.0, 0.0),
        (scale_len, 0.0),
        arrowstyle=ArrowStyle.Simple(head_length=10.0, head_width=6.0, tail_width=2.0),
        mutation_scale=1.0,
        linewidth=1.2,
        facecolor="k",
        edgecolor="k",
        zorder=4,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    arrow_scale.set_clip_on(False)
    ax.add_patch(arrow_scale)

    ax.text(
        scale_len / 2.0,
        0.14,
        f"{disp_cap_px:.2g} px",
        ha="center",
        va="bottom",
        fontsize=9,
        color="w",
    )

    # Crosshairs & generous limits to avoid clipping
    ax.plot([0, 0], [-0.9, 0.9], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
    ax.plot([-0.9, 0.9], [0, 0], color=(0, 0, 0, 0.15), lw=0.8, zorder=2)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.35)


def _plot_polarization_legend(lattice, figax: tuple | None = None, **kwargs):
    """
    Diagram of the measured atom, the reference neighbours used to compute its polarization,
    and the other sites in the unit cell. Call after measure_polarization().

    Parameters
    ----------
    figax : tuple, optional
        (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
    **kwargs : optional
        atom_size : float
            Size of atoms (default: 100)
        linewidth : float
            Width of the border lines (default: 1.5)
        measured_color : str or tuple
            Color for measured atoms (default: red)
        reference_color : str or tuple
            Color for reference atoms (default: light blue)
        other_color : str or tuple
            Color for other atoms (default: white)
        alpha : float
            Transparency (default: 0.8)
        figsize : tuple
            Figure size if creating a new figure (default: (4, 4))
    """
    from matplotlib.patches import Rectangle

    # Extract parameters
    atom_size = kwargs.get("atom_size", 100)
    linewidth = kwargs.get("linewidth", 1.5)
    measured_color = kwargs.get("measured_color", (1.00, 0.00, 0.00))
    reference_color = kwargs.get("reference_color", (0.00, 0.70, 1.00))
    other_color = kwargs.get("other_color", (1.00, 1.00, 1.00))
    alpha = kwargs.get("alpha", 0.8)
    figsize = kwargs.get("figsize", (4, 4))

    # Get stored information
    _, u, v = (np.asarray(x, dtype=float) for x in lattice._lat)
    frac_positions = lattice._positions_frac
    measure_ind, reference_ind = lattice._pol_meas_ref_ind

    A = np.column_stack((u, v))
    corner_ind = np.array([[i, j] for i in [1.0, 0.0, -1.0] for j in [1.0, 0.0, -1.0]])

    # Get reference, measured, and other atoms
    measured_atom_ind = np.array([[0.0, 0.0]])
    reference_atom_ind = lattice._most_common_neighbours.copy()
    other_atom_ind = frac_positions[
        (np.arange(len(frac_positions)) != measure_ind)
        & (np.arange(len(frac_positions)) != reference_ind)
    ]
    if other_atom_ind.size > 0:
        other_atom_ind = (other_atom_ind[:, None, :] + corner_ind[None, :, :]).reshape(-1, 2)

    # Tile to get all sites within the distance of the farthest reference neighbour
    max_dist_ref_ind = reference_atom_ind[
        np.argmax(np.linalg.norm(reference_atom_ind @ A.T, axis=1))
    ]
    max_dist = np.linalg.norm(max_dist_ref_ind @ A.T)
    for _ in range(int(np.ceil(np.max(np.abs(max_dist_ref_ind))))):
        measured_atom_ind = (measured_atom_ind[:, None, :] + corner_ind[None, :, :]).reshape(-1, 2)
        if other_atom_ind.size > 0:
            other_atom_ind = (other_atom_ind[:, None, :] + corner_ind[None, :, :]).reshape(-1, 2)
    measured_atom_ind = measured_atom_ind[
        np.where(np.linalg.norm(measured_atom_ind @ A.T, axis=1) < max_dist)
    ]
    other_atom_ind = other_atom_ind[
        np.where(np.linalg.norm(other_atom_ind @ A.T, axis=1) < max_dist)
    ]
    # Same-site measurement: tiled measured atoms coincide with the reference neighbours
    if measure_ind == reference_ind:
        is_ref = np.any(
            np.all(
                np.isclose(measured_atom_ind[:, None, :], reference_atom_ind[None, :, :]), axis=2
            ),
            axis=1,
        )
        measured_atom_ind = measured_atom_ind[~is_ref]

    # Convert to Cartesian coordinates
    reference_atom_pos = reference_atom_ind @ A.T
    measured_atom_pos = measured_atom_ind @ A.T
    other_atom_pos = other_atom_ind @ A.T

    # Create figure
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the three sets of positions using plot_atoms_2d
    if len(other_atom_pos) > 0:
        fig, ax = plot_atoms_2d(
            other_atom_pos,
            site_number=-1,
            figax=(fig, ax),
            size=atom_size,
            alpha=alpha * 0.5,
            zorder=1,
            color_override=other_color,
        )

    fig, ax = plot_atoms_2d(
        reference_atom_pos,
        site_number=1,
        figax=(fig, ax),
        size=atom_size,
        alpha=alpha,
        zorder=2,
        color_override=reference_color,
    )

    fig, ax = plot_atoms_2d(
        measured_atom_pos,
        site_number=0,
        figax=(fig, ax),
        size=atom_size,
        alpha=alpha,
        zorder=3,
        color_override=measured_color,
    )

    # Dashed lines from each reference neighbour to the measured atom
    for ref_atom in reference_atom_pos:
        ax.plot(
            [ref_atom[1], 0],
            [ref_atom[0], 0],
            linestyle="--",
            linewidth=2,
            color="black",
            zorder=0,
        )

    # Formatting
    ax.set_aspect("equal")
    ax.invert_xaxis()

    # Draw rectangle border
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    rect = Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        linewidth=linewidth,
        edgecolor="black",
        facecolor="none",
        zorder=100,
    )
    ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Atom Positions", fontsize=14, fontweight="bold")

    return fig, ax


def plot_atoms_2d(
    coords,
    site_number,
    figax=None,
    size=150,
    zorder=5,
    alpha=1.0,
    coords_in_xy: bool = False,
    **kwargs,
):
    """
    2D version of plot_atoms that can be called multiple times on the same figure.

    Parameters:
    -----------
    coords : array
        Atom coordinates as N x 2 array (row, col positions)
    site_number : int or array
        Site number(s) to pass to site_colors() for coloring.
        If int, all atoms use the same color.
        If array, must have length N (one color per atom).
    figax : tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        Existing (figure, axes) tuple to plot on. If None, creates new figure and axes.
    size : float, optional
        Base size for atoms. All layer sizes are scaled by this factor. Default is 150.
    zorder : int, optional
        Drawing order for layering. Higher values draw on top. Default is 5.
    alpha : float, optional
        Transparency of markers. Default is 1.0.
    coords_in_xy : bool, optional
        If True, input coords are in (x,y) format;
        If False, input coords are in (row,col) format. Default is False.
    **kwargs : optional keyword arguments
        color_override : tuple or array, optional
            If provided, overrides site_colors. Can be:
            - Single RGB tuple (r, g, b) to apply to all atoms
            - Array of shape (3,) for single color
            - Array of shape (N, 3) for per-atom colors
        bg_color : array-like, default (1.0, 1.0, 1.0)
            Background color for depth cueing as RGB tuple
        bg_power_law : float, default 1.5
            Power law exponent for depth cueing falloff
        bg_scale : float, default 0.15
            Scale factor for depth cueing effect (0 = no effect, 1 = full effect)
        cam_pos : array-like, default (0.0, 0.0, 1000.0)
            Camera position for depth calculation
        layer_offsets : array-like, default [[0.00, 0.0], [0.05, 0.0], ...]
            XY offsets for each layer (first 2 columns of data array)
        layer_shading : array-like, default [0.00, 0.25, 0.50, 0.75, 1.00, 0.00]
            Shading values for each layer
        layer_tinting : array-like, default [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            Tinting values for each layer
        layer_sizes : array-like, default [100, 80, 60, 40, 20, 4]
            Relative marker sizes for each layer (scaled by 'size' parameter)
        layer_linewidths : array-like, default [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            Edge linewidths for each layer
        tint_color : array-like, default (1.0, 1.0, 1.0)
            Color used for tinting (white highlights)
        edge_color : tuple, default (0, 0, 0)
            Color of marker edges
        figsize : tuple, default (8, 8)
            Figure size if creating new figure

    Returns:
    --------
    fig, ax : tuple
        The figure and axes objects for reuse
    """
    # Convert to numpy array and ensure correct shape
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be an N x 2 array, got shape {coords.shape}")

    num_atoms = coords.shape[0]

    # Extract customizable parameters from kwargs with defaults
    bg_color = np.array(kwargs.get("bg_color", (1.0, 1.0, 1.0)))
    bg_power_law = kwargs.get("bg_power_law", 1.5)
    bg_scale = kwargs.get("bg_scale", 0.15)
    cam_pos = np.array(kwargs.get("cam_pos", (0.0, 0.0, 1000.0)))

    # Layer appearance parameters
    layer_offsets = kwargs.get(
        "layer_offsets",
        [
            [0.00, 0.0],
            [0.05, 0.0],
            [0.10, 0.0],
            [0.15, 0.0],
            [0.20, 0.0],
            [0.25, 0.0],
        ],
    )
    layer_shading = kwargs.get("layer_shading", [0.00, 0.25, 0.50, 0.75, 1.00, 0.00])
    layer_tinting = kwargs.get("layer_tinting", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    layer_sizes = kwargs.get("layer_sizes", [100, 80, 60, 40, 20, 4])
    layer_linewidths = kwargs.get("layer_linewidths", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    tint_color = np.array(kwargs.get("tint_color", (1.0, 1.0, 1.0)))
    edge_color = kwargs.get("edge_color", (0, 0, 0))
    figsize = kwargs.get("figsize", (8, 8))

    # Check for color override
    color_override = kwargs.get("color_override", None)

    # Build data array from layer parameters
    num_layers = len(layer_sizes)
    data = np.zeros((num_layers, 7))

    for i in range(num_layers):
        data[i, 0:2] = layer_offsets[i] if i < len(layer_offsets) else [0.0, 0.0]
        data[i, 2] = 0.0  # dz (always 0 for 2D)
        data[i, 3] = layer_shading[i] if i < len(layer_shading) else 0.0
        data[i, 4] = layer_tinting[i] if i < len(layer_tinting) else 0.0
        # Scale layer sizes by base_size (now using 'size' parameter)
        data[i, 5] = (layer_sizes[i] if i < len(layer_sizes) else 100) * (size / 100.0)
        data[i, 6] = layer_linewidths[i] if i < len(layer_linewidths) else 0.0

    # atoms_rgb_size stores: [x, y, z, r, g, b, size, linewidth]
    atoms_rgb_size = np.zeros((8, num_atoms * data.shape[0]))

    # Get colors - use override if provided, otherwise use site_colors
    if color_override is not None:
        color_override = np.asarray(color_override)

        # Handle different shapes of color_override
        if color_override.ndim == 1 and len(color_override) == 3:
            # Single RGB color for all atoms
            base_colors = np.tile(color_override[:, None], (1, num_atoms))
        elif color_override.shape == (num_atoms, 3):
            # Per-atom colors
            base_colors = color_override.T  # Shape: (3, num_atoms)
        elif color_override.shape == (3, num_atoms):
            # Already in correct shape
            base_colors = color_override
        else:
            raise ValueError(
                f"color_override must be shape (3,), (num_atoms, 3), or (3, num_atoms), got {color_override.shape}"
            )
    else:
        # Use site_colors function
        base_colors = site_colors(site_number)

        # If site_number is a single value, expand to match all atoms
        if isinstance(site_number, (int, np.integer)) or (
            isinstance(site_number, np.ndarray) and site_number.ndim == 0
        ):
            base_colors = np.tile(np.array(base_colors)[:, None], (1, num_atoms))
        else:
            # site_number is an array
            site_number = np.asarray(site_number)
            if len(site_number) != num_atoms:
                raise ValueError(
                    f"site_number array length ({len(site_number)}) must match number of atoms ({num_atoms})"
                )
            base_colors = base_colors.T  # Shape: (3, num_atoms)

    for a0 in range(data.shape[0]):
        inds = np.arange(num_atoms) + a0 * num_atoms

        # Set x, y coordinates (with offset from data)
        atoms_rgb_size[0, inds] = coords[:, 0] + data[a0, 0]
        atoms_rgb_size[1, inds] = coords[:, 1] + data[a0, 1]
        atoms_rgb_size[2, inds] = 0.0 + data[a0, 2]  # z = 0 for 2D

        atoms_rgb_size[6, inds] = data[a0, 5]  # size
        atoms_rgb_size[7, inds] = data[a0, 6]  # linewidth

        # Coloring logic using base_colors (either from site_colors or color_override)
        c = base_colors * data[a0, 3] + tint_color[:, None] * data[a0, 4]
        atoms_rgb_size[3:6, inds] = c

    # Apply depth cueing
    dist = np.sqrt(np.sum((atoms_rgb_size[0:3, :] - cam_pos[:, None]) ** 2, axis=0))
    dist -= np.min(dist)
    if np.max(dist) > 0:  # Avoid division by zero
        dist /= np.max(dist)  # scale to be 0 to 1
    dist **= bg_power_law
    dist *= bg_scale
    atoms_rgb_size[3:6, :] = atoms_rgb_size[3:6, :] * (1 - dist) + bg_color[:, None] * dist

    # Handle figure and axes
    if figax is None:
        # Create new figure and axes
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # Use provided figure and axes
        fig, ax = figax

    # Atomic sites - 2D scatter plot
    if coords_in_xy:
        ax.scatter(
            atoms_rgb_size[0, :],
            atoms_rgb_size[1, :],
            c=atoms_rgb_size[3:6, :].T,
            s=atoms_rgb_size[6, :],
            linewidth=atoms_rgb_size[7, :],
            edgecolor=edge_color,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        ax.scatter(
            atoms_rgb_size[1, :],
            atoms_rgb_size[0, :],
            c=atoms_rgb_size[3:6, :].T,
            s=atoms_rgb_size[6, :],
            linewidth=atoms_rgb_size[7, :],
            edgecolor=edge_color,
            alpha=alpha,
            zorder=zorder,
        )

    # Plot appearance
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax
