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
  "gmm_classification" - GMM fit in (da, db) space, or divergence histogram
                        (after calculate_order_parameter)
  "order_parameter"   - image + atoms colored by phase probability, with optional
                        per-phase reference diagrams (after calculate_order_parameter)
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


@_register("gmm_classification")
def plot_gmm_classification(
    lattice,
    *,
    returnfig: bool = False,
    figax: tuple | None = None,
    phase_colours=None,
    clip_range: tuple[float, float] | None = None,
    saturation_boost: float = 1.0,
    contour_cmap: str = "gray_r",
    contour_levels: int = 15,
    gmm_center_colour=None,
    gmm_ellipse_colour=None,
    n_std: float = 2.0,
    show_colorbar: bool = True,
    divergence_hist_bins: int = 50,
    figsize: tuple[float, float] | None = None,
) -> None | tuple:
    """
    Visualize the GMM fitted by calculate_order_parameter().

    In the 2D mode, the polarization (da, db) of every atom is drawn over its kernel density
    estimate, colored by phase probability, with the GMM centers and n_std confidence ellipses.
    In the divergence mode, a histogram of the divergence is drawn with the bars colored by
    phase and the fitted Gaussian components overlaid.

    Parameters
    ----------
    returnfig : bool, default False
        If True, return (fig, ax) instead of displaying.
    figax : tuple | None
        (fig, ax) to draw on. If None, a new figure is created.
    phase_colours : callable | color | sequence of colors | NDArray | None
        Color of each phase: a function i -> RGB, a single color for all phases, or
        num_phases colors (names or an RGB array of shape (num_phases, 3)).
        If None, site_colors is used.
    clip_range : tuple[float, float] | None
        Order parameter range mapped to full saturation (2 and 3 phases).
    saturation_boost : float, default 1.0
        Multiplier on the color saturation.
    contour_cmap : str, default "gray_r"
        Colormap of the density contours (2D mode).
    contour_levels : int, default 15
        Number of density contour levels (2D mode).
    gmm_center_colour, gmm_ellipse_colour : color | None
        Colors of the GMM centers and ellipses (2D mode). If None, chosen by num_phases.
    n_std : float, default 2.0
        Size of the confidence ellipses in standard deviations (2D mode).
    show_colorbar : bool, default True
        If True, draw the 2-phase colorbar or 3-phase color triangle (2D mode).
    divergence_hist_bins : int, default 50
        Number of histogram bins (divergence mode).
    figsize : tuple | None
        Figure size if a new figure is created.
    """
    import warnings

    from matplotlib.ticker import PercentFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.stats import gaussian_kde, norm

    _check_order_parameter(lattice)
    settings = lattice._order_parameter_settings
    num_phases = settings["num_phases"]
    probabilities = lattice._order_parameter_probabilities
    colours = _resolve_phase_colours(phase_colours, num_phases)

    if figax is not None:
        fig, ax = figax
    else:
        default_figsize = (8, 5) if settings["spatial_divergence"] else (8, 7)
        fig, ax = plt.subplots(figsize=figsize or default_figsize)

    if probabilities.shape[0] == 0:
        ax.set_title("GMM classification (no data)")
        return (fig, ax) if returnfig else None

    gmm = lattice.gmm
    means = np.asarray(lattice._polarization_means, dtype=float)
    covariances = np.asarray(gmm.covariances_, dtype=float)  # (K, D, D)

    if settings["spatial_divergence"]:
        divergence = lattice._order_parameter_divergence
        counts, bin_edges = np.histogram(divergence, bins=divergence_hist_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        # Bars are colored by the phase probability of the bin center
        bar_colors = create_colors_from_probabilities(
            gmm.predict_proba(bin_centers.reshape(-1, 1)),
            num_phases,
            colours,
            clip_range=clip_range,
            saturation_boost=saturation_boost,
        )
        ax.bar(
            bin_edges[:-1],
            counts,
            width=bin_width,
            color=bar_colors,
            align="edge",
            edgecolor="none",
            alpha=0.85,
            zorder=2,
        )

        # Fitted components, scaled from probability density to counts
        x_plot = np.linspace(bin_edges[0], bin_edges[-1], 500)
        for k in range(num_phases):
            mu_k = float(means[k, 0])
            std_k = float(np.sqrt(max(covariances[k, 0, 0], 1e-12)))
            curve = (
                float(gmm.weights_[k])
                * norm.pdf(x_plot, mu_k, std_k)
                * divergence.size
                * bin_width
            )
            ax.plot(
                x_plot,
                curve,
                color=colours[k],
                linewidth=2.5,
                label=f"Phase {k}  (μ={mu_k:.3g}, σ={std_k:.3g})",
                zorder=4,
            )
            ax.axvline(mu_k, color=colours[k], linestyle="--", linewidth=1.5, alpha=0.75, zorder=3)

        ax.set_xlabel("Divergence (px)")
        ax.set_ylabel("Count")
        ax.set_title("Local polarization divergence — GMM decomposition")
        ax.legend(loc="best")
        return (fig, ax) if returnfig else None

    if settings["spatial_average"]:
        warnings.warn(
            "Atoms are drawn at their measured polarization but colored by the spatially "
            "averaged probabilities."
        )

    pol = lattice.polarization[0]
    da = pol.select_fields("da").array[:, 0].astype(float)
    db = pol.select_fields("db").array[:, 0].astype(float)
    max_bound = float(max(np.abs(da).max(), np.abs(db).max())) or 0.01

    # Kernel density estimate of the polarization
    if da.size >= 2:
        grid = np.linspace(-max_bound, max_bound, 100)
        X, Y = np.meshgrid(grid, grid)
        try:
            Z = gaussian_kde(np.vstack((da, db)))(np.vstack((X.ravel(), Y.ravel())))
            Z = Z.reshape(X.shape)
            ax.contourf(X, Y, Z, levels=contour_levels, cmap=contour_cmap, alpha=0.9)
            ax.contour(
                X, Y, Z, levels=contour_levels, cmap=contour_cmap, linewidths=0.5, alpha=0.9
            )
        except (np.linalg.LinAlgError, ValueError):
            warnings.warn("Polarization density is degenerate. Contours are not drawn.")
    else:
        warnings.warn(f"Cannot estimate the density of {da.size} atom(s). Contours are not drawn.")

    ax.scatter(
        da,
        db,
        c=create_colors_from_probabilities(probabilities, num_phases, colours),
        alpha=0.7,
        s=20,
        edgecolors="black",
        linewidths=0.3,
        zorder=7,
    )

    # GMM centers and confidence ellipses
    if num_phases == 2:
        preset_center, preset_ellipse = (0, 0.7, 0), (0, 0.7, 0)
    elif num_phases < 5:
        preset_center, preset_ellipse = "yellow", "yellow"
    else:
        preset_center, preset_ellipse = "black", "white"
    gmm_center_colour = preset_center if gmm_center_colour is None else gmm_center_colour
    gmm_ellipse_colour = preset_ellipse if gmm_ellipse_colour is None else gmm_ellipse_colour

    ax.scatter(
        means[:, 0],
        means[:, 1],
        c=[gmm_center_colour],
        s=300,
        marker="x",
        linewidths=4,
        alpha=0.8,
        label="GMM Centers",
        zorder=10,
    )
    for k in range(num_phases):
        _plot_gaussian_ellipse(
            ax,
            means[k],
            covariances[k],
            n_std=n_std,
            edgecolor=gmm_ellipse_colour,
            linewidth=1.5,
            alpha=0.6,
            zorder=8,
        )

    ax.axhline(y=0, color="black", linewidth=1.5, alpha=0.7, zorder=1)
    ax.axvline(x=0, color="black", linewidth=1.5, alpha=0.7, zorder=1)
    ax.set_xlim(-max_bound, max_bound)
    ax.set_ylim(-max_bound, max_bound)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("da")
    ax.set_ylabel("db")
    ax.set_title("Classification & Contour Overlay")
    ax.legend(loc="best")

    if show_colorbar and num_phases in (2, 3):
        divider = make_axes_locatable(ax)
        if num_phases == 2:
            add_2phase_colorbar(
                divider.append_axes("right", size="4%", pad="4%"),
                colours,
                clip_range=clip_range,
                saturation_boost=saturation_boost,
            )
        else:
            add_3phase_color_triangle(
                divider.append_axes("right", size="35%", pad="4%"),
                colours,
                clip_range=clip_range,
                saturation_boost=saturation_boost,
            )

    return (fig, ax) if returnfig else None


@_register("order_parameter")
def plot_order_parameter(
    lattice,
    *,
    returnfig: bool = False,
    show_phase_references: bool = False,
    phase_colours=None,
    clip_range: tuple[float, float] | None = None,
    saturation_boost: float = 1.0,
    marker_size: float = 50.0,
    marker_alpha: float = 0.8,
    show_colorbar: bool = True,
    reference_kwargs: dict | None = None,
    **kwargs,
) -> None | tuple:
    """
    Overlay the atoms colored by phase probability on the image.
    Call after calculate_order_parameter().

    With 2 phases the color goes from phase 0 through white (uncertain) to phase 1, and a
    colorbar shows the order parameter p1 - p0. With 3 phases the colors are mixed and a
    color triangle is shown. With more phases the phase colors are mixed by probability.

    Parameters
    ----------
    returnfig : bool, default False
        If True, return (fig, ax) instead of displaying.
        With show_phase_references=True, ax is [ax_main, (ax_colorbar,) ax_phase_0, ...].
    show_phase_references : bool, default False
        If True, draw a unit cell diagram of the mean polarization of each phase next to the
        map. Not available in the divergence mode.
    phase_colours : callable | color | sequence of colors | NDArray | None
        Color of each phase: a function i -> RGB, a single color for all phases, or
        num_phases colors (names or an RGB array of shape (num_phases, 3)).
        If None, site_colors is used.
    clip_range : tuple[float, float] | None
        Order parameter range mapped to full saturation (2 and 3 phases).
    saturation_boost : float, default 1.0
        Multiplier on the color saturation.
    marker_size, marker_alpha : float
        Size and transparency of the atom markers.
    show_colorbar : bool, default True
        If True, draw the 2-phase colorbar or 3-phase color triangle.
    reference_kwargs : dict | None
        Styling forwarded to the phase reference diagrams: atom_size, arrow_scale_factor,
        adaptive_head, max_head_ratio, phase_arrow_headlength, phase_arrow_headwidth,
        phase_arrow_tail_width, lattice_linewidth, reference_atom_colour, other_atom_colour,
        lattice_line_colour, alpha_phase_atom, alpha_reference_atom, alpha_other_atom,
        alpha_shadow_atom, alpha_phase_arrow, alpha_lattice_lines.
    **kwargs forwarded to show_2d (e.g. cmap, title, figsize).
    """
    import warnings

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _check_order_parameter(lattice)
    settings = lattice._order_parameter_settings
    num_phases = settings["num_phases"]
    probabilities = lattice._order_parameter_probabilities
    has_data = probabilities.shape[0] > 0
    colours = _resolve_phase_colours(phase_colours, num_phases)

    if show_phase_references and settings["spatial_divergence"]:
        warnings.warn("Phase references are not available in the divergence mode.")
        show_phase_references = False

    draw_colorbar = show_colorbar and num_phases in (2, 3)
    figsize = kwargs.pop("figsize", (14, 10) if show_phase_references else (10, 10))
    show_title = "title" not in kwargs

    ax_cbar = None
    ref_axes = []
    figax = None
    if show_phase_references:
        fig = plt.figure(figsize=figsize)
        if num_phases == 2 and draw_colorbar:
            outer = gridspec.GridSpec(1, 3, figure=fig, width_ratios=(10, 0.4, 4), wspace=0.15)
            ax_cbar = fig.add_subplot(outer[0, 1])
            ref_spec = outer[0, 2]
        else:
            outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=(10, 4), wspace=0.1)
            ref_spec = outer[0, 1]
        figax = (fig, fig.add_subplot(outer[0, 0]))

        # With 3 phases the color triangle sits above the phase references
        num_rows = num_phases + (1 if num_phases == 3 and draw_colorbar else 0)
        inner = gridspec.GridSpecFromSubplotSpec(num_rows, 1, subplot_spec=ref_spec, hspace=0.25)
        ref_axes = [fig.add_subplot(inner[i, 0]) for i in range(num_rows)]
        if num_phases == 3 and draw_colorbar:
            ax_cbar = ref_axes.pop(0)

    fig, ax = show_2d(lattice._image.array, returnfig=True, figax=figax, figsize=figsize, **kwargs)
    if ax.images:
        ax.images[-1].set_zorder(0)

    if has_data:
        pol = lattice.polarization[0]
        x = pol.select_fields("x").array[:, 0]
        y = pol.select_fields("y").array[:, 0]
        ax.scatter(
            y,
            x,
            c=create_colors_from_probabilities(
                probabilities,
                num_phases,
                colours,
                clip_range=clip_range,
                saturation_boost=saturation_boost,
            ),
            s=marker_size,
            alpha=marker_alpha,
            edgecolors="black",
            linewidth=1,
            zorder=11,
        )

    H, W = lattice._image.shape
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    if show_title:
        ax.set_title("Spatial phase probability map")

    if draw_colorbar:
        if ax_cbar is None:
            size = "4%" if num_phases == 2 else "35%"
            ax_cbar = make_axes_locatable(ax).append_axes("right", size=size, pad="4%")
        if num_phases == 2:
            add_2phase_colorbar(
                ax_cbar, colours, clip_range=clip_range, saturation_boost=saturation_boost
            )
        else:
            add_3phase_color_triangle(
                ax_cbar, colours, clip_range=clip_range, saturation_boost=saturation_boost
            )

    for phase_index, ax_ref in enumerate(ref_axes):
        if has_data:
            _plot_phase_reference(
                lattice,
                (fig, ax_ref),
                phase_index,
                colours[phase_index],
                **(reference_kwargs or {}),
            )
        else:
            ax_ref.axis("off")

    if show_phase_references:
        ax = [ax] + ([ax_cbar] if ax_cbar is not None else []) + ref_axes

    return (fig, ax) if returnfig else None


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


def _check_order_parameter(lattice) -> None:
    """Raise if calculate_order_parameter() has not been run for the current polarization."""
    if not hasattr(lattice, "_order_parameter_probabilities"):
        raise ValueError(
            "No order parameter to plot. Call `Lattice.calculate_order_parameter()` first."
        )
    cell = lattice.polarization[0].array
    num_atoms = 0 if isinstance(cell, list) or cell is None else cell.shape[0]
    if lattice._order_parameter_probabilities.shape[0] != num_atoms:
        raise ValueError(
            "The polarization has changed since the order parameter was calculated. "
            "Call `Lattice.calculate_order_parameter()` again."
        )


def _resolve_phase_colours(phase_colours, num_phases: int) -> np.ndarray:
    """
    Convert phase_colours to an RGB array of shape (num_phases, 3).

    Accepts None (site_colors), a function i -> RGB(A), a single color applied to every
    phase, or a sequence / array of num_phases colors. Raises ValueError otherwise.
    """
    import matplotlib.colors as mcolors

    if phase_colours is None:
        phase_colours = site_colors
    if callable(phase_colours):
        return np.array([phase_colours(i)[:3] for i in range(num_phases)], dtype=float)

    try:
        return np.tile(mcolors.to_rgb(phase_colours), (num_phases, 1))
    except (ValueError, TypeError):
        pass

    try:
        rgb = np.array([mcolors.to_rgb(c) for c in phase_colours], dtype=float)
    except (ValueError, TypeError):
        rgb = None
    if rgb is None or rgb.shape != (num_phases, 3):
        raise ValueError(
            f"phase_colours must be a color, a function, or {num_phases} colors, "
            f"got {phase_colours!r}."
        )
    return rgb


def _plot_gaussian_ellipse(ax, mean, cov, n_std: float = 2.0, **kwargs):
    """Draw the n_std confidence ellipse of a 2D Gaussian with the given mean and covariance."""
    from matplotlib.patches import Ellipse

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0.0))

    ellipse = Ellipse(mean, width, height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def _plot_phase_reference(
    lattice,
    figax: tuple,
    phase_index: int,
    phase_colour,
    *,
    atom_size: float = 100.0,
    arrow_scale_factor: float = 2.0,
    adaptive_head: bool = True,
    max_head_ratio: float = 3.5,
    phase_arrow_headlength: float = 8.0,
    phase_arrow_headwidth: float = 8.0,
    phase_arrow_tail_width: float = 3.0,
    lattice_linewidth: float = 1.0,
    reference_atom_colour=None,
    other_atom_colour=None,
    lattice_line_colour=None,
    alpha_phase_atom: float = 1.0,
    alpha_reference_atom: float = 1.0,
    alpha_other_atom: float = 1.0,
    alpha_shadow_atom: float = 0.2,
    alpha_phase_arrow: float = 1.0,
    alpha_lattice_lines: float = 0.3,
):
    """
    Unit cell diagram of the mean polarization of one phase.

    The measured atom is drawn displaced along the phase mean (scaled by arrow_scale_factor),
    with a faint shadow at its ideal position, an arrow between them, the reference neighbours,
    the other sites, and the lattice lines of the surrounding unit cell.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import ArrowStyle, FancyArrowPatch

    fig, ax = figax
    reference_rgb = np.array(mcolors.to_rgb(reference_atom_colour or site_colors(-1)))
    other_rgb = np.array(mcolors.to_rgb(other_atom_colour or site_colors(-2)))
    line_rgb = np.array(mcolors.to_rgb(lattice_line_colour or site_colors(-1)))
    phase_rgb = np.array(mcolors.to_rgb(phase_colour))

    _, u, v = (np.asarray(x, dtype=float) for x in lattice._lat)
    A = np.column_stack((u, v))
    phase_vector = np.asarray(lattice._polarization_means[phase_index], dtype=float)
    measure_ind, reference_ind = lattice._pol_meas_ref_ind
    frac_positions = np.asarray(lattice._positions_frac, dtype=float)
    corner_ind = np.array([[i, j] for i in [1.0, 0.0, -1.0] for j in [1.0, 0.0, -1.0]])

    def tile(ind: np.ndarray) -> np.ndarray:
        return (ind[:, None, :] + corner_ind[None, :, :]).reshape(-1, 2)

    def within_unit_cell(ind: np.ndarray) -> np.ndarray:
        return ind[np.max(np.abs(ind), axis=1) <= 1.0 + 1e-9]

    # Fractional positions of each atom group within one unit cell of the measured atom
    reference_frac = within_unit_cell(lattice._most_common_neighbours.reshape(-1, 2))
    measured_frac = within_unit_cell(tile(np.zeros((1, 2))))
    is_origin = np.all(np.abs(measured_frac) < 1e-6, axis=1)
    other_measured_frac = measured_frac[~is_origin]
    shadow_frac = measured_frac[is_origin]
    tip_frac = (phase_vector * arrow_scale_factor).reshape(1, 2)

    indices = np.arange(len(frac_positions))
    other_frac = frac_positions[(indices != measure_ind) & (indices != reference_ind)]
    if other_frac.size > 0:
        # Offsets of up to 2 cells so that every image within one cell is found
        other_frac = np.unique(within_unit_cell(tile(tile(other_frac))), axis=0)
    else:
        other_frac = np.zeros((0, 2))

    # Lattice lines between atoms that share a u or v index of -1, 0 or 1
    all_frac = np.vstack((reference_frac, shadow_frac, tip_frac, other_measured_frac, other_frac))
    all_pos = all_frac @ A.T
    drawn_lines = set()
    for i in range(len(all_frac)):
        for j in range(i + 1, len(all_frac)):
            same_u = abs(all_frac[i, 0] - all_frac[j, 0]) < 1e-6
            same_v = abs(all_frac[i, 1] - all_frac[j, 1]) < 1e-6
            on_u_line = same_u and np.any(
                np.abs(all_frac[i, 0] - np.array([1.0, 0.0, -1.0])) < 1e-6
            )
            on_v_line = same_v and np.any(
                np.abs(all_frac[i, 1] - np.array([1.0, 0.0, -1.0])) < 1e-6
            )
            if not (on_u_line or on_v_line):
                continue
            line_id = tuple(sorted((tuple(all_pos[i]), tuple(all_pos[j]))))
            if line_id in drawn_lines:
                continue
            drawn_lines.add(line_id)
            ax.plot(
                [all_pos[i, 1], all_pos[j, 1]],
                [all_pos[i, 0], all_pos[j, 0]],
                linestyle="-",
                linewidth=lattice_linewidth,
                color=line_rgb,
                alpha=alpha_lattice_lines,
                zorder=1,
            )

    # Atoms
    for frac, colour, alpha, zorder in (
        (other_frac, other_rgb, alpha_other_atom, 5),
        (reference_frac, reference_rgb, alpha_reference_atom, 6),
        (shadow_frac, phase_rgb, alpha_shadow_atom, 4),
        (other_measured_frac, phase_rgb, alpha_phase_atom, 7),
        (tip_frac, phase_rgb, alpha_phase_atom, 7),
    ):
        if len(frac) > 0:
            fig, ax = plot_atoms_2d(
                frac @ A.T,
                site_number=phase_index,
                figax=(fig, ax),
                size=atom_size,
                alpha=alpha,
                zorder=zorder,
                color_override=colour,
            )

    # Arrow from the ideal position to the displaced atom
    pol_vector = phase_vector @ A.T
    if adaptive_head:
        arrow_length = float(np.linalg.norm(pol_vector))
        headlength = max(min(phase_arrow_headlength, arrow_length * max_head_ratio), 0.5)
        headwidth = max(min(phase_arrow_headwidth, arrow_length * max_head_ratio), 0.5)
    else:
        headlength, headwidth = phase_arrow_headlength, phase_arrow_headwidth
    arrow = FancyArrowPatch(
        (0, 0),
        (pol_vector[1] * arrow_scale_factor, pol_vector[0] * arrow_scale_factor),
        arrowstyle=ArrowStyle.Simple(
            head_length=headlength, head_width=headwidth, tail_width=phase_arrow_tail_width
        ),
        mutation_scale=1.0,
        facecolor=phase_rgb,
        edgecolor=reference_rgb,
        alpha=alpha_phase_arrow,
        zorder=8,
        capstyle="round",
        joinstyle="round",
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arrow)

    # Formatting: 10% padding around the atoms
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_pad, y_pad = 0.1 * (xlim[1] - xlim[0]), 0.1 * (ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - x_pad, xlim[1] + x_pad)
    ax.set_ylim(ylim[0] - y_pad, ylim[1] + y_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_title(f"Phase {phase_index}", fontsize=14, fontweight="bold")

    return fig, ax


def create_colors_from_probabilities(
    probabilities,
    num_phases,
    category_colors=None,
    clip_range=None,
    saturation_boost=1.0,
):
    """
    Colors of atoms from their phase probabilities.

    - 2 phases: the order parameter OP = p1 - p0 is mapped from phase 0 (OP = -1) through white
      (OP = 0) to phase 1 (OP = +1). clip_range limits the OP range mapped to full saturation.
    - 3 phases: the phase colors are mixed by probability and faded to white with a smooth
      function of the certainty max(p). clip_range limits the certainty range.
    - Otherwise: the phase colors are mixed by probability.

    Parameters
    ----------
    probabilities : NDArray, shape (N, num_phases)
        Probabilities of each phase (rows sum to 1).
    num_phases : int
        Number of phases.
    category_colors : NDArray | None, shape (num_phases, 3)
        RGB colors of the phases. If None, site_colors is used.
    clip_range : tuple[float, float] | None
        See above.
    saturation_boost : float, default 1.0
        Multiplier on the saturation.

    Returns
    -------
    colors : NDArray, shape (N, 3)
    """
    import matplotlib.colors as mcolors

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1, num_phases)
    if category_colors is None:
        category_colors = np.array([site_colors(i) for i in range(num_phases)], dtype=float)
    mixed_colors = probabilities @ np.asarray(category_colors, dtype=float)

    if num_phases == 2:
        op = probabilities[:, 1] - probabilities[:, 0]
        if clip_range is not None:
            clip_min, clip_max = clip_range
            op_normalized = (np.clip(op, clip_min, clip_max) - clip_min) / (clip_max - clip_min)
        else:
            op_normalized = (op + 1.0) / 2.0
        # 0 at OP = 0 (uncertain), 1 at the edges of the range
        certainty = np.abs(op_normalized - 0.5) * 2.0
    elif num_phases == 3:
        certainty = np.max(probabilities, axis=1)
        if clip_range is not None:
            clip_min, clip_max = clip_range
            mid_point = (clip_min + clip_max) / 2.0
            certainty = np.abs(np.clip(certainty, clip_min, clip_max) - mid_point) / (
                np.abs(clip_max - clip_min) / 2.0
            )
        certainty = 3 * certainty**2 - 2 * certainty**3
    else:
        certainty = None

    if certainty is None:
        final_colors = np.clip(mixed_colors, 0, 1)
        saturation_scale = np.full(len(final_colors), saturation_boost)
    else:
        # Blend with white: uncertain -> white, certain -> phase color
        final_colors = np.clip(
            certainty[:, None] * mixed_colors + (1 - certainty[:, None]) * np.ones(3), 0, 1
        )
        saturation_scale = certainty * saturation_boost

    hsv_colors = mcolors.rgb_to_hsv(final_colors)
    hsv_colors[:, 1] = np.clip(hsv_colors[:, 1] * saturation_scale, 0, 1)

    return np.clip(mcolors.hsv_to_rgb(hsv_colors), 0, 1)


def add_2phase_colorbar(
    ax_cbar, scatter_colours, match_ax=None, clip_range=None, saturation_boost=1.0
):
    """
    Draw the 2-phase order parameter colorbar on ax_cbar.

    The colormap goes from phase 0 (OP = -1, bottom) through white (OP = 0) to phase 1
    (OP = +1, top), where OP = p1 - p0. If clip_range is given, only that range is labeled.
    If match_ax is given, the colorbar is aligned to its height.
    """
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    color0, color1 = np.asarray(scatter_colours[0]), np.asarray(scatter_colours[1])
    if saturation_boost != 1.0:
        hsv = mcolors.rgb_to_hsv(np.array([color0, color1], dtype=float))
        hsv[:, 1] = np.clip(hsv[:, 1] * saturation_boost, 0, 1)
        color0, color1 = mcolors.hsv_to_rgb(hsv)

    cmap = LinearSegmentedColormap.from_list("two_phase", [color0, (1, 1, 1), color1], N=256)
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax_cbar.imshow(gradient, aspect="auto", cmap=cmap, origin="lower")

    if match_ax is not None:
        pos_match = match_ax.get_position()
        pos_cbar = ax_cbar.get_position()
        ax_cbar.set_position([pos_cbar.x0, pos_match.y0, pos_cbar.width, pos_match.height])

    if clip_range is not None:
        clip_min, clip_max = clip_range
        span = clip_max - clip_min
        if span < 0.3:
            tick_vals = [clip_min, (clip_min + clip_max) / 2.0, clip_max]
        elif span < 1.0:
            step = 0.2 if span >= 0.4 else 0.1
            tick_vals = np.arange(clip_min, clip_max + step / 2, step)
        else:
            tick_vals = clip_min + span * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tick_pixels = [int((val - clip_min) / span * 255) for val in tick_vals]
        tick_labels = [f"{val:.2g}" for val in tick_vals]
    else:
        tick_pixels = [0, 64, 128, 192, 255]
        tick_labels = ["-1", "-0.5", "0", "+0.5", "+1"]

    ax_cbar.set_xticks([])
    ax_cbar.set_yticks(tick_pixels)
    ax_cbar.set_yticklabels(tick_labels)
    ax_cbar.yaxis.tick_right()
    ax_cbar.set_ylabel("Order Parameter", rotation=270, labelpad=14)
    ax_cbar.yaxis.set_label_position("right")

    return ax_cbar


def add_3phase_color_triangle(
    ax_triangle, scatter_colours, match_ax=None, clip_range=None, saturation_boost=1.0
):
    """
    Draw the 3-phase ternary color triangle on ax_triangle, with phase 0 at the bottom left,
    phase 1 at the bottom right and phase 2 at the top. clip_range and saturation_boost are
    passed to create_colors_from_probabilities. If match_ax is given, the top of the triangle
    is aligned with its top.
    """
    resolution = 100
    probabilities, positions = [], []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            p0, p1, p2 = i / resolution, j / resolution, (resolution - i - j) / resolution
            probabilities.append([p0, p1, p2])
            positions.append([0.5 * (2 * p1 + p2), (np.sqrt(3) / 2) * p2])
    positions = np.array(positions)

    colors = create_colors_from_probabilities(
        np.array(probabilities),
        3,
        scatter_colours,
        clip_range=clip_range,
        saturation_boost=saturation_boost,
    )
    ax_triangle.scatter(
        positions[:, 0], positions[:, 1], c=colors, s=20, marker="s", edgecolors="none"
    )

    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax_triangle.plot(vertices[:, 0], vertices[:, 1], "k-", linewidth=2)
    for k, (vx, vy) in enumerate(vertices[:3]):
        ax_triangle.scatter(
            vx, vy, s=150, c=[scatter_colours[k]], edgecolors="black", linewidths=2, zorder=10
        )
        text_y, va = (vy + 0.1, "bottom") if k == 2 else (vy - 0.1, "top")
        ax_triangle.text(
            vx, text_y, f"Phase {k}", ha="center", va=va, fontsize=10, fontweight="bold"
        )

    ax_triangle.set_aspect("equal")
    ax_triangle.axis("off")

    if match_ax is not None:
        pos_match = match_ax.get_position()
        pos_tri = ax_triangle.get_position()
        ax_triangle.set_position(
            [pos_tri.x0, pos_match.y1 - pos_tri.height, pos_tri.width, pos_tri.height]
        )

    return ax_triangle


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
