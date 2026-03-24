import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d


class Lattice(AutoSerialize):
    """
    Atomic lattice fitting in 2D.
    """

    _token = object()

    def __init__(
        self,
        image: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Lattice.from_data() to instantiate this class.")
        self._image: Dataset2d = image

    ### --- Constructors ---
    @classmethod
    def from_data(
        cls,
        image: Dataset2d | NDArray,
        normalize_min: bool = True,
        normalize_max: bool = True,
    ) -> "Lattice":
        """
        Create a Lattice instance from a 2D image-like input.

        Parameters:
        - image: A 2D numpy array or a Dataset2d instance representing the image.
        - normalize_min: If True, shift the image so its minimum becomes 0.
        - normalize_max: If True, scale the image by its maximum after min-shift
          so values are in [0, 1]. If the maximum is 0 or non-finite (NaN/Inf),
          scaling is skipped to avoid invalid operations.

        Notes:
        - Non-2D inputs and empty arrays raise a ValueError.
        - Inputs with boolean dtype are safely converted to float before normalization.
        - NaN values are ignored when computing min/max (using nanmin/nanmax). If the
          data is all-NaN, normalization is skipped.
        """
        if isinstance(image, Dataset2d):
            ds2d = image
            # Ensure numeric operations are valid (e.g., for bool dtype)
            ds2d.array = np.asarray(ds2d.array, dtype=float)
            # Validate shape
            if ds2d.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if ds2d.array.size == 0:
                raise ValueError("Input image array must not be empty.")
        else:
            # Validate dimensionality and emptiness before any processing
            arr = np.asarray(image)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            # Convert to float for safe arithmetic (handles bool arrays)
            arr = arr.astype(float, copy=False)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                ds2d = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                ds2d = Dataset2d(arr)  # type: ignore[call-arg]

        # Normalization (robust to constant, NaN, and bool inputs)
        if normalize_min:
            # Use nanmin to ignore NaNs; if all-NaN, skip
            try:
                min_val = np.nanmin(ds2d.array)
                if np.isfinite(min_val):
                    ds2d.array = ds2d.array - min_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        if normalize_max:
            # Use nanmax to ignore NaNs; skip division if max <= 0 or not finite
            try:
                max_val = np.nanmax(ds2d.array)
                if np.isfinite(max_val) and max_val > 0.0:
                    ds2d.array = ds2d.array / max_val
            except ValueError:
                # Raised when all values are NaN; skip
                pass

        return cls(image=ds2d, _token=cls._token)

    ### --- Properties ---
    @property
    def image(self) -> Dataset2d:
        return self._image

    @image.setter
    def image(self, value: Dataset2d | NDArray):
        if isinstance(value, Dataset2d):
            # Ensure numeric dtype to avoid boolean arithmetic issues downstream
            value.array = np.asarray(value.array, dtype=float)
            # Validate shape
            if value.array.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if value.array.size == 0:
                raise ValueError("Input image array must not be empty.")
            self._image = value
        else:
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError("Input image must be a 2D array.")
            if arr.size == 0:
                raise ValueError("Input image array must not be empty.")
            arr = arr.astype(float, copy=False)
            if hasattr(Dataset2d, "from_array") and callable(getattr(Dataset2d, "from_array")):
                self._image = Dataset2d.from_array(arr)  # type: ignore[attr-defined]
            else:
                self._image = Dataset2d(arr)  # type: ignore[call-arg]

    ### --- Functions ---
    def define_lattice(
        self,
        origin,
        u,
        v,
        refine_lattice: bool = True,
        block_size: int | None = None,
        plot_lattice: bool = True,
        bound_num_vectors: int | None = None,
        refine_maxiter: int = 200,
        **kwargs,
    ) -> "Lattice":
        """
        Define the lattice for the image using the origin and the u and v vectors starting from the origin.
        The lattice is defined as r = r0 + nu + mv.

        Parameters
        ----------
        origin : NDArray[2] | Sequence[float]
            Start point (r0) to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
            Ideally a lattice point.
        u : NDArray[2] | Sequence[float]
            Basis vector u to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        v : NDArray[2] | Sequence[float]
            Basis vector v to define the lattice.
            Enter as (row, col) as a numpy array, list, or tuple.
        refine_lattice : bool, default=True
            If True, refines the values of r0, u, and v by maximizing the bilinear intensity sum.
        block_size : int | None , default=None
            Fit the lattice points in steps of block_size * lattice_vectors(u, v).
            For example, if block_size = 5, then the lattice points will be fit in steps of
            (-5, 5)u * (-5, 5)v -> (-10, 10)u * (-10, 10)v -> ...
            block_size = None means the entire image will be fit at once.
        plot_lattice : bool, default=True
            If True, the lattice vectors and lines will be plotted overlaid on the image.
        bound_num_vectors : int | None, default=None
            The maximum number of lattice vectors to plot in each direction.
            For example, if bound_num_vectors = 5, lattice lines between (-5, 5)u * (-5, 5)v will be plotted.
            If None, the plotting bounds are set to the image edges.
        refine_maxiter : int, default=200
            Maximum number of iterations for the lattice refinement optimizer (Powell method).
        **kwargs
            Additional keyword arguments forwarded to the plotting function (show_2d), e.g., cmap, title, etc.

        Returns
        -------
        self : Lattice
            Returns the same object, modified in-place.
            The final values of r0, u, v are stored in self._lat.
        """
        # Lattice
        self._lat = np.vstack(
            (
                np.array(origin),
                np.array(u),
                np.array(v),
            )
        )
        if not self._lat.shape == (3, 2):
            raise ValueError("origin, u, v must be in (row, col) format only.")

        # Refine lattice coordinates
        # Note that we currently assume corners are local maxima
        if refine_lattice:
            from scipy.optimize import minimize

            if block_size is not None and block_size < 0:
                raise ValueError("block_size must be positive or None.")

            H, W = self._image.shape
            im = np.asarray(self._image.array, dtype=float)
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

            corners = np.array(
                [
                    [0.0, 0.0],
                    [float(H), 0.0],
                    [0.0, float(W)],
                    [float(H), float(W)],
                ],
                dtype=float,
            )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))  # (2,2)
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]  # (2,4)

            # Getting the min and max values for the indices a, b from the corners
            a_min, a_max = int(np.floor(ab[0].min())), int(np.ceil(ab[0].max()))
            b_min, b_max = int(np.floor(ab[1].min())), int(np.ceil(ab[1].max()))

            max_ind = max(abs(a_min), a_max, abs(b_min), b_max)
            if not block_size:
                steps = [max_ind]
            else:
                steps = (
                    [*np.arange(0, max_ind + 1, block_size)[1:], max_ind]
                    if max_ind > 0
                    else [max_ind]
                )

            PENALTY = 1e10
            H_CLIP = H - 2
            W_CLIP = W - 2
            a_range = np.arange(max(a_min, -max_ind), min(a_max, max_ind) + 1, dtype=np.int32)
            b_range = np.arange(max(b_min, -max_ind), min(b_max, max_ind) + 1, dtype=np.int32)
            aa, bb = np.meshgrid(a_range, b_range, indexing="ij")

            # Pre-compute all masks and bases
            all_masks = {}
            all_bases = {}
            for curr_block_size in steps:
                a_min_blk = max(a_min, -curr_block_size)
                a_max_blk = min(a_max, curr_block_size)
                b_min_blk = max(b_min, -curr_block_size)
                b_max_blk = min(b_max, curr_block_size)
                mask = (
                    (aa >= a_min_blk) & (aa <= a_max_blk) & (bb >= b_min_blk) & (bb <= b_max_blk)
                )

                aa_masked = aa[mask]
                bb_masked = bb[mask]

                all_masks[curr_block_size] = mask
                all_bases[curr_block_size] = np.column_stack(
                    [np.ones(aa_masked.size), aa_masked.ravel(), bb_masked.ravel()]
                )

            # Pre-allocate cache
            max_points = max(basis.shape[0] for basis in all_bases.values())
            x0_cache = np.empty(max_points, dtype=np.int32)
            y0_cache = np.empty(max_points, dtype=np.int32)
            dx_cache = np.empty(max_points, dtype=np.float64)
            dy_cache = np.empty(max_points, dtype=np.float64)

            def bilinear_sum(im_: np.ndarray, xy: np.ndarray) -> float:
                """Sum of bilinearly interpolated intensities at (x,y) points."""

                n_points = xy.shape[0]
                if n_points == 0:
                    return 0.0

                x, y = xy[:, 0], xy[:, 1]

                # Filter points that are within valid bounds for bilinear interpolation
                # Need x in [0, H-2] and y in [0, W-2] so that x+1 and y+1 are valid
                valid_mask = (
                    (x >= 0)
                    & (x <= H_CLIP)
                    & (y >= 0)
                    & (y <= W_CLIP)
                    & np.isfinite(x)
                    & np.isfinite(y)
                )

                n_valid = np.sum(valid_mask)
                if n_valid == 0:
                    return -PENALTY

                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                # Use pre-allocated arrays
                x0, y0 = x0_cache[:n_valid], y0_cache[:n_valid]
                dx, dy = dx_cache[:n_valid], dy_cache[:n_valid]

                np.floor(x_valid, out=dx)
                x0[:] = dx.astype(np.int32)
                np.floor(y_valid, out=dy)
                y0[:] = dy.astype(np.int32)

                np.subtract(x_valid, x0, out=dx)
                np.subtract(y_valid, y0, out=dy)

                Ia = im_[x0, y0]
                Ib = im_[x0 + 1, y0]
                Ic = im_[x0, y0 + 1]
                Id = im_[x0 + 1, y0 + 1]

                return np.sum(
                    Ia * (1 - dx) * (1 - dy)
                    + Ib * dx * (1 - dy)
                    + Ic * (1 - dx) * dy
                    + Id * dx * dy
                )

            current_basis = None

            def objective(theta: np.ndarray) -> float:
                """Function to be minimized"""
                # theta is 6-vector -> (3,2) matrix [[r0],[u],[v]]
                lat = theta.reshape(3, 2)
                xy = current_basis @ lat  # (N,2) with columns (x,y)
                # Negative: maximize intensity sum by minimizing its negative
                return -bilinear_sum(im, xy)

            minimize_options = {
                "maxiter": int(refine_maxiter),
                "xtol": 1e-3,
                "ftol": 1e-3,
                "disp": False,
            }

            lat_flat = self._lat.astype(np.float32).reshape(-1)

            for curr_block_size in steps:
                current_basis = all_bases[curr_block_size]

                res = minimize(
                    objective,
                    lat_flat,
                    method="Powell",
                    options=minimize_options,
                )

                # Update for next iteration
                lat_flat = res.x
                self._lat = res.x.reshape(3, 2)

        # plotting
        if plot_lattice:
            fig, ax = show_2d(
                self._image.array,
                returnfig=True,
                **kwargs,
            )

            # Put the image at lowest zorder so overlays sit on top
            if ax.images:
                ax.images[-1].set_zorder(0)

            H, W = self._image.shape
            r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)

            # Origin marker (TOP of stack)
            ax.scatter(
                r0[1],
                r0[0],  # (y, x)
                s=60,
                edgecolor=(0, 0, 0),
                facecolor=(0, 0.5, 0),
                marker="s",
                zorder=30,
            )

            # Lattice vectors as arrows
            n_vec = int(bound_num_vectors) if bound_num_vectors is not None else 1

            # draw n_vec arrows for u (red)
            for k in range(1, n_vec + 1):
                tip = r0 + k * u
                ax.arrow(
                    r0[1],
                    r0[0],  # base (y, x)
                    (tip - r0)[1],
                    (tip - r0)[0],  # delta (y, x)
                    length_includes_head=True,
                    head_width=4.0,
                    head_length=6.0,
                    linewidth=2.0,
                    color="red",
                    zorder=20,
                )

            # draw n_vec arrows for v (cyan)
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
                    zorder=20,
                )

            # Solve for a,b at plot corners (bounds)
            if bound_num_vectors is None:
                corners = np.array(
                    [
                        [0.0, 0.0],
                        [float(H), 0.0],
                        [0.0, float(W)],
                        [float(H), float(W)],
                    ]
                )
            else:
                n = float(bound_num_vectors)
                corners = np.array(
                    [
                        r0 - n * u,
                        r0 - n * v,
                        r0 + n * u,
                        r0 + n * v,
                    ],
                    dtype=float,
                )

            # a,b from corners; A = [u v] in columns (2x2), rhs = (corner - r0)
            A = np.column_stack((u, v))
            ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]

            a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
            b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

            # Clipping rectangle (image or custom)
            if bound_num_vectors is None:
                x_lo, x_hi = 0.0, float(H)
                y_lo, y_hi = 0.0, float(W)
            else:
                # Bounds are the min/max over the provided corners
                x_lo, x_hi = float(np.min(corners[:, 0])), float(np.max(corners[:, 0]))
                y_lo, y_hi = float(np.min(corners[:, 1])), float(np.max(corners[:, 1]))

            def clipped_segment(base: np.ndarray, direction: np.ndarray):
                """Clip base + t*direction to rectangle [x_lo,x_hi] x [y_lo,y_hi]."""
                x0, y0 = base
                dx, dy = direction
                t0, t1 = -np.inf, np.inf
                eps = 1e-12

                # x in [x_lo, x_hi]
                if abs(dx) < eps:
                    if not (x_lo <= x0 <= x_hi):
                        return None
                else:
                    tx0 = (x_lo - x0) / dx
                    tx1 = (x_hi - x0) / dx
                    t_enter, t_exit = (tx0, tx1) if tx0 <= tx1 else (tx1, tx0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                # y in [y_lo, y_hi]
                if abs(dy) < eps:
                    if not (y_lo <= y0 <= y_hi):
                        return None
                else:
                    ty0 = (y_lo - y0) / dy
                    ty1 = (y_hi - y0) / dy
                    t_enter, t_exit = (ty0, ty1) if ty0 <= ty1 else (ty1, ty0)
                    t0, t1 = max(t0, t_enter), min(t1, t_exit)

                if t0 > t1:
                    return None

                p1 = base + t0 * direction  # (x, y)
                p2 = base + t1 * direction
                return p1, p2

            # Lattice lines (zorder above image)
            # Using x=rows, y=cols: plot(y, x)

            # Lines parallel to v (vary a)
            for a in range(a_min, a_max + 1):
                base = r0 + a * u
                seg = clipped_segment(base, v)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color=(0.0, 0.7, 1.0), lw=1, clip_on=True, zorder=10)

            # Lines parallel to u (vary b)
            for b in range(b_min, b_max + 1):
                base = r0 + b * v
                seg = clipped_segment(base, u)
                if seg is None:
                    continue
                (x1, y1), (x2, y2) = seg
                ax.plot([y1, y2], [x1, x2], color="red", lw=1, clip_on=True, zorder=10)

            # Axes limits (x=rows vertical; y=cols horizontal)
            ax.set_xlim(y_lo, y_hi)
            ax.set_ylim(x_hi, x_lo)

        return self

    def add_atoms(
        self,
        positions_frac,
        numbers=None,
        intensity_min=None,
        intensity_radius=None,
        plot_atoms=True,
        *,
        edge_min_dist_px=None,
        mask=None,
        contrast_min=None,
        annulus_radii=None,
        **kwargs,
    ) -> "Lattice":
        """
        Add atoms for each lattice site by sampling all integer lattice translations that fall inside
        the image, measuring local intensity, and filtering candidates by bounds, edge distance,
        mask, and optional intensity/contrast thresholds. Optionally plots the detected atoms.

        Parameters
        ----------
        positions_frac : array-like, shape (S, 2)
            Fractional positions (a, b) of S lattice sites within the unit cell. These are offsets
            relative to the lattice origin r0 and basis vectors (u, v), and are used to tile the
            image with candidate atom centers at all visible integer translations.
        numbers : array-like of int, shape (S,), optional
            Identifier per site (e.g., species or label). If None, uses 1..S. Used only for plotting
            color coding; not used in detection logic.
        intensity_min : float, optional
            Minimum mean intensity inside the detection disk required to keep a candidate atom.
            If None, no intensity thresholding is applied.
        intensity_radius : float, optional
            Radius (in pixels) of the detection disk used to compute the mean intensity at each
            candidate center. If None, an automatic radius is estimated as half of the nearest-neighbor
            spacing in pixels (see Notes).
        plot_atoms : bool, default True
            If True, displays the image and overlays the detected atoms for each site.
        edge_min_dist_px : float, optional
            Minimum distance (in pixels) that candidate centers must maintain from the image borders.
            If a mask is provided and a distance transform can be computed, this same threshold is also
            used to enforce a minimum distance from masked boundaries.
        mask : array-like of bool, shape (H, W), optional
            Binary mask defining valid regions. If provided:
            - When a distance transform is available, candidates must be at least edge_min_dist_px away
            from masked boundaries.
            - Otherwise, candidates are kept only if the nearest integer-pixel location is True in the mask.
        contrast_min : float, optional
            Minimum contrast required to keep a candidate, defined as (disk mean) - (annulus mean).
            If None, no contrast thresholding is applied.
        annulus_radii : tuple of float, optional
            Inner and outer radii (in pixels) of the background annulus used for contrast estimation.
            If None, defaults to (1.5 * intensity_radius, 3.0 * intensity_radius).
        **kwargs
            Additional keyword arguments forwarded to the plotting helper (show_2d) when plot_atoms is True.

        Returns
        -------
        self
            The current object, with the following side effects:
            - self._positions_frac set from positions_frac
            - self._num_sites set to S
            - self._numbers set from numbers or default sequence
            - self.atoms populated with detected atom data per site

        Raises
        ------
        ValueError
            If a provided mask does not match the image shape (H, W).

        Side Effects
        ------------
        self.atoms : Vector
            shape=(S,), fields=("x", "y", "a", "b", "int_peak"), units=("px", "px", "ind", "ind", "counts").
            For each site index s, self.atoms[s] holds a table with one row per detected atom:
            - x, y: pixel coordinates of the atom center (x is row, y is column; origin at top-left)
            - a, b: fractional lattice indices for that atom (including the site's fractional offset plus integer translations)
            - int_peak: mean intensity inside the detection disk at (x, y)

        Notes
        -----
        Lattice and image geometry
            - The image array is of shape (H, W), where x indexes rows and y indexes columns.
            - Lattice parameters are taken from self._lat = [r0, u, v], with r0 the origin (in pixels)
            and u, v the lattice basis vectors (in pixels). Candidate centers are generated by tiling
            each site's fractional offset across all integer translations that map into the image bounds.
            - The visible range of integer translations (a, b) is determined by projecting the image corners
            through the inverse lattice transform.

        Automatic detection radius (when intensity_radius is None)
            - If there are at least two sites, the nearest-neighbor spacing is computed from fractional
            differences between site positions, accounting for periodic wrapping, and converted to pixels
            via the lattice matrix [u v]. The radius is set to half of this spacing.
            - If there is only one site, the spacing fallback is min(||u||, ||v||, ||u+v||, ||u-v||), and the
            radius is half of this value.
            - If the estimate is invalid or non-positive, a robust fallback of 0.5 * (0.5 * (||u|| + ||v||)) is used.

        Filtering
            - Candidates must lie fully within image bounds and satisfy the edge_min_dist_px constraint.
            - If mask is provided and a distance transform can be computed, candidates must also be at least
            edge_min_dist_px inside the masked region; otherwise, the mask must be True at the nearest integer pixel.
            - intensity_min filters by the disk mean; contrast_min filters by the difference between the disk mean
            and the annulus mean, where the annulus default is (1.5 * r, 3.0 * r).

        Plotting
            - When plot_atoms is True, the image is shown and detected atoms are rendered as semi-transparent
            colored markers per site. Colors are determined by site numbers. Axes are set to match image
            coordinates (x increasing downward).
        """
        if not hasattr(self, "_lat") or self._lat is None:
            raise ValueError(
                "Lattice vectors have not been fitted. Please call define_lattice() first."
            )
        # Handle empty positions early without creating a Vector of length 0
        positions_frac_arr = np.asarray(positions_frac, dtype=float)
        if positions_frac_arr.size == 0:
            # Bookkeeping for consistency
            self._positions_frac = np.empty((0, 2), dtype=float)
            self._num_sites = 0
            self._numbers = (
                np.array([], dtype=int)
                if numbers is None
                else np.atleast_1d(np.array(numbers, dtype=int))
            )
            # Do not construct an empty Vector with zero shape (causes error). Just return.
            return self

        self._positions_frac = np.atleast_2d(np.array(positions_frac, dtype=float))
        self._num_sites = self._positions_frac.shape[0]
        self._numbers = (
            np.arange(0, self._num_sites, dtype=int)
            if numbers is None
            else np.atleast_1d(np.array(numbers, dtype=int))
        )

        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        # Min and max values of a,b are calculated based on corners
        corners = np.array(
            [[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]], dtype=float
        )
        ab = np.linalg.lstsq(A, (corners - r0[None, :]).T, rcond=None)[0]
        a_min, a_max = int(np.floor(np.min(ab[0]))), int(np.ceil(np.max(ab[0])))
        b_min, b_max = int(np.floor(np.min(ab[1]))), int(np.ceil(np.max(ab[1])))

        def _auto_radius_px() -> float:
            """
            Estimate a default disk radius in pixels as half the nearest-neighbor spacing
            (with periodic wrapping), or from lattice vectors if insufficient points.
            """
            S = self._positions_frac
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_px = float(intensity_radius) if intensity_radius is not None else _auto_radius_px()
        rin, rout = (1.5 * r_px, 3.0 * r_px) if annulus_radii is None else annulus_radii
        R_disk = int(np.ceil(r_px))
        R_ring = int(np.ceil(rout))
        edge_thresh = float(edge_min_dist_px) if edge_min_dist_px is not None else 0.0

        DT = None
        if mask is not None:
            m = np.asarray(mask).astype(bool)
            if m.shape != (H, W):
                raise ValueError(f"mask shape {m.shape} must match image shape {(H, W)}")
            try:
                from scipy.ndimage import distance_transform_edt

                DT = distance_transform_edt(m)
            except Exception:
                DT = None

        def mean_disk(x: float, y: float) -> float:
            """
            Compute the mean image intensity within a circular disk of radius r_px centered at (x, y),
            with boundary clipping and fallback to the center pixel if empty.
            """
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_disk), min(H - 1, ix0 + R_disk)
            j0, j1 = max(0, iy0 - R_disk), min(W - 1, iy0 + R_disk)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            mask_circle = (dx * dx + dy * dy) <= (r_px * r_px)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_circle]
            if vals.size == 0:
                return float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
            return float(vals.mean())

        def mean_std_annulus(x: float, y: float) -> tuple[float, float]:
            """
            Compute the mean and standard deviation of intensities within an annulus [rin, rout] centered at (x, y),
            with boundary clipping and fallback to the center pixel and zero std if empty.
            """
            ix0, iy0 = int(np.floor(x)), int(np.floor(y))
            i0, i1 = max(0, ix0 - R_ring), min(H - 1, ix0 + R_ring)
            j0, j1 = max(0, iy0 - R_ring), min(W - 1, iy0 + R_ring)
            ii = np.arange(i0, i1 + 1)[:, None]
            jj = np.arange(j0, j1 + 1)[None, :]
            dx, dy = ii - x, jj - y
            r2 = dx * dx + dy * dy
            mask_ring = (r2 >= rin * rin) & (r2 <= rout * rout)
            vals = im[i0 : i1 + 1, j0 : j1 + 1][mask_ring]
            if vals.size == 0:
                val = float(im[np.clip(round(x), 0, H - 1), np.clip(round(y), 0, W - 1)])
                return val, 0.0
            return float(vals.mean()), float(vals.std(ddof=0))

        self.atoms = Vector.from_shape(
            shape=(self._num_sites,),
            fields=["x", "y", "a", "b", "int_peak"],
            units=["px", "px", "ind", "ind", "counts"],
        )

        for a0 in range(self._num_sites):
            da, db = self._positions_frac[a0, 0], self._positions_frac[a0, 1]
            aa, bb = np.meshgrid(
                np.arange(a_min - 1 + da, a_max + 1 + da),
                np.arange(b_min - 1 + db, b_max + 1 + db),
                indexing="ij",
            )
            basis = np.vstack((np.ones(aa.size), aa.ravel(), bb.ravel())).T
            xy = basis @ self._lat  # (N,2)

            x, y = xy[:, 0], xy[:, 1]
            in_bounds = (x >= 0.0) & (x <= H - 1) & (y >= 0.0) & (y <= W - 1)
            border_ok = (
                (x - edge_thresh >= 0.0)
                & (x + edge_thresh <= H - 1)
                & (y - edge_thresh >= 0.0)
                & (y + edge_thresh <= W - 1)
            )

            if mask is not None:
                if DT is not None:
                    ii = np.clip(np.round(x).astype(int), 0, H - 1)
                    jj = np.clip(np.round(y).astype(int), 0, W - 1)
                    mask_ok = DT[ii, jj] >= edge_thresh
                else:
                    m = np.asarray(mask).astype(bool)
                    mask_ok = m[
                        np.clip(np.round(x).astype(int), 0, H - 1),
                        np.clip(np.round(y).astype(int), 0, W - 1),
                    ]
            else:
                mask_ok = np.ones_like(in_bounds, dtype=bool)

            int_center = np.empty(xy.shape[0], dtype=float)
            for i in range(xy.shape[0]):
                int_center[i] = mean_disk(x[i], y[i])

            keep = in_bounds & border_ok & mask_ok
            if intensity_min is not None:
                keep &= int_center >= float(intensity_min)
            if contrast_min is not None:
                bg_mean = np.empty(xy.shape[0], dtype=float)
                for i in range(xy.shape[0]):
                    bg_mean[i], _ = mean_std_annulus(x[i], y[i])
                keep &= (int_center - bg_mean) >= float(contrast_min)

            if np.any(keep):
                arr = np.vstack(
                    (x[keep], y[keep], basis[keep, 1], basis[keep, 2], int_center[keep])
                ).T
            else:
                arr = np.zeros((0, 5), dtype=float)

            # --- Correct API usage ---
            self.atoms[a0] = arr

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            for a0 in range(self._num_sites):
                cell = self.atoms[a0].array
                if isinstance(cell, list) or cell is None or cell.size == 0:
                    continue
                x = self.atoms[a0].select_fields("x").array[:, 0]
                y = self.atoms[a0].select_fields("y").array[:, 0]
                rgb = site_colors(int(self._numbers[a0]))
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

        return self

    def refine_atoms(
        self,
        fit_radius=None,
        max_nfev: int = 200,
        max_move_px: float | None = None,
        plot_atoms: bool = False,
        **kwargs,
    ) -> "Lattice":
        """
        Refine atom centers by local 2D Gaussian fitting around each previously detected atom.
        Updates atom positions and peak intensity and adds per-atom sigma and background fields.
        Optionally plots the refined atoms.
        Parameters
        ----------
        fit_radius : float, optional
            Radius (in pixels) of the circular fitting region around each atom's current center.
            If None, an automatic radius is estimated as half of the nearest-neighbor spacing
            between lattice sites in pixels. When there is only one site, the spacing fallback
            is min(||u||, ||v||, ||u+v||, ||u-v||) where u and v are lattice vectors. If this
            estimate is invalid or non-positive, a robust fallback is used.
        max_nfev : int, default 200
            Maximum number of function evaluations for the non-linear least-squares solver.
        max_move_px : float, optional
            Maximum allowed movement (in pixels) of the refined center from its initial position.
            If None, defaults to the fitting radius. Bounds also enforce staying within image limits.
        plot_atoms : bool, default False
            If True, displays the image and overlays the refined atom positions.
        **kwargs
            Additional keyword arguments forwarded to the plotting helper when plot_atoms is True.

        Returns
        -------
        self
            The current object, with self.atoms updated per site to refined values.

        Raises
        ------
        ValueError
            If no atoms are present to refine (call add_atoms() first).

        Side Effects
        ------------
        self.atoms : Vector
            For each site index s, the per-atom rows are updated:
            - x, y: pixel coordinates refined by local Gaussian fitting (x is row, y is column).
            - int_peak: updated to the fitted Gaussian amplitude at the center.
            - sigma: added or updated; the fitted Gaussian width (pixels).
            - int_bg: added or updated; the fitted local constant background level.
            If "sigma" and "int_bg" fields do not exist, they are added automatically.

        Notes
        -----
        Model and fitting
            - A circular patch of radius fit_radius is extracted around each atom's current center.
            - Within that patch, a 2D isotropic Gaussian plus constant background is fit:
            I(x, y) = amp * exp(-0.5 * r^2 / sigma^2) + bg, where r^2 is the squared distance
            to the fitted center (x_c, y_c).
            - Initial guesses:
            - Center starts at the current atom position.
            - amp starts from the central pixel value minus the local median background.
            - sigma starts at max(0.5 * fit_radius, 0.5).
            - bg starts at the median of the patch outside the circular mask (or full patch median).
            - Parameter bounds:
            - Center (x_c, y_c) limited to within max_move_px of the initial center and within
                image bounds.
            - amp in [0, max(pmax - pmin, 4 * amp0)], using local patch extrema and initial amp0.
            - sigma in [0.25, max(2 x fit_radius, 1.0)].
            - bg in [pmin * (pmax - pmin), pmax + (pmax - pmin)].
            - Optimization uses scipy.optimize.least_squares with "trf" method and "soft_l1" loss.

        Automatic fitting radius (when fit_radius is None)
            - If there are at least two sites, the nearest-neighbor spacing is computed from fractional
            differences between site positions (wrapped to [-0.5, 0.5]) and converted to pixels using
            the lattice matrix [u v]; the radius is set to half of this spacing.
            - If there is only one site, the spacing fallback is min(||u||, ||v||, ||u+v||, ||u-v||),
            and the radius is half of this value.
            - If the estimate is invalid or non-positive, a robust fallback is used based on the lattice
            vector norms to ensure a reasonable, non-zero radius.

        Plotting
            - When plot_atoms is True, the image is shown and refined atom centers are rendered as
            semi-transparent colored markers per site. Colors are determined by site numbers.
            - Axes are set to match image coordinates (x increasing downward).
        """

        if not hasattr(self, "atoms"):
            raise ValueError("No atoms to refine. Call add_atoms() first.")

        im = np.asarray(self._image.array, dtype=float)
        H, W = self._image.shape
        r0, u, v = (np.asarray(x, dtype=float) for x in self._lat)
        A = np.column_stack((u, v))

        def _auto_radius_px() -> float:
            S = np.asarray(getattr(self, "_positions_frac", [[0.0, 0.0]]), dtype=float)
            if S.shape[0] >= 2:
                d = S[:, None, :] - S[None, :, :]
                d = d - np.round(d)
                same = (np.abs(d[..., 0]) < 1e-12) & (np.abs(d[..., 1]) < 1e-12)
                dpix = d @ A.T
                dist = np.linalg.norm(dpix, axis=2)
                dist[same] = np.inf
                nn = float(np.min(dist))
            else:
                nn = float(np.min(np.linalg.norm(np.stack((u, v, u + v, u - v)), axis=1)))
            if not np.isfinite(nn) or nn <= 0:
                nn = max(1.0, 0.25 * (np.linalg.norm(u) + np.linalg.norm(v)))
            return 0.5 * nn

        r_fit = float(fit_radius) if fit_radius is not None else _auto_radius_px()
        R = int(np.ceil(r_fit))
        max_move = float(max_move_px) if max_move_px is not None else r_fit

        # Ensure extra fields exist
        needed = [f for f in ("sigma", "int_bg") if f not in self.atoms.fields]
        if needed:
            self.atoms.add_fields(needed)

        # Single lookup of column indices for writing
        idx_x = self.atoms.fields.index("x")
        idx_y = self.atoms.fields.index("y")
        idx_amp = self.atoms.fields.index("int_peak")
        idx_sigma = self.atoms.fields.index("sigma")
        idx_bg = self.atoms.fields.index("int_bg")

        for s in range(self._num_sites):
            row = self.atoms[s].array
            if isinstance(row, list) or row is None or row.size == 0:
                continue

            # Intuitive reads: per-cell field arrays
            x_arr = self.atoms[s].select_fields("x").array[:, 0]
            y_arr = self.atoms[s].select_fields("y").array[:, 0]

            updated = row.copy()
            for i in range(row.shape[0]):
                x0, y0 = float(x_arr[i]), float(y_arr[i])

                ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
                i0, i1 = max(0, ix0 - R), min(H - 1, ix0 + R)
                j0, j1 = max(0, iy0 - R), min(W - 1, iy0 + R)
                if i1 <= i0 or j1 <= j0:
                    continue

                patch = im[i0 : i1 + 1, j0 : j1 + 1]

                # broadcast coordinate grids to patch shape
                ii = np.arange(i0, i1 + 1)[:, None]
                jj = np.arange(j0, j1 + 1)[None, :]
                II = np.broadcast_to(ii, patch.shape)
                JJ = np.broadcast_to(jj, patch.shape)

                r2 = (II - x0) ** 2 + (JJ - y0) ** 2
                mask = r2 <= (r_fit * r_fit)
                if not np.any(mask):
                    continue

                vals = patch[mask].astype(float).ravel()
                pmin, pmax = float(vals.min()), float(vals.max())
                bg0 = float(np.median(patch[~mask])) if np.any(~mask) else float(np.median(patch))
                amp0 = max(float(im[np.clip(ix0, 0, H - 1), np.clip(iy0, 0, W - 1)] - bg0), 1e-6)
                sig0 = max(r_fit * 0.5, 0.5)

                x_coords = II[mask].astype(float).ravel()
                y_coords = JJ[mask].astype(float).ravel()

                def residual(theta):
                    x_c, y_c, amp, sig, bg = theta
                    sig2 = max(sig, 1e-6) ** 2
                    rr = (x_coords - x_c) ** 2 + (y_coords - y_c) ** 2
                    model = amp * np.exp(-0.5 * rr / sig2) + bg
                    return model - vals

                # movement-limited bounds + image bounds
                x_lb = max(x0 - max_move, 0.0)
                x_ub = min(x0 + max_move, H - 1.0)
                y_lb = max(y0 - max_move, 0.0)
                y_ub = min(y0 + max_move, W - 1.0)

                lb = [x_lb, y_lb, 0.0, 0.25, pmin - (pmax - pmin)]
                ub = [
                    x_ub,
                    y_ub,
                    max(pmax - pmin, amp0 * 4.0),
                    max(2.0 * r_fit, 1.0),
                    pmax + (pmax - pmin),
                ]
                theta0 = [x0, y0, amp0, sig0, bg0]

                res = least_squares(
                    residual,
                    theta0,
                    bounds=(lb, ub),
                    method="trf",
                    loss="soft_l1",
                    max_nfev=int(max_nfev),
                    xtol=1e-6,
                    ftol=1e-6,
                    gtol=1e-6,
                )

                x_c, y_c, amp, sig, bg = res.x
                updated[i, idx_x] = x_c
                updated[i, idx_y] = y_c
                updated[i, idx_amp] = amp
                updated[i, idx_sigma] = sig
                updated[i, idx_bg] = bg

            self.atoms[s] = updated

        if plot_atoms:
            fig, ax = show_2d(self._image.array, returnfig=True, **kwargs)
            if ax.images:
                ax.images[-1].set_zorder(0)
            for s in range(self._num_sites):
                cell = self.atoms[s].array
                if isinstance(cell, list) or cell is None or cell.size == 0:
                    continue
                xs = self.atoms[s].select_fields("x").array[:, 0]
                ys = self.atoms[s].select_fields("y").array[:, 0]
                rgb = site_colors(int(self._numbers[s]))
                ax.scatter(
                    ys,
                    xs,
                    s=18,
                    facecolor=(rgb[0], rgb[1], rgb[2], 0.25),
                    edgecolor=(rgb[0], rgb[1], rgb[2], 0.9),
                    linewidths=0.75,
                    marker="o",
                    zorder=25,
                )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        return self


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
        # extras to improve variety when cycling:
        (0.60, 0.20, 0.80),
        (0.30, 0.75, 0.75),
        (0.80, 0.40, 0.00),
        (0.20, 0.60, 0.20),
        (0.70, 0.70, 0.00),
        (1.00, 1.00, 1.00),  # -2: white
        (0.00, 0.00, 0.00),  # -1: black
        # ENSURE BLACK IS ALWAYS LAST IF ADDING NEW COLORS
    ]

    # Check if input is a list, tuple, or array
    if isinstance(number, int):
        # Original behavior for single integer
        idx = int(number) % len(palette)
        return palette[idx]
    else:
        # Convert to numpy array for vectorized operations
        numbers = np.asarray(number, dtype=int)
        indices = numbers % len(palette)
        # Return array of RGB tuples
        return np.array([palette[idx] for idx in indices.flat]).reshape(numbers.shape + (3,))
