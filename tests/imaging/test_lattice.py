import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.testing import assert_array_almost_equal

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import load
from quantem.imaging.lattice import Lattice


class TestLatticeInit:
    """Test Lattice initialization and from_data."""

    def test_init_and_constructor(self):
        """Test that direct init is blocked and from_data works."""
        image = np.random.randn(100, 100)
        ds2d = Dataset2d.from_array(image)

        with pytest.raises(RuntimeError, match="Use Lattice.from_data"):
            Lattice(ds2d)

        lattice_img = Lattice.from_data(image)
        lattice_dset = Lattice.from_data(ds2d)
        assert isinstance(lattice_img, Lattice)
        assert lattice_img.image is not None
        assert isinstance(lattice_dset, Lattice)
        assert lattice_dset.image is not None

    def test_normalization(self):
        """Test min/max normalization."""
        image = np.random.randn(100, 100) * 1000.0
        image[0, 0] = -10.0
        image[99, 99] = 1000.0

        # Both normalizations
        lattice = Lattice.from_data(image)
        assert lattice.image.array.min() == 0
        assert lattice.image.array.max() == 1

        # No normalization
        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=False)
        assert_array_almost_equal(lattice.image.array, image)

        # Min normalization
        lattice = Lattice.from_data(image, normalize_min=True, normalize_max=False)
        assert lattice.image.array.min() == 0

        # Max normalization
        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=True)
        assert lattice.image.array.max() == 1

    def test_edge_cases(self):
        """Test NaN handling."""
        nan_arr = np.array([[1, np.nan], [3, 4]], dtype=float)
        lattice = Lattice.from_data(nan_arr)
        assert isinstance(lattice, Lattice)

    def test_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError, match="must be a 2D array"):
            Lattice.from_data(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="must be a 2D array"):
            Lattice.from_data(np.ones((2, 2, 2)))

        with pytest.raises(ValueError, match="must not be empty"):
            Lattice.from_data(np.array([[]]))

        with pytest.raises(ValueError, match="does not have any valid values"):
            Lattice.from_data(np.full((3, 3), np.nan))


class TestLatticeImage:
    """Test image property getter and setter."""

    def test_image_property(self):
        """Test getting and setting image."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # Get
        assert isinstance(lattice.image, Dataset2d)

        # Set with new array
        new_image = np.random.randn(50, 50)
        lattice.image = new_image
        assert lattice.image.array.shape == (50, 50)

        # Invalid set
        with pytest.raises(ValueError, match="must be a 2D array"):
            lattice.image = np.array([1, 2, 3])


class TestDefineLatticeVectors:
    """Test define_lattice_vectors method."""

    def test_basic_define(self):
        """Test basic lattice definition."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        result = lattice.define_lattice_vectors(
            origin=[50, 50], u=[5, 0], v=[0, 5], refine_lattice=False
        )

        assert result is lattice
        assert hasattr(lattice, "_lat")
        assert lattice._lat.shape == (3, 2)

    def test_refinement_options(self):
        """Test lattice refinement and block_size."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # With refinement
        lattice.define_lattice_vectors(
            origin=[50, 50],
            u=[5, 0],
            v=[0, 5],
            refine_lattice=True,
            refine_maxiter=5,
        )
        assert lattice._lat.shape == (3, 2)

        # With block_size
        lattice.define_lattice_vectors(
            origin=[50, 50],
            u=[5, 0],
            v=[0, 5],
            refine_lattice=True,
            refine_maxiter=5,
            block_size=5,
        )
        assert lattice._lat.shape == (3, 2)

    def test_invalid_lattice_params(self):
        """Test invalid lattice parameters."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # Wrong shape
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[1, 2, 3], u=[5, 0], v=[0, 5])

        # Negative block_size
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[50, 50], u=[5, 0], v=[0, 5], block_size=-1)

        # Origin out of bounds
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[10, 105], u=[5, 0], v=[0, 5])

        # Non-ivertible lattice vectors
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[50, 50], u=[5, 0], v=[10, 0])


class TestLatticeAtoms:
    """Test add_atoms method."""

    @pytest.fixture
    def simple_lattice(self):
        """Create a fitted lattice with atoms without defined lattice vectors."""
        # Create synthetic image
        H, W = 100, 100
        image = np.random.randn(H, W) * 0.1

        # Add some peaks with random shifts
        peaks = [
            (25, 25),
            (25, 50),
            (25, 75),
            (50, 25),
            (50, 50),
            (50, 75),
            (75, 25),
            (75, 50),
            (75, 75),
        ]
        for y, x in peaks:
            # Add random shift up to 3 pixels in each direction
            y_shift = np.random.randint(-3, 4)
            x_shift = np.random.randint(-3, 4)
            y_shifted = y + y_shift
            x_shifted = x + x_shift

            yy, xx = np.ogrid[-10:11, -10:11]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)
            y_start, y_end = max(0, y_shifted - 10), min(H, y_shifted + 11)
            x_start, x_end = max(0, x_shifted - 10), min(W, x_shifted + 11)
            peak_h, peak_w = y_end - y_start, x_end - x_start
            image[y_start:y_end, x_start:x_end] += peak[:peak_h, :peak_w]

        lattice = Lattice.from_data(image)

        return lattice

    @pytest.fixture
    def fitted_lattice(self):
        """Create a fitted lattice with atoms."""
        # Create synthetic image
        H, W = 100, 100
        image = np.random.randn(H, W) * 0.1

        # Add some peaks with random shifts
        peaks = [
            (25, 25),
            (25, 50),
            (25, 75),
            (50, 25),
            (50, 50),
            (50, 75),
            (75, 25),
            (75, 50),
            (75, 75),
        ]
        for y, x in peaks:
            # Add random shift up to 2 pixels in each direction
            y_shift = np.random.randint(-2, 3)
            x_shift = np.random.randint(-2, 3)
            y_shifted = y + y_shift
            x_shifted = x + x_shift

            yy, xx = np.ogrid[-10:11, -10:11]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)
            y_start, y_end = max(0, y_shifted - 10), min(H, y_shifted + 11)
            x_start, x_end = max(0, x_shifted - 10), min(W, x_shifted + 11)
            peak_h, peak_w = y_end - y_start, x_end - x_start
            image[y_start:y_end, x_start:x_end] += peak[:peak_h, :peak_w]

        lattice = Lattice.from_data(image)

        # Define lattice vectors before adding atoms
        lattice.define_lattice_vectors(
            origin=[23.0, 24.0],
            u=[27.0, 0.0],
            v=[0.0, 25.0],
        )

        return lattice

    def test_add_atoms_basic(self, fitted_lattice: Lattice):
        """Test basic add_atoms."""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac)

        assert result is fitted_lattice
        # Check that atoms were added
        assert hasattr(fitted_lattice, "_atoms") or hasattr(fitted_lattice, "atoms")

    def test_add_atoms_plotting(self, fitted_lattice: Lattice):
        """Test add_atoms sets defualt_plot to atoms."""
        positions_frac = np.array([[0.0, 0.0]])

        fitted_lattice.add_atoms(positions_frac)

        assert fitted_lattice.default_plot == "atoms"

    def test_add_atoms_with_all_parameters(self, simple_lattice: Lattice):
        """Test add_atoms with all optional parameters."""
        fitted_lattice = simple_lattice.define_lattice_vectors(
            origin=[25.0, 25.0], u=[50.0, 0.0], v=[0.0, 50.0]
        )
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = np.array([0, 1, 1, 2])
        mask = np.ones(fitted_lattice.image.shape, dtype=bool)
        mask[:30, :30] = False

        result = fitted_lattice.add_atoms(
            positions_frac,
            numbers=numbers,
            intensity_min=0.1,
            intensity_radius=5,
            edge_min_dist_px=5,
            mask=mask,
            contrast_min=0.2,
            annulus_radii=(5, 10),
        )

        assert result is simple_lattice

    def test_add_atoms_empty_positions(self, fitted_lattice: Lattice):
        """Test add_atoms with empty positions array."""
        positions_frac = np.array([]).reshape(0, 2)

        result = fitted_lattice.add_atoms(positions_frac)

        assert result is fitted_lattice

    def test_add_atoms_without_fitting_raises_error(self):
        """Test that add_atoms raises error if lattice not fitted."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        positions_frac = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Lattice vectors have not been fitted"):
            lattice.add_atoms(positions_frac)

    def test_refine_atoms_chaining_and_plotting(self, fitted_lattice: Lattice):
        """Test refine_atoms can be called by chaining functions and default_plot is set to atoms."""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac).refine_atoms()

        assert result is fitted_lattice
        # Check that atoms were added
        assert hasattr(fitted_lattice, "_atoms") or hasattr(fitted_lattice, "atoms")

        assert fitted_lattice.default_plot == "atoms"

    def test_refine_atoms_without_add_atoms_raises_error(self, fitted_lattice: Lattice):
        """Test that refine_atoms raises error if add_atoms hasn't been called."""
        with pytest.raises(ValueError, match=r"No atoms to refine\. Call add_atoms\(\) first\."):
            fitted_lattice.refine_atoms()

    def test_refine_atoms_internal_call(self, fitted_lattice: Lattice):
        """Test refine_atoms = True parameter in add_atoms"""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac, refine_atoms=True)

        assert result is fitted_lattice
        # Check that atoms were added
        assert hasattr(fitted_lattice, "_atoms") or hasattr(fitted_lattice, "atoms")
        assert fitted_lattice.default_plot == "atoms"

    def test_refine_atoms_with_all_parameters(self, simple_lattice: Lattice):
        """Test refine_atoms with all optional parameters."""
        fitted_lattice = simple_lattice.define_lattice_vectors(
            origin=[25.0, 25.0], u=[50.0, 0.0], v=[0.0, 50.0]
        )
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = np.array([0, 1, 1, 2])
        mask = np.ones(fitted_lattice.image.shape, dtype=bool)
        mask[:30, :30] = False

        result = fitted_lattice.add_atoms(
            positions_frac,
            numbers=numbers,
            intensity_min=0.1,
            intensity_radius=5,
            edge_min_dist_px=5,
            mask=mask,
            contrast_min=0.2,
            annulus_radii=(5, 10),
        ).refine_atoms(
            fit_radius=15.0,
            max_nfev=250,
            max_move_px=5.0,
        )

        assert result is simple_lattice

    def test_refine_atoms_internal_call_with_all_parameters(self, simple_lattice: Lattice):
        """Test refine_atoms with all optional parameters."""
        fitted_lattice = simple_lattice.define_lattice_vectors(
            origin=[25.0, 25.0], u=[50.0, 0.0], v=[0.0, 50.0]
        )
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = np.array([0, 1, 1, 2])
        mask = np.ones(fitted_lattice.image.shape, dtype=bool)
        mask[:30, :30] = False

        result = fitted_lattice.add_atoms(
            positions_frac,
            numbers=numbers,
            intensity_min=0.1,
            intensity_radius=5,
            edge_min_dist_px=5,
            mask=mask,
            contrast_min=0.2,
            annulus_radii=(5, 10),
            refine_atoms=True,
            fit_radius=15.0,
            max_nfev=250,
            max_move_px=5.0,
        )

        assert result is simple_lattice


class TestLatticeAtomsSynGT:
    """
    End-to-end test of define_lattice_vectors -> add_atoms -> refine_atoms
    against a synthetic image with known ground-truth atom positions.
    """

    @pytest.fixture
    def synthetic_lattice_data(self):
        """
        Build a synthetic image from known lattice vectors (u, v) and a
        known fractional basis, with a small random perturbation applied to
        each atom's position before it is rendered as a Gaussian peak.

        The perturbed position (not the ideal lattice site) is stored as
        ground truth, since that is where the actual peak in the image sits.
        Ideal lattice geometry is only used to seed define_lattice_vectors
        and to tile candidate sites in add_atoms.

        Unit-cell translations are tiled over whatever range fits inside the
        image (projecting the image corners through the inverse lattice,
        same approach used internally by add_atoms), rather than a fixed
        a/b range. Sites within `margin` pixels of the border are dropped so
        that rendered peaks are never clipped by the image edge.
        """
        rng = np.random.default_rng(42)

        H, W = 150, 150
        margin = 8  # pixels; also passed to add_atoms/refine_atoms below

        # Ground-truth lattice geometry, in (row, col) pixel convention
        r0_true = np.array([12.0, 10.0])
        u_true = np.array([14.0, 1.0])
        v_true = np.array([1.0, 13.0])

        # Two-atom basis in fractional unit-cell coordinates, with distinct
        # peak intensities per site
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.5]])
        site_amplitudes = [1.0, 0.6]

        # Determine the range of integer translations (a, b) that could fall
        # inside the image, by projecting the image corners through the
        # inverse lattice transform (same approach used in add_atoms).
        A = np.column_stack((u_true, v_true))
        corners = np.array([[0.0, 0.0], [float(H), 0.0], [0.0, float(W)], [float(H), float(W)]])
        ab = np.linalg.lstsq(A, (corners - r0_true[None, :]).T, rcond=None)[0]
        a_min, a_max = int(np.floor(ab[0].min())) - 1, int(np.ceil(ab[0].max())) + 1
        b_min, b_max = int(np.floor(ab[1].min())) - 1, int(np.ceil(ab[1].max())) + 1

        gauss_sigma = 1.2
        perturb_scale = 1.2  # pixels; typical polarization values (~10%)

        image = np.zeros((H, W))
        rr_grid, cc_grid = np.mgrid[0:H, 0:W]  # rr = row index, cc = col index

        # Maps (site_idx, a, b) -> true (perturbed) (x, y) = (row, col) position
        ground_truth = {}

        # Adding perturbations
        # This is slightly inaccurate compared to realistic polarization values;
        # Realistic values are more direction dependent and not uniformly random

        for site_idx, ((da, db), amp) in enumerate(zip(positions_frac, site_amplitudes)):
            for a in range(a_min, a_max + 1):
                for b in range(b_min, b_max + 1):
                    frac = np.array([a + da, b + db])
                    pos_ideal = r0_true + frac[0] * u_true + frac[1] * v_true

                    perturbation = rng.normal(scale=perturb_scale, size=2)
                    x_true, y_true = pos_ideal + perturbation

                    # Clip off any site whose (perturbed) peak would fall
                    # too close to, or outside, the image border.
                    if not (margin <= x_true <= H - margin and margin <= y_true <= W - margin):
                        continue

                    r2 = (rr_grid - x_true) ** 2 + (cc_grid - y_true) ** 2
                    image += amp * np.exp(-0.5 * r2 / gauss_sigma**2)

                    ground_truth[(site_idx, int(a), int(b))] = (x_true, y_true)

        image += rng.normal(scale=0.01, size=image.shape)

        return {
            "image": image,
            "r0_true": r0_true,
            "u_true": u_true,
            "v_true": v_true,
            "positions_frac": positions_frac,
            "ground_truth": ground_truth,
            "margin": margin,
        }

    def test_recovers_known_atom_positions(self, synthetic_lattice_data):
        """
        Fit a lattice on synthetic data and check that refine_atoms() recovers
        the true (perturbed) peak positions to sub-pixel accuracy.
        """
        data = synthetic_lattice_data
        rng = np.random.default_rng(7)
        margin = data["margin"]

        lattice = Lattice.from_data(data["image"], normalize_min=False, normalize_max=False)

        # Deliberately offset the initial guess from the true lattice so that
        # refine_lattice has real work to do, rather than starting exact.
        origin_guess = data["r0_true"] + rng.normal(scale=0.4, size=2)
        u_guess = data["u_true"] + rng.normal(scale=0.3, size=2)
        v_guess = data["v_true"] + rng.normal(scale=0.3, size=2)

        lattice.define_lattice_vectors(
            origin=origin_guess,
            u=u_guess,
            v=v_guess,
            refine_lattice=True,
        )

        lattice.add_atoms(
            data["positions_frac"],
            intensity_radius=3.0,
            edge_min_dist_px=margin,
        )
        lattice.refine_atoms(fit_radius=5.0, max_move_px=margin)

        errors = []
        n_matched = 0
        n_total = 0

        for site_idx, (da, db) in enumerate(data["positions_frac"]):
            site_atoms = lattice.atoms[site_idx]

            if isinstance(site_atoms.array, list) or site_atoms.array.size == 0:
                continue

            # Fetching all the data for all atoms of one site
            # [:,0] simply converts (N,1) shape array to (N,) array
            x_arr = site_atoms.select_fields("x").array[:, 0]
            y_arr = site_atoms.select_fields("y").array[:, 0]
            a_arr = site_atoms.select_fields("a").array[:, 0]
            b_arr = site_atoms.select_fields("b").array[:, 0]

            for x_fit, y_fit, a_val, b_val in zip(x_arr, y_arr, a_arr, b_arr):
                a_int = int(round(a_val - da))
                b_int = int(round(b_val - db))
                key = (site_idx, a_int, b_int)

                n_total += 1
                if key not in data["ground_truth"]:
                    continue

                x_true, y_true = data["ground_truth"][key]
                err = np.hypot(x_fit - x_true, y_fit - y_true)
                errors.append(err)
                n_matched += 1

        errors = np.array(errors)

        # Sanity: we should have matched the large majority of detected atoms
        # back to a known ground-truth site.
        assert n_total > 0
        assert n_matched / n_total >= 0.9

        # Accuracy: refined positions should be within a fraction of a pixel
        # of the true (perturbed) peak centers on average, with no gross outliers.
        assert errors.mean() < 0.3
        assert errors.max() < 1.0


def _grid_lattice_with_atoms() -> Lattice:
    """
    200x200 image with a square grid of Gaussian peaks every 20 px, fitted with a 40 px
    unit cell holding 4 sites, so every site sits on a peak at its ideal lattice position.
    """
    H, W = 200, 200
    image = np.random.randn(H, W) * 0.1
    rr, cc = np.mgrid[0:H, 0:W]
    for x in range(15, H - 15, 20):
        for y in range(15, W - 15, 20):
            image += np.exp(-((rr - x) ** 2 + (cc - y) ** 2) / 20.0)

    lattice = Lattice.from_data(image)
    lattice.define_lattice_vectors(
        origin=[15.0, 15.0], u=[40.0, 0.0], v=[0.0, 40.0], refine_lattice=False
    )
    lattice.add_atoms(np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]))
    return lattice


class TestMeasurePolarization:
    """Test measure_polarization method."""

    @pytest.fixture
    def lattice_with_atoms(self):
        return _grid_lattice_with_atoms()

    def test_returns_self_and_sets_default_plot(self, lattice_with_atoms: Lattice):
        """Test measure_polarization chains, stores a Vector and sets default_plot."""
        result = lattice_with_atoms.measure_polarization(
            measure_ind=0, reference_ind=1, reference_radius=50.0
        )

        assert result is lattice_with_atoms
        assert isinstance(lattice_with_atoms.polarization, Vector)
        assert lattice_with_atoms.default_plot == "polarization"
        assert lattice_with_atoms._pol_meas_ref_ind == (0, 1)

    def test_polarization_fields(self, lattice_with_atoms: Lattice):
        """Test the stored polarization has the expected shape, fields and one row per atom."""
        lattice_with_atoms.measure_polarization(
            measure_ind=0, reference_ind=1, reference_radius=50.0
        )
        pol = lattice_with_atoms.polarization

        assert pol.shape == (1,)
        assert pol.fields == ["x", "y", "a", "b", "da", "db"]
        assert pol[0].array.shape == (lattice_with_atoms.atoms[0].array.shape[0], 6)

    def test_ideal_lattice_has_zero_polarization(self, lattice_with_atoms: Lattice):
        """Atoms placed at ideal lattice positions have zero polarization."""
        lattice_with_atoms.measure_polarization(
            measure_ind=0, reference_ind=1, reference_radius=50.0
        )
        pol = lattice_with_atoms.polarization[0]

        assert np.allclose(pol.select_fields("da").array, 0.0)
        assert np.allclose(pol.select_fields("db").array, 0.0)
        assert lattice_with_atoms._most_common_neighbours.shape[1] == 2

    def test_chaining_after_add_atoms(self):
        """Test measure_polarization can be chained after add_atoms and refine_atoms."""
        lattice = _grid_lattice_with_atoms()
        result = (
            lattice.add_atoms(np.array([[0.0, 0.0], [0.5, 0.0]]))
            .refine_atoms()
            .measure_polarization(measure_ind=0, reference_ind=1, reference_radius=50.0)
        )

        assert result is lattice

    def test_radius_too_small_raises(self, lattice_with_atoms: Lattice):
        """Test a radius that misses neighbours for edge atoms raises."""
        with pytest.raises(ValueError, match=r"Increase the reference_radius"):
            lattice_with_atoms.measure_polarization(
                measure_ind=0, reference_ind=1, reference_radius=30.0
            )

    @pytest.mark.parametrize("min_neighbours,max_neighbours", [(2, 4), (3, 8), (2, 10)])
    def test_knn(self, lattice_with_atoms: Lattice, min_neighbours, max_neighbours):
        """Test polarization measurement with k-nearest neighbours."""
        lattice_with_atoms.measure_polarization(
            measure_ind=0,
            reference_ind=1,
            reference_radius=None,
            min_neighbours=min_neighbours,
            max_neighbours=max_neighbours,
        )

        assert isinstance(lattice_with_atoms.polarization, Vector)
        assert len(lattice_with_atoms._most_common_neighbours) <= max_neighbours

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"reference_radius": 0.5},
            {"reference_radius": None, "min_neighbours": None, "max_neighbours": None},
            {"reference_radius": None, "min_neighbours": 10, "max_neighbours": 5},
            {"reference_radius": None, "min_neighbours": 1, "max_neighbours": 5},
            {"reference_radius": 50.0, "min_neighbours": 4, "max_neighbours": 2},
        ],
    )
    def test_invalid_parameters(self, lattice_with_atoms: Lattice, kwargs):
        """Test invalid neighbour search parameters raise ValueError."""
        with pytest.raises(ValueError):
            lattice_with_atoms.measure_polarization(measure_ind=0, reference_ind=1, **kwargs)

    def test_invalid_site_index(self, lattice_with_atoms: Lattice):
        """Test out of range site indices raise ValueError."""
        with pytest.raises(ValueError, match="measure_ind"):
            lattice_with_atoms.measure_polarization(
                measure_ind=4, reference_ind=1, reference_radius=50.0
            )

    def test_requires_lattice_and_atoms(self):
        """Test measure_polarization raises before define_lattice_vectors / add_atoms."""
        lattice = Lattice.from_data(np.random.randn(100, 100))
        with pytest.raises(ValueError, match="Lattice vectors have not been fitted"):
            lattice.measure_polarization(measure_ind=0, reference_ind=1, reference_radius=50.0)

        lattice.define_lattice_vectors(origin=[50, 50], u=[5, 0], v=[0, 5], refine_lattice=False)
        with pytest.raises(ValueError, match=r"Call add_atoms\(\) first"):
            lattice.measure_polarization(measure_ind=0, reference_ind=1, reference_radius=50.0)

    def test_empty_site(self, lattice_with_atoms: Lattice):
        """Test an empty reference site stores an empty polarization and warns."""
        lattice_with_atoms.atoms[1] = np.zeros((0, lattice_with_atoms.atoms.num_fields))

        with pytest.warns(UserWarning, match="has no atoms"):
            result = lattice_with_atoms.measure_polarization(
                measure_ind=0, reference_ind=1, reference_radius=50.0
            )

        assert result is lattice_with_atoms
        assert lattice_with_atoms.polarization[0].array.shape == (0, 6)
        assert lattice_with_atoms.default_plot == "polarization"


class TestMeasurePolarizationSynGT:
    """
    End-to-end test of add_atoms -> refine_atoms -> measure_polarization against a synthetic
    image where one site is displaced by a known amount from its ideal lattice position.
    """

    def test_recovers_known_polarization(self):
        rng = np.random.default_rng(3)

        H, W = 160, 160
        margin = 6
        r0 = np.array([20.0, 20.0])
        u = np.array([20.0, 0.0])
        v = np.array([0.0, 20.0])
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.5]])
        site_amplitudes = [1.0, 0.6]

        # Site 1 is displaced by +1.5 px along u, i.e. a polarization of (da, db) = (0.075, 0)
        shift_px = np.array([1.5, 0.0])
        da_true = shift_px[0] / np.linalg.norm(u)

        image = np.zeros((H, W))
        rr, cc = np.mgrid[0:H, 0:W]
        for site_idx, ((fa, fb), amp) in enumerate(zip(positions_frac, site_amplitudes)):
            for a in range(-1, 9):
                for b in range(-1, 9):
                    pos = r0 + (a + fa) * u + (b + fb) * v
                    if site_idx == 1:
                        pos = pos + shift_px
                    image += amp * np.exp(
                        -0.5 * ((rr - pos[0]) ** 2 + (cc - pos[1]) ** 2) / 1.5**2
                    )
        image += rng.normal(scale=0.01, size=image.shape)

        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=False)
        lattice.define_lattice_vectors(origin=r0, u=u, v=v, refine_lattice=False)
        lattice.add_atoms(positions_frac, intensity_radius=3.0, edge_min_dist_px=margin)
        lattice.refine_atoms(fit_radius=4.0, max_move_px=3.0)
        lattice.measure_polarization(
            measure_ind=1, reference_ind=0, reference_radius=None, max_neighbours=4
        )

        pol = lattice.polarization[0]
        da = pol.select_fields("da").array[:, 0]
        db = pol.select_fields("db").array[:, 0]

        assert da.size > 0
        assert abs(np.median(da) - da_true) < 0.01
        assert abs(np.median(db)) < 0.01
        assert np.all(np.abs(da - da_true) < 0.03)
        assert np.all(np.abs(db) < 0.03)


class TestPlotPolarization:
    """Test plot(kind='polarization')."""

    @pytest.fixture(autouse=True)
    def close_figures(self):
        yield
        plt.close("all")

    @pytest.fixture
    def lattice_with_polarization(self):
        """Measured lattice with small random polarization values injected for plotting."""
        lattice = _grid_lattice_with_atoms()
        lattice.measure_polarization(measure_ind=0, reference_ind=1, reference_radius=50.0)

        arr = lattice.polarization[0].array.copy()
        arr[:, 4:] = np.random.default_rng(0).normal(scale=0.02, size=(arr.shape[0], 2))
        lattice.polarization[0] = arr
        return lattice

    def test_plot_returns_figure(self, lattice_with_polarization: Lattice):
        """Test plot(kind='polarization') and the default plot return a figure."""
        fig, ax = lattice_with_polarization.plot(kind="polarization", returnfig=True)
        assert isinstance(fig, Figure)

        fig, ax = lattice_with_polarization.plot(returnfig=True)
        assert isinstance(fig, Figure)

        assert lattice_with_polarization.plot(kind="polarization") is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"show_image": False},
            {"show_colorbar": False},
            {"figsize": (6, 6), "title": "custom"},
            {
                "subtract_median": True,
                "show_ref_points": True,
                "adaptive_head": False,
                "phase_dir_flip": True,
                "use_magnitude_lightness": False,
                "length_scale": 2.0,
                "outline_color": "blue",
                "alpha": 0.5,
            },
        ],
    )
    def test_plot_options(self, lattice_with_polarization: Lattice, kwargs):
        """Test plot(kind='polarization') with optional parameters."""
        fig, ax = lattice_with_polarization.plot(kind="polarization", returnfig=True, **kwargs)
        assert isinstance(fig, Figure)

    def test_plot_with_legend(self, lattice_with_polarization: Lattice):
        """Test show_legend=True returns the main, color wheel and legend axes."""
        fig, axs = lattice_with_polarization.plot(
            kind="polarization",
            show_legend=True,
            returnfig=True,
            legend_kwargs={"atom_size": 50},
        )

        assert isinstance(fig, Figure)
        assert len(axs) == 3

    def test_plot_legend_with_figax_raises(self, lattice_with_polarization: Lattice):
        """Test show_legend=True cannot be combined with figax."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="figax"):
            lattice_with_polarization.plot(kind="polarization", show_legend=True, figax=(fig, ax))

    def test_plot_zero_polarization(self):
        """Test plotting an ideal lattice where all polarization vectors are zero."""
        lattice = _grid_lattice_with_atoms()
        lattice.measure_polarization(measure_ind=0, reference_ind=1, reference_radius=50.0)

        fig, axs = lattice.plot(kind="polarization", show_legend=True, returnfig=True)
        assert isinstance(fig, Figure)

    def test_plot_empty_polarization(self, lattice_with_polarization: Lattice):
        """Test plotting with an empty polarization."""
        lattice_with_polarization.polarization[0] = np.zeros((0, 6))

        fig, ax = lattice_with_polarization.plot(kind="polarization", returnfig=True)
        assert isinstance(fig, Figure)

        fig, axs = lattice_with_polarization.plot(
            kind="polarization", show_legend=True, returnfig=True
        )
        assert isinstance(fig, Figure)

    def test_plot_before_measure_raises(self):
        """Test plot(kind='polarization') raises before measure_polarization()."""
        lattice = _grid_lattice_with_atoms()
        with pytest.raises(ValueError, match="measure_polarization"):
            lattice.plot(kind="polarization")


class TestLatticeSerialize:
    """Test Lattice Autoserialize implementation."""

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_lattice_save_load(self, tmp_path, store):
        """Test save/load of lattice."""
        # Create lattice with image and defined lattice
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)
        lattice.define_lattice_vectors(origin=[50, 50], u=[5, 0], v=[0, 5], refine_lattice=False)

        # Save
        filepath = tmp_path / ("lattice.zip" if store == "zip" else "lattice_dir")
        lattice.save(str(filepath), mode="w", store=store)

        # Load
        loaded = load(str(filepath))

        # Verify
        assert isinstance(loaded, Lattice)
        assert isinstance(loaded.image, Dataset2d)
        assert loaded.image.array.shape == lattice.image.array.shape
        assert np.allclose(loaded.image.array, lattice.image.array)
        assert hasattr(loaded, "_lat")
        assert loaded._lat.shape == (3, 2)
        assert np.allclose(loaded._lat, lattice._lat)
