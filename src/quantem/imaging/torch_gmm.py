import numpy as np
import torch


# Implementing GMM using Torch (don't want skimage as a dependency)
class TorchGMM:
    """
    PyTorch Gaussian Mixture Model with full covariances optimized via EM.
    Allows custom means initialization, cov regularization, and device/dtype control.
    After fit, exposes means_, covariances_, and weights_; use predict_proba for probabilities.
    """

    def __init__(
        self,
        n_components,
        covariance_type="full",  # Now support: "full", "diag", "spherical", "tied"
        means_init=None,
        tol=1e-4,
        max_iter=200,
        reg_covar=1e-6,
        device=None,
        dtype=torch.float32,
    ):
        self.n_components = int(n_components)
        self.max_iter = abs(int(max_iter))
        self.covariance_type = covariance_type  # Remove the restriction
        self.means_init = None if means_init is None else np.asarray(means_init, dtype=np.float32)
        self.tol = abs(float(tol))
        self.reg_covar = float(reg_covar)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self._means = None
        self._covariances = None
        self._weights = None

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=self.dtype, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        else:
            return torch.tensor(x, dtype=self.dtype, device=self.device)

    def _kmeans_plusplus_init(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """Initialize means using k-means++ algorithm for better spread."""
        N, D = X.shape

        # Work on CPU for deterministic behavior
        X_cpu = X.cpu()

        # First center: random choice
        indices = [torch.randint(0, N, (1,), device="cpu").item()]

        # Remaining centers: choose based on distance to existing centers
        for _ in range(1, K):
            # Compute distances to nearest existing center
            centers = X_cpu[indices]
            dists = torch.cdist(X_cpu, centers)  # [N, num_centers]
            min_dists = dists.min(dim=1)[0]  # [N]

            # Square distances for probability weighting
            probs = min_dists**2
            probs_sum = probs.sum()

            # Handle case where all points are identical (probs_sum == 0)
            if probs_sum > 1e-10:
                probs = probs / probs_sum
                # Sample next center
                next_idx = torch.multinomial(probs, 1).item()
            else:
                # All points are very close, just pick randomly
                next_idx = torch.randint(0, N, (1,), device="cpu").item()

            indices.append(next_idx)

        return X_cpu[indices].to(device=self.device, dtype=self.dtype)

    def _init_params(self, X: torch.Tensor) -> None:
        N, D = X.shape
        K = self.n_components

        # Initialize means (unchanged)
        if self.means_init is not None:
            if self.means_init.shape != (K, D):
                raise ValueError(
                    f"means_init must have shape ({K}, {D}), got {self.means_init.shape}"
                )
            self._means = self._to_tensor(self.means_init).clone()
        else:
            if N > 0 and K > 0:
                if N >= K:
                    self._means = self._kmeans_plusplus_init(X, K)
                else:
                    X_cpu = X.cpu()
                    indices = torch.randint(0, N, (K,), device="cpu")
                    self._means = X_cpu[indices].clone().to(device=self.device, dtype=self.dtype)
            else:
                self._means = torch.zeros((K, D), device=self.device, dtype=self.dtype)

        # Initialize covariances based on type
        if N > 1:
            X_centered = X - X.mean(dim=0, keepdim=True)
            global_cov = (X_centered.T @ X_centered) / (N - 1)
            global_cov = global_cov + self.reg_covar * torch.eye(
                D, device=self.device, dtype=self.dtype
            )
        else:
            global_cov = self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

        if self.covariance_type == "full":
            self._covariances = global_cov.unsqueeze(0).repeat(K, 1, 1).clone()
        elif self.covariance_type == "diag":
            # Diagonal covariance: [K, D]
            diag_var = torch.diag(global_cov)
            self._covariances = diag_var.unsqueeze(0).repeat(K, 1).clone()
        elif self.covariance_type == "spherical":
            # Spherical covariance: [K] (single variance per component)
            mean_var = torch.diag(global_cov).mean()
            self._covariances = mean_var.unsqueeze(0).repeat(K).clone()
        elif self.covariance_type == "tied":
            # Tied covariance: [D, D] (shared across all components)
            self._covariances = global_cov.clone()
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")

        self._weights = torch.full(
            (K,), 1.0 / K if K > 0 else 1.0, device=self.device, dtype=self.dtype
        )

    def _log_gaussians(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        K = self.n_components
        log_probs = []

        for k in range(K):
            mean_k = self._means[k]

            if self.covariance_type == "full":
                cov_k = self._covariances[k]
                try:
                    dist = torch.distributions.MultivariateNormal(
                        loc=mean_k, covariance_matrix=cov_k, validate_args=False
                    )
                    log_prob = dist.log_prob(X)
                except (RuntimeError, ValueError):
                    cov_reg = cov_k + 1e-3 * torch.eye(D, device=self.device, dtype=self.dtype)
                    dist = torch.distributions.MultivariateNormal(
                        loc=mean_k, covariance_matrix=cov_reg, validate_args=False
                    )
                    log_prob = dist.log_prob(X)

            elif self.covariance_type == "diag":
                # Diagonal covariance
                var_k = self._covariances[k].clamp_min(self.reg_covar)
                diff = X - mean_k
                log_prob = -0.5 * (
                    D * np.log(2 * np.pi) + torch.log(var_k).sum() + ((diff**2) / var_k).sum(dim=1)
                )

            elif self.covariance_type == "spherical":
                # Spherical covariance
                var_k = self._covariances[k].clamp_min(self.reg_covar)
                diff = X - mean_k
                log_prob = -0.5 * (
                    D * np.log(2 * np.pi) + D * torch.log(var_k) + (diff**2).sum(dim=1) / var_k
                )

            elif self.covariance_type == "tied":
                # Tied covariance
                cov = self._covariances
                try:
                    dist = torch.distributions.MultivariateNormal(
                        loc=mean_k, covariance_matrix=cov, validate_args=False
                    )
                    log_prob = dist.log_prob(X)
                except (RuntimeError, ValueError):
                    cov_reg = cov + 1e-3 * torch.eye(D, device=self.device, dtype=self.dtype)
                    dist = torch.distributions.MultivariateNormal(
                        loc=mean_k, covariance_matrix=cov_reg, validate_args=False
                    )
                    log_prob = dist.log_prob(X)

            log_probs.append(log_prob)

        log_comp = torch.stack(log_probs, dim=1)
        return log_comp

    def _e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_comp = self._log_gaussians(X)  # [N, K]
        log_weights = torch.log(self._weights.clamp_min(1e-12))  # [K]
        log_post = log_comp + log_weights[None, :]  # [N, K]
        r = torch.softmax(log_post, dim=1)  # responsibilities [N, K]
        return r, log_post

    def _m_step(self, X: torch.Tensor, r: torch.Tensor) -> None:
        N, D = X.shape
        K = self.n_components
        Nk = r.sum(dim=0).clamp_min(1e-12)
        self._weights = (Nk / N).clamp_min(1e-12)

        # Update means (unchanged)
        self._means = (r.T @ X) / Nk[:, None]

        # Update covariances based on type
        if self.covariance_type == "full":
            covs = []
            for k in range(K):
                diff = X - self._means[k]
                cov_k = (r[:, k][:, None] * diff).T @ diff / Nk[k]
                cov_k = cov_k + self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

                eigenvalues = torch.linalg.eigvalsh(cov_k)
                if eigenvalues.min() < self.reg_covar:
                    cov_k = cov_k + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                        D, device=self.device, dtype=self.dtype
                    )
                covs.append(cov_k)
            self._covariances = torch.stack(covs, dim=0)

        elif self.covariance_type == "diag":
            # Diagonal covariance: only update diagonal elements
            diag_vars = []
            for k in range(K):
                diff = X - self._means[k]
                var_k = (r[:, k][:, None] * (diff**2)).sum(dim=0) / Nk[k]
                var_k = var_k.clamp_min(self.reg_covar)
                diag_vars.append(var_k)
            self._covariances = torch.stack(diag_vars, dim=0)

        elif self.covariance_type == "spherical":
            # Spherical: single variance per component
            spherical_vars = []
            for k in range(K):
                diff = X - self._means[k]
                var_k = (r[:, k][:, None] * (diff**2)).sum() / (Nk[k] * D)
                var_k = var_k.clamp_min(self.reg_covar)
                spherical_vars.append(var_k)
            self._covariances = torch.stack(spherical_vars, dim=0)

        elif self.covariance_type == "tied":
            # Tied: single covariance shared across all components
            cov_tied = torch.zeros((D, D), device=self.device, dtype=self.dtype)
            for k in range(K):
                diff = X - self._means[k]
                cov_tied += (r[:, k][:, None] * diff).T @ diff
            cov_tied = cov_tied / N
            cov_tied = cov_tied + self.reg_covar * torch.eye(
                D, device=self.device, dtype=self.dtype
            )

            eigenvalues = torch.linalg.eigvalsh(cov_tied)
            if eigenvalues.min() < self.reg_covar:
                cov_tied = cov_tied + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                    D, device=self.device, dtype=self.dtype
                )
            self._covariances = cov_tied

    def _convert_covariances_to_full(self) -> np.ndarray:
        """
        Convert internal covariance representation to full [K, D, D] format.
        This ensures compatibility with plotting functions that expect full covariance matrices.
        """
        K = self.n_components

        if self._covariances is None:
            raise ValueError("Model has not been fitted yet.")

        D = self._means.shape[1]

        if self.covariance_type == "full":
            return self._covariances.detach().clone().cpu().numpy()

        elif self.covariance_type == "diag":
            covs_full = []
            for k in range(K):
                cov_k = torch.diag(self._covariances[k])
                covs_full.append(cov_k)
            return torch.stack(covs_full, dim=0).detach().cpu().numpy()

        elif self.covariance_type == "spherical":
            covs_full = []
            for k in range(K):
                cov_k = self._covariances[k] * torch.eye(D, device=self.device, dtype=self.dtype)
                covs_full.append(cov_k)
            return torch.stack(covs_full, dim=0).detach().cpu().numpy()

        elif self.covariance_type == "tied":
            cov_shared = self._covariances.unsqueeze(0).repeat(K, 1, 1)
            return cov_shared.detach().cpu().numpy()

    def fit(self, data) -> "TorchGMM":
        X = self._to_tensor(data)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D with shape (N, D)")

        self._init_params(X)

        prev_ll = torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)

        for iteration in range(self.max_iter):
            r, _ = self._e_step(X)
            self._m_step(X, r)

            # Compute average log-likelihood of data under mixture
            log_comp = self._log_gaussians(X)
            log_weighted = log_comp + torch.log(self._weights)[None, :]
            ll = torch.logsumexp(log_weighted, dim=1).mean()

            # Check convergence
            if iteration > 0 and torch.isfinite(prev_ll) and torch.isfinite(ll):
                improvement = (ll - prev_ll).abs()
                if improvement < self.tol:
                    break
            prev_ll = ll

        # Store NumPy copies for external use (decoupled from internal tensors)
        self.means_ = self._means.detach().clone().cpu().numpy()
        self.covariances_ = self._convert_covariances_to_full()  # Changed this line
        self.weights_ = self._weights.detach().clone().cpu().numpy()

        return self

    def predict_proba(self, data) -> np.ndarray:
        X = self._to_tensor(data)
        r, _ = self._e_step(X)
        return r.detach().cpu().numpy()


class FixedMeansGMM(TorchGMM):
    """
    GMM variant with fixed component means.
    Means are set via fixed_means at init and held constant during EM;
    only weights and covariances are updated.
    """

    def __init__(self, fixed_means, **kwargs):
        fixed_means = np.asarray(fixed_means, dtype=np.float32)
        super().__init__(n_components=len(fixed_means), means_init=fixed_means, **kwargs)
        self.fixed_means = fixed_means

    def _m_step(self, X, r):
        """
        M-step with fixed means:
        update mixture weights and covariances from responsibilities,
        keeping means unchanged.
        """
        N, D = X.shape
        K = self.n_components
        Nk = r.sum(dim=0).clamp_min(1e-12)
        self._weights = (Nk / N).clamp_min(1e-12)

        # Keep means fixed
        self._means = self._to_tensor(self.fixed_means).clone()

        # Update covariances based on type
        if self.covariance_type == "full":
            covs = []
            for k in range(K):
                diff = X - self._means[k]
                cov_k = (r[:, k][:, None] * diff).T @ diff / Nk[k]
                cov_k = cov_k + self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

                eigenvalues = torch.linalg.eigvalsh(cov_k)
                if eigenvalues.min() < self.reg_covar:
                    cov_k = cov_k + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                        D, device=self.device, dtype=self.dtype
                    )
                covs.append(cov_k)
            self._covariances = torch.stack(covs, dim=0)

        elif self.covariance_type == "diag":
            diag_vars = []
            for k in range(K):
                diff = X - self._means[k]
                var_k = (r[:, k][:, None] * (diff**2)).sum(dim=0) / Nk[k]
                var_k = var_k.clamp_min(self.reg_covar)
                diag_vars.append(var_k)
            self._covariances = torch.stack(diag_vars, dim=0)

        elif self.covariance_type == "spherical":
            spherical_vars = []
            for k in range(K):
                diff = X - self._means[k]
                var_k = (r[:, k][:, None] * (diff**2)).sum() / (Nk[k] * D)
                var_k = var_k.clamp_min(self.reg_covar)
                spherical_vars.append(var_k)
            self._covariances = torch.stack(spherical_vars, dim=0)

        elif self.covariance_type == "tied":
            cov_tied = torch.zeros((D, D), device=self.device, dtype=self.dtype)
            for k in range(K):
                diff = X - self._means[k]
                cov_tied += (r[:, k][:, None] * diff).T @ diff
            cov_tied = cov_tied / N
            cov_tied = cov_tied + self.reg_covar * torch.eye(
                D, device=self.device, dtype=self.dtype
            )

            eigenvalues = torch.linalg.eigvalsh(cov_tied)
            if eigenvalues.min() < self.reg_covar:
                cov_tied = cov_tied + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                    D, device=self.device, dtype=self.dtype
                )
            self._covariances = cov_tied


class DirectionalGMM(TorchGMM):
    """
    GMM variant that prioritizes directional clustering for 2D vectors.

    Extends TorchGMM with a magnitude-gated von Mises-Fisher-like directional
    likelihood, blended with the standard spatial Gaussian likelihood. Vectors
    with small magnitudes are down-weighted in the directional signal to prevent
    noise amplification from normalization of near-zero vectors.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    directional_weight : float
        Blend weight in [0, 1] between directional (1.0) and spatial (0.0) likelihood.
    concentration : float
        Sharpness of directional clustering; higher values produce tighter angular clusters.
    covariance_type : str
        One of 'full', 'diag', 'spherical', 'tied'.
    magnitude_threshold : float or None
        Vectors with norm below this are treated as directionless. If None,
        estimated automatically as the 25th percentile of norms in the training data.
    magnitude_sharpness : float
        Controls the steepness of the sigmoid magnitude gate around the threshold.
    """

    def __init__(
        self,
        n_components,
        directional_weight=0.99,
        concentration=7.0,
        covariance_type="full",
        magnitude_threshold=None,
        magnitude_sharpness=5.0,
        **kwargs,
    ):
        super().__init__(n_components, covariance_type=covariance_type, **kwargs)
        self.directional_weight = directional_weight
        self.concentration = concentration
        self.magnitude_threshold = magnitude_threshold
        self.magnitude_sharpness = magnitude_sharpness
        self._principal_directions = None
        self._magnitude_scale = None

    def _magnitude_confidence(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute per-vector confidence weights in [0, 1] based on magnitude.

        Uses a sigmoid gate centered at the magnitude threshold so that
        near-zero vectors contribute minimally to directional decisions.

        Returns
        -------
        torch.Tensor of shape [N, 1]
        """
        norms = torch.norm(X, dim=1)
        confidence = torch.sigmoid(
            self.magnitude_sharpness * (norms / (self._magnitude_scale + 1e-12) - 1.0)
        )
        return confidence.unsqueeze(1)

    def _normalize_vectors(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize rows of X to unit length. Safe for near-zero vectors."""
        norms = torch.norm(X, dim=1, keepdim=True).clamp_min(1e-10)
        return X / norms

    def _estimate_magnitude_scale(self, X: torch.Tensor) -> None:
        """
        Set the magnitude threshold from data.

        Uses the 25th percentile of vector norms so the smallest quarter of
        vectors are treated as effectively directionless by default.
        Can be overridden by setting magnitude_threshold explicitly.
        """
        if self.magnitude_threshold is not None:
            self._magnitude_scale = torch.tensor(
                self.magnitude_threshold, device=self.device, dtype=self.dtype
            )
        else:
            norms = torch.norm(X, dim=1)
            self._magnitude_scale = torch.quantile(norms, 0.25).clamp_min(1e-10)

    def _init_directional_means(self, X: torch.Tensor) -> None:
        """
        Initialize component means and principal directions.

        If means_init is not provided, evenly partitions angular space into K
        sectors and seeds each mean as the magnitude-confidence-weighted centroid
        of points most aligned with that sector's direction.
        """
        N, D = X.shape
        K = self.n_components

        if D != 2:
            raise ValueError("DirectionalGMM currently only supports 2D vectors")

        self._estimate_magnitude_scale(X)

        X_normalized = self._normalize_vectors(X)
        conf = self._magnitude_confidence(X)

        if self.means_init is not None:
            self._means = self._to_tensor(self.means_init).clone()
            self._principal_directions = self._normalize_vectors(self._means)
        else:
            angles = torch.linspace(0, 2 * np.pi, K + 1, device=self.device)[:-1]
            self._principal_directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

            means_list = []
            for k in range(K):
                direction = self._principal_directions[k]
                alignment = (X_normalized @ direction).squeeze()
                # Weight by both alignment and magnitude confidence to avoid
                # noisy near-zero vectors dominating the initial mean estimate
                combined_weight = torch.softmax(self.concentration * alignment * conf[:, 0], dim=0)
                means_list.append((combined_weight[:, None] * X).sum(dim=0))

            self._means = torch.stack(means_list, dim=0)

        self._init_directional_covariances(X)

    def _init_directional_covariances(self, X: torch.Tensor) -> None:
        """
        Initialize covariances with a directional prior.

        For 'full' type, each component is initialized as an ellipse elongated
        along its principal direction. Other types use standard initializations.
        """
        N, D = X.shape
        K = self.n_components

        if self.covariance_type == "full":
            covs = []
            for k in range(K):
                direction = self._principal_directions[k]
                perp = torch.tensor(
                    [-direction[1], direction[0]], device=self.device, dtype=self.dtype
                )
                basis = torch.stack([direction, perp], dim=1)
                # Larger variance along principal direction, smaller perpendicular
                eigenvalues = torch.tensor([1.0, 0.1], device=self.device, dtype=self.dtype)
                cov_k = basis @ torch.diag(eigenvalues) @ basis.T
                cov_k = cov_k + self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)
                covs.append(cov_k)
            self._covariances = torch.stack(covs, dim=0)

        elif self.covariance_type == "diag":
            diag_vars = []
            for k in range(K):
                direction = self._principal_directions[k]
                var_k = (torch.abs(direction) * 0.5 + 0.1).clamp_min(self.reg_covar)
                diag_vars.append(var_k)
            self._covariances = torch.stack(diag_vars, dim=0)

        elif self.covariance_type == "spherical":
            self._covariances = torch.full(
                (K,), 0.3, device=self.device, dtype=self.dtype
            ).clamp_min(self.reg_covar)

        elif self.covariance_type == "tied":
            if N > 1:
                X_centered = X - X.mean(dim=0, keepdim=True)
                global_cov = (X_centered.T @ X_centered) / (N - 1)
            else:
                global_cov = torch.eye(D, device=self.device, dtype=self.dtype)
            self._covariances = global_cov + self.reg_covar * torch.eye(
                D, device=self.device, dtype=self.dtype
            )

        self._weights = torch.full((K,), 1.0 / K, device=self.device, dtype=self.dtype)

    def _directional_log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude-gated von Mises-Fisher-like log likelihood.

        The concentration is scaled by per-vector magnitude confidence,
        so near-zero vectors produce a near-uniform directional distribution
        rather than a spuriously strong directional signal.

        Returns
        -------
        torch.Tensor of shape [N, K]
        """
        X_normalized = self._normalize_vectors(X)
        conf = self._magnitude_confidence(X)
        alignment = X_normalized @ self._principal_directions.T
        return self.concentration * conf * alignment

    def _e_step(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        E-step combining spatial Gaussian and directional likelihoods.

        The directional_weight parameter controls the blend:
        - 1.0 → purely directional
        - 0.0 → purely spatial (equivalent to standard GMM)
        """
        log_spatial = self._log_gaussians(X)
        log_directional = self._directional_log_likelihood(X)
        log_combined = (
            1 - self.directional_weight
        ) * log_spatial + self.directional_weight * log_directional
        log_weights = torch.log(self._weights.clamp_min(1e-12))
        log_post = log_combined + log_weights[None, :]
        r = torch.softmax(log_post, dim=1)
        return r, log_post

    def _m_step(self, X: torch.Tensor, r: torch.Tensor) -> None:
        """
        M-step updating weights, means, principal directions, and covariances.

        Principal directions are updated as the magnitude-confidence-weighted
        mean of assigned unit vectors, so near-zero vectors do not corrupt
        the directional estimate for a component.
        """
        N, D = X.shape
        K = self.n_components

        Nk = r.sum(dim=0).clamp_min(1e-12)
        self._weights = (Nk / N).clamp_min(1e-12)
        self._means = (r.T @ X) / Nk[:, None]

        X_normalized = self._normalize_vectors(X)
        conf = self._magnitude_confidence(X)[:, 0]

        for k in range(K):
            # Weight directional votes by responsibility and magnitude confidence
            combined_weight = r[:, k] * conf
            weighted_dir = (combined_weight[:, None] * X_normalized).sum(dim=0)
            norm = weighted_dir.norm()
            if norm > 1e-10:
                self._principal_directions[k] = weighted_dir / norm

        if self.covariance_type == "full":
            covs = []
            for k in range(K):
                diff = X - self._means[k]
                cov_k = (r[:, k][:, None] * diff).T @ diff / Nk[k]

                direction = self._principal_directions[k]
                perp = torch.tensor(
                    [-direction[1], direction[0]], device=self.device, dtype=self.dtype
                )
                basis = torch.stack([direction, perp], dim=1)

                # Project into directional frame to enforce anisotropy
                cov_rotated = basis.T @ cov_k @ basis
                diag_vals = torch.diagonal(cov_rotated)
                if diag_vals[1] > diag_vals[0] * 0.5:
                    cov_rotated[1, 1] = diag_vals[0] * 0.3

                cov_k = basis @ cov_rotated @ basis.T
                cov_k = cov_k + self.reg_covar * torch.eye(D, device=self.device, dtype=self.dtype)

                eigenvalues = torch.linalg.eigvalsh(cov_k)
                if eigenvalues.min() < self.reg_covar:
                    cov_k = cov_k + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                        D, device=self.device, dtype=self.dtype
                    )
                covs.append(cov_k)
            self._covariances = torch.stack(covs, dim=0)

        elif self.covariance_type == "diag":
            diag_vars = []
            for k in range(K):
                diff = X - self._means[k]
                var_k = (r[:, k][:, None] * (diff**2)).sum(dim=0) / Nk[k]
                direction = self._principal_directions[k]
                var_k = var_k * (1 + 0.2 * torch.abs(direction))
                diag_vars.append(var_k.clamp_min(self.reg_covar))
            self._covariances = torch.stack(diag_vars, dim=0)

        elif self.covariance_type == "spherical":
            spherical_vars = []
            for k in range(K):
                diff = X - self._means[k]
                var_k = (r[:, k][:, None] * (diff**2)).sum() / (Nk[k] * D)
                spherical_vars.append(var_k.clamp_min(self.reg_covar))
            self._covariances = torch.stack(spherical_vars, dim=0)

        elif self.covariance_type == "tied":
            cov_tied = torch.zeros((D, D), device=self.device, dtype=self.dtype)
            for k in range(K):
                diff = X - self._means[k]
                cov_tied += (r[:, k][:, None] * diff).T @ diff
            cov_tied = cov_tied / N + self.reg_covar * torch.eye(
                D, device=self.device, dtype=self.dtype
            )
            eigenvalues = torch.linalg.eigvalsh(cov_tied)
            if eigenvalues.min() < self.reg_covar:
                cov_tied = cov_tied + (self.reg_covar - eigenvalues.min() + 1e-6) * torch.eye(
                    D, device=self.device, dtype=self.dtype
                )
            self._covariances = cov_tied

    def fit(self, data) -> "DirectionalGMM":
        """
        Fit the DirectionalGMM to 2D vector data via EM.

        Parameters
        ----------
        data : array-like of shape (N, 2)

        Returns
        -------
        self
        """
        X = self._to_tensor(data)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("DirectionalGMM requires 2D input data")

        self._init_directional_means(X)

        prev_ll = torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)

        for iteration in range(self.max_iter):
            r, log_post = self._e_step(X)
            self._m_step(X, r)
            ll = log_post.logsumexp(dim=1).mean()
            if iteration > 0 and torch.isfinite(prev_ll) and torch.isfinite(ll):
                if (ll - prev_ll).abs() < self.tol:
                    break
            prev_ll = ll

        self.means_ = self._means.detach().clone().cpu().numpy()
        self.covariances_ = self._convert_covariances_to_full()
        self.weights_ = self._weights.detach().clone().cpu().numpy()
        self.principal_directions_ = self._principal_directions.detach().clone().cpu().numpy()

        return self
