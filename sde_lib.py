"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import pdb
import torch
import numpy as np


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N
        self._shape = None

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        _shape = self._shape
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
                self._shape = _shape

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                if self._shape is None:
                    self._shape = x.dim()

                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - self._unsqueeze(diffusion) ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                if self._shape is None:
                    self._shape = x.dim()

                f, G = discretize_fn(x, t)
                rev_f = f - self._unsqueeze(G) ** 2 * score_fn(x, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    def _unsqueeze(self, x):
        # Add empty axes to match dims of batch
        assert self._shape is not None
        if x.dim() != self._shape:
            for i in range(self._shape - 1):
                x = torch.unsqueeze(x, axis=-1)
        return x


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        if self._shape is None:
            self._shape = x.dim()
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        # Add empty axes to match dims
        # for i in range(x.dim() - 1):
        #     beta_t = torch.unsqueeze(beta_t, axis=-1)
        # drift = -0.5 * beta_t[:, None, None, None] * x
        drift = -0.5 * self._unsqueeze(beta_t) * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        if self._shape is None:
            self._shape = x.dim()
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )

        # mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        mean = torch.exp(self._unsqueeze(log_mean_coeff)) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        if self._shape is None:
            self._shape = x.dim()

        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        # sqrt_alpha = torch.sqrt(alpha)[:, None, None, None]
        sqrt_alpha = self._unsqueeze(torch.sqrt(alpha))
        f = sqrt_alpha * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        if self._shape is None:
            self._shape = x.dim()
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * self._unsqueeze(beta_t) * x
        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        if self._shape is None:
            self._shape = x.dim()

        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = self._unsqueeze(torch.exp(log_mean_coeff)) * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

    def noise_schedule_inverse(self, sigma):
        """
        Returns the timepoint at which the sigma is observed
        according to the subVPSDE schedule
        This is simply the solution obtained via the quadratic formula for the
        std calculation for a subVPSDE (b24ac => b^2 - 4ac)
        """
        b = self.beta_0
        b24ac = b**2 - 2 * (self.beta_1 - self.beta_0) * torch.log(1 - sigma + 1e-12)
        t = (-b + b24ac**0.5) / (self.beta_1 - self.beta_0)
        return torch.clip(t, max=1.0)


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        ).float()
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        if self._shape is None:
            self._shape = x.dim()
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        if self._shape is None:
            self._shape = x.dim()
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape, dtype=torch.float32) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        self.discrete_sigmas = self.discrete_sigmas.to(t.device)
        sigma = self.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.discrete_sigmas[timestep - 1].to(t.device),
        )
        f = torch.zeros_like(x, dtype=torch.float32)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        # pdb.set_trace()
        return f, G
