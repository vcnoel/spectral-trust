# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""
Seeded generators of the sequence families of numerical in-context learning.

The families of Bao et al. (2026): :func:`periodic` (a period-p pattern),
:func:`logistic` (the logistic map at r, periodic or chaotic) and
:func:`lorenz` (one coordinate of the Lorenz system). Smooth families, whose
dynamics are simple but continuous: :func:`sine`, :func:`damped_oscillator`
and :func:`pendulum` (integrated with the symplectic Stormer-Verlet scheme,
so its energy stays bounded over long runs).

Every generator returns a :class:`Trajectory`: the raw values and the
integers of the ``000``-``999`` format (rescaled to ``[50, 950]`` with the
bounds of the whole run, as :func:`~spectral_trust.numeric.quantize` does),
so ``serialize(traj.ints[:n])`` is a prompt and ``traj.ints[n:]`` its true
continuation. The same ``seed`` gives the same trajectory; it draws the
initial condition (or the phase), never the parameters.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Union

import numpy as np

from .serialize import quantize, serialize


@dataclass
class Trajectory:
    """One realisation of a family.

    Attributes:
        values: ``[n]`` raw values.
        ints: ``[n]`` integers in ``[low, high]`` (default ``[50, 950]``).
        family: the generator's name.
        params: its parameters, the seed included.
    """
    values: np.ndarray
    ints: np.ndarray
    family: str
    params: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ints)

    def text(self, n=None) -> str:
        """The first ``n`` values (all by default) in the ``"087,173,"`` format."""
        return serialize(self.ints[:n])

    def split(self, n_context: int, n_steps=None):
        """``(context, continuation)``: the first ``n_context`` integers and the
        ``n_steps`` that follow (all that follow by default)."""
        end = None if n_steps is None else n_context + int(n_steps)
        return self.ints[:n_context], self.ints[n_context:end]


def _traj(values, family, params, low, high) -> Trajectory:
    values = np.asarray(values, dtype=np.float64)
    return Trajectory(values, quantize(values, low, high), family, dict(params))


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


def periodic(p: int, n: int = 1000, seed: int = 0, *, low: int = 50,
             high: int = 950) -> Trajectory:
    """A period-``p`` sequence: ``p`` values drawn uniformly in ``[0, 1)``
    (distinct for ``p >= 2``) and repeated. ``p = 1`` is a constant."""
    p = int(p)
    if not 1 <= p <= high - low + 1:
        raise ValueError(f"p must lie in [1, {high - low + 1}]")
    rng = _rng(seed)
    while True:
        pattern = rng.uniform(0.0, 1.0, p)
        if p == 1 or len(np.unique(quantize(pattern, low, high))) == p:
            break
    return _traj(np.resize(pattern, int(n)), "periodic", {"p": p, "seed": seed}, low, high)


def logistic(r: float, n: int = 1000, seed: int = 0, *, burn_in: int = 0,
             low: int = 50, high: int = 950) -> Trajectory:
    """The logistic map ``x <- r x (1 - x)`` from ``x0`` uniform in
    ``[0.3, 0.7]``, after ``burn_in`` discarded steps (pass a few thousand
    to start on the attractor). After burn-in, r = 3.2, 3.5, 3.55 and 3.566
    give periods 2, 4, 8 and 16; r = 3.9 and 4.0 are chaotic."""
    x = _rng(seed).uniform(0.3, 0.7)
    x0 = x
    for _ in range(int(burn_in)):
        x = r * x * (1.0 - x)
    out = np.empty(int(n))
    for t in range(int(n)):
        out[t] = x
        x = r * x * (1.0 - x)
    return _traj(out, "logistic", {"r": float(r), "x0": x0, "burn_in": int(burn_in),
                                   "seed": seed}, low, high)


_COORD = {"x": 0, "y": 1, "z": 2}


def _rk4(f, y, h):
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz(n: int = 1000, seed: int = 0, *, coord: Union[str, int] = "x", dt: float = 0.02,
           substeps: int = 4, burn_in: int = 500, sigma: float = 10.0, rho: float = 28.0,
           beta: float = 8.0 / 3.0, low: int = 50, high: int = 950) -> Trajectory:
    """One coordinate (``"x"``, ``"y"``, ``"z"`` or 0, 1, 2) of the Lorenz
    system, sampled every ``dt`` (RK4 with ``substeps`` steps per sample),
    from a seeded initial condition near the attractor after ``burn_in``
    discarded samples."""
    c = _COORD.get(coord, coord) if isinstance(coord, str) else int(coord)
    if c not in (0, 1, 2):
        raise ValueError(f"coord must be 'x', 'y', 'z', 0, 1 or 2, got {coord!r}")

    def f(s):
        return np.array([sigma * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1],
                         s[0] * s[1] - beta * s[2]])

    y = np.array([1.0, 1.0, 20.0]) + _rng(seed).normal(0.0, 1.0, 3)
    h = float(dt) / int(substeps)
    out = np.empty((int(n), 3))
    for t in range(int(burn_in) + int(n)):
        if t >= burn_in:
            out[t - burn_in] = y
        for _ in range(int(substeps)):
            y = _rk4(f, y, h)
    return _traj(out[:, c], "lorenz", {"coord": "xyz"[c], "dt": float(dt), "substeps": substeps,
                                       "burn_in": burn_in, "sigma": sigma, "rho": rho,
                                       "beta": beta, "seed": seed}, low, high)


def sine(n: int = 1000, seed: int = 0, *, frequency: float = 1.0 / 32, amplitude: float = 1.0,
         phase=None, low: int = 50, high: int = 950) -> Trajectory:
    """``amplitude sin(2 pi frequency t + phase)``, ``frequency`` in cycles
    per value (1/32: one period every 32 values); the phase is drawn from
    ``seed`` unless given."""
    ph = _rng(seed).uniform(0.0, 2 * np.pi) if phase is None else float(phase)
    t = np.arange(int(n), dtype=np.float64)
    return _traj(amplitude * np.sin(2 * np.pi * float(frequency) * t + ph), "sine",
                 {"frequency": float(frequency), "amplitude": float(amplitude), "phase": ph,
                  "seed": seed}, low, high)


def damped_oscillator(n: int = 1000, seed: int = 0, *, frequency: float = 1.0 / 32,
                      damping: float = 0.01, phase=None, low: int = 50,
                      high: int = 950) -> Trajectory:
    """The position of ``x'' + 2 zeta w x' + w^2 x = 0`` (``w = 2 pi
    frequency``, ``zeta = damping`` < 1), in closed form:
    ``exp(-zeta w t) cos(w_d t + phase)`` with ``w_d = w sqrt(1 - zeta^2)``;
    one time unit per value."""
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must lie in [0, 1) (underdamped)")
    ph = _rng(seed).uniform(0.0, 2 * np.pi) if phase is None else float(phase)
    w = 2 * np.pi * float(frequency)
    wd = w * np.sqrt(1.0 - damping ** 2)
    t = np.arange(int(n), dtype=np.float64)
    return _traj(np.exp(-damping * w * t) * np.cos(wd * t + ph), "damped_oscillator",
                 {"frequency": float(frequency), "damping": float(damping), "phase": ph,
                  "seed": seed}, low, high)


def pendulum(n: int = 1000, seed: int = 0, *, theta0=None, omega: float = 2 * np.pi / 32,
             dt: float = 1.0, substeps: int = 8, coord: str = "theta", low: int = 50,
             high: int = 950) -> Trajectory:
    """A frictionless pendulum ``theta'' = -omega^2 sin(theta)``, a simple
    Hamiltonian system ``H = p^2 / 2 + omega^2 (1 - cos theta)``,
    integrated with the symplectic Stormer-Verlet (leapfrog) scheme,
    ``substeps`` steps per sample ``dt``. Starts at rest at ``theta0``
    (radians; drawn uniformly in ``[0.5, 2.5]`` from ``seed`` unless given:
    large swings, so not a sine). ``coord``: ``"theta"`` or ``"momentum"``.
    ``params["energy_drift"]`` is the largest relative energy error."""
    if coord not in ("theta", "momentum"):
        raise ValueError("coord must be 'theta' or 'momentum'")
    th = _rng(seed).uniform(0.5, 2.5) if theta0 is None else float(theta0)
    th0, p, w2 = th, 0.0, float(omega) ** 2
    h = float(dt) / int(substeps)
    out = np.empty((int(n), 2))
    for t in range(int(n)):
        out[t] = th, p
        for _ in range(int(substeps)):
            p -= 0.5 * h * w2 * np.sin(th)
            th += h * p
            p -= 0.5 * h * w2 * np.sin(th)
    energy = 0.5 * out[:, 1] ** 2 + w2 * (1.0 - np.cos(out[:, 0]))
    drift = float(np.max(np.abs(energy - energy[0])) / max(abs(energy[0]), 1e-300))
    vals = out[:, 0] if coord == "theta" else out[:, 1]
    return _traj(vals, "pendulum", {"theta0": th0, "omega": float(omega), "dt": float(dt),
                                    "substeps": int(substeps), "coord": coord,
                                    "energy_drift": drift, "seed": seed}, low, high)


#: Every generator by name, for :func:`generate`.
FAMILIES: Dict[str, Callable[..., Trajectory]] = {
    "periodic": periodic, "logistic": logistic, "lorenz": lorenz, "sine": sine,
    "damped_oscillator": damped_oscillator, "pendulum": pendulum,
}


def generate(family: str, n: int = 1000, seed: int = 0, **params) -> Trajectory:
    """``FAMILIES[family](n=n, seed=seed, **params)``, e.g.
    ``generate("logistic", 200, seed=3, r=3.9, burn_in=1000)``."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; use one of {sorted(FAMILIES)}")
    return FAMILIES[family](n=n, seed=seed, **params)
