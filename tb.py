"""Three-band tight-binding model of monolayer 1H-NbSe2.

Orbitals (d_z2, d_xy, d_x2-y2), third-nearest-neighbour hopping in the form of Liu et al., Phys. Rev. B 88, 085433
(2013), with the NbSe2 parameters of He et al., Commun. Phys. 1, 40 (2018), and Ising spin-orbit coupling
(lambda/2) L_z s_z. Energies in eV, momenta in 1/Angstrom, azimuth phi measured from Gamma-K.
"""
import numpy as np

A = 3.44                                    # lattice constant (Angstrom)
S3 = np.sqrt(3.0)
P = dict(e1=1.4466, e2=1.8496, t0=-0.2308, t1=0.3116, t2=0.3459, t11=0.2795, t12=0.2787, t22=-0.0539,
         r0=0.0037, r1=-0.0997, r2=0.0385, r11=0.0320, r12=0.0986,
         u0=0.0685, u1=-0.0381, u2=0.0535, u11=0.0601, u12=-0.0179, u22=-0.0425, lam=0.0784)
KPT = np.array([4 * np.pi / (3 * A), 0.0])  # K point
LZ = np.array([[0, 0, 0], [0, 0, 2j], [0, -2j, 0]])


def H0(kx, ky, p=P):
    a = 0.5 * kx * A
    b = 0.5 * S3 * ky * A
    c, s = np.cos, np.sin
    h0 = (p['e1'] + 2 * p['t0'] * (2 * c(a) * c(b) + c(2 * a)) + 2 * p['r0'] * (2 * c(3 * a) * c(b) + c(2 * b))
          + 2 * p['u0'] * (2 * c(2 * a) * c(2 * b) + c(4 * a)))
    re1 = -2 * S3 * p['t2'] * s(a) * s(b) + 2 * (p['r1'] + p['r2']) * s(3 * a) * s(b) - 2 * S3 * p['u2'] * s(2 * a) * s(2 * b)
    im1 = (2 * p['t1'] * s(a) * (2 * c(a) + c(b)) + 2 * (p['r1'] - p['r2']) * s(3 * a) * c(b)
           + 2 * p['u1'] * s(2 * a) * (2 * c(2 * a) + c(2 * b)))
    re2 = (2 * p['t2'] * (c(2 * a) - c(a) * c(b)) - (2 / S3) * (p['r1'] + p['r2']) * (c(3 * a) * c(b) - c(2 * b))
           + 2 * p['u2'] * (c(4 * a) - c(2 * a) * c(2 * b)))
    im2 = (2 * S3 * p['t1'] * c(a) * s(b) + (2 / S3) * s(b) * (p['r1'] - p['r2']) * (c(3 * a) + 2 * c(b))
           + 2 * S3 * p['u1'] * c(2 * a) * s(2 * b))
    h11 = (p['e2'] + (p['t11'] + 3 * p['t22']) * c(a) * c(b) + 2 * p['t11'] * c(2 * a) + 4 * p['r11'] * c(3 * a) * c(b)
           + 2 * (p['r11'] + S3 * p['r12']) * c(2 * b) + (p['u11'] + 3 * p['u22']) * c(2 * a) * c(2 * b) + 2 * p['u11'] * c(4 * a))
    re12 = S3 * (p['t22'] - p['t11']) * s(a) * s(b) + 4 * p['r12'] * s(3 * a) * s(b) + S3 * (p['u22'] - p['u11']) * s(2 * a) * s(2 * b)
    im12 = 4 * p['t12'] * s(a) * (c(a) - c(b)) + 4 * p['u12'] * s(2 * a) * (c(2 * a) - c(2 * b))
    h22 = (p['e2'] + (3 * p['t11'] + p['t22']) * c(a) * c(b) + 2 * p['t22'] * c(2 * a) + 2 * p['r11'] * (2 * c(3 * a) * c(b) + c(2 * b))
           + (2 / S3) * p['r12'] * (4 * c(3 * a) * c(b) - c(2 * b)) + (3 * p['u11'] + p['u22']) * c(2 * a) * c(2 * b)
           + 2 * p['u22'] * c(4 * a))
    H = np.zeros(np.shape(kx) + (3, 3), complex)
    H[..., 0, 0], H[..., 1, 1], H[..., 2, 2] = h0, h11, h22
    H[..., 0, 1], H[..., 1, 0] = re1 + 1j * im1, re1 - 1j * im1
    H[..., 0, 2], H[..., 2, 0] = re2 + 1j * im2, re2 - 1j * im2
    H[..., 1, 2], H[..., 2, 1] = re12 + 1j * im12, re12 - 1j * im12
    return H


def H(kx, ky, s, lam=1.0, p=P):
    """Bloch Hamiltonian for spin s = +1/-1; lam scales the Ising coupling (lam = 0: no spin-orbit coupling)."""
    return H0(kx, ky, p) + s * 0.5 * p['lam'] * lam * LZ


def lowest(k, s, lam=1.0):
    """energy and orbital vector of the lowest band at momenta k[..., 2]."""
    E, U = np.linalg.eigh(H(k[..., 0], k[..., 1], s, lam))
    return E[..., 0], U[..., :, 0]


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def grad(p, s, lam=1.0, h=1e-5):
    g = np.zeros(p.shape)
    for i in range(2):
        dp = np.zeros(2)
        dp[i] = h
        g[..., i] = (lowest(p + dp, s, lam)[0] - lowest(p - dp, s, lam)[0]) / (2 * h)
    return g


def fermi_contour(center, s, mu, n, rmax, lam=1.0, nr=200, chunk=2000):
    """Fermi contour of the lowest band around `center` on n rays: points, unit normals, speed |grad E| (eV A) and
    arc-length weights (1/A)."""
    ph = (np.arange(n) + 0.5) / n * 2 * np.pi
    d = np.stack([np.cos(ph), np.sin(ph)], -1)
    r = np.linspace(1e-4, rmax, nr)
    lo, hi = np.empty(n), np.empty(n)
    for c0 in range(0, n, chunk):
        dd = d[c0:c0 + chunk]
        e = lowest(center[None, None] + r[None, :, None] * dd[:, None], s, lam)[0] - mu
        sg = np.sign(e)
        first = np.argmax(sg[:, 1:] != sg[:, :1], axis=1)
        lo[c0:c0 + chunk], hi[c0:c0 + chunk] = r[first], r[first + 1]
    elo = lowest(center[None] + lo[:, None] * d, s, lam)[0] - mu
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        em = lowest(center[None] + mid[:, None] * d, s, lam)[0] - mu
        same = np.sign(em) == np.sign(elo)
        lo, hi, elo = np.where(same, mid, lo), np.where(same, hi, mid), np.where(same, em, elo)
    pF = center[None] + 0.5 * (lo + hi)[:, None] * d
    g = grad(pF, s, lam)
    v = np.linalg.norm(g, axis=1)
    ds = np.linalg.norm(np.roll(pF, -1, 0) - np.roll(pF, 1, 0), axis=1) / 2
    return pF, g / v[:, None], v, ds


def kF_gamma(phi_deg, s, mu, lam=1.0, rlo=0.05, rhi=0.70, it=60):
    """radius of the Gamma pocket of spin s along the azimuths phi_deg (bisection)."""
    ph = np.radians(np.atleast_1d(np.asarray(phi_deg, float)))
    d = np.stack([np.cos(ph), np.sin(ph)], -1)
    e = lambda r: lowest(r[:, None] * d, s, lam)[0] - mu
    lo, hi = np.full(len(ph), rlo), np.full(len(ph), rhi)
    elo = e(lo)
    if not np.all(np.sign(elo) != np.sign(e(hi))):
        raise ValueError('no single Gamma-pocket crossing in the search interval')
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        em = e(mid)
        same = np.sign(em) == np.sign(elo)
        lo, hi, elo = np.where(same, mid, lo), np.where(same, hi, mid), np.where(same, em, elo)
    return 0.5 * (lo + hi)
