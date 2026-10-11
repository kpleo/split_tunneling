"""Twisted 1H-NbSe2 / 1H-NbSe2 tunnel junction in the tight-binding model (all pockets, orbital matrix elements).

Layer 1 at angle 0, layer 2 rotated by theta: E2(p) = E(R_-theta p), in-plane orbitals of layer 2 rotated by 2 theta.
Tunnelling conserves the lab-frame in-plane momentum p (forward channel of an incommensurate twist) and the spin, with
|T_s(p)|^2 = |u2^dag W u1|^2, W = diag(1, r_ip, r_ip) f(|p|), f(p) = exp(-(sqrt(k0^2 + p^2) - k0) d).
  * weak tunnelling: G_N with Lorentzian spectral functions of full width gamma (energy integral done analytically with a
    linearised layer-2 band) and I_c with the Abrikosov-Gor'kov kernel on contour-normal grids;
  * exact in T: Caroli conductance and Josephson current of the band-projected bilayer (each layer broadened by gamma);
  * first-shell umklapp channel p -> p + G (Gamma-pocket partners).
Energies in eV, momenta in 1/A.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tb

POCKETS = (('G', np.zeros(2), 0.70), ('K', tb.KPT, 0.62), ('Kp', -tb.KPT, 0.62))


def Dorb(th):
    c, s = np.cos(2 * th), np.sin(2 * th)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], complex)


def face(p, k0=1.0, d=3.0):
    return np.exp(-(np.sqrt(k0 ** 2 + p ** 2) - k0) * d)


def ag_kernel(D, G, xmax=0.2, n=601, nw=4001):
    """K(xi1, xi2) = int dw/2pi F(xi1, w) F(xi2, w), F = D eta/(eta^2 (w^2 + D^2) + xi^2), eta = 1 + G/(2 sqrt(w^2 + D^2))."""
    s = np.linspace(-np.arcsinh(xmax / D), np.arcsinh(xmax / D), n)
    x = D * np.sinh(s)
    u = np.linspace(-np.arcsinh(2e3), np.arcsinh(2e3), nw)
    w, dw = D * np.sinh(u), D * np.cosh(u) * (u[1] - u[0])
    eta = 1 + G / (2 * np.sqrt(w ** 2 + D ** 2))
    F = D * eta[None, :] / (eta[None, :] ** 2 * (w[None, :] ** 2 + D ** 2) + x[:, None] ** 2)
    itp = RegularGridInterpolator((s, s), (F * dw[None, :]) @ F.T / (2 * np.pi), bounds_error=False, fill_value=0.0)

    def kern(x1, x2):
        a = np.arcsinh(np.clip(x1, -xmax, xmax) / D)
        b = np.arcsinh(np.clip(x2, -xmax, xmax) / D)
        return itp(np.stack([a, b], -1))
    return kern


def caroli(xi1, xi2, T2, gamma):
    g = gamma / 2
    D = (-xi1 + 1j * g) * (-xi2 + 1j * g) - T2
    return gamma ** 2 * T2 / np.abs(D) ** 2


def omega_grid(Delta, Gamma, nw=90):
    u = np.linspace(0, np.arcsinh(4000.0), nw)
    wv, dw = Delta * np.sinh(u), Delta * np.cosh(u) * (u[1] - u[0])
    dw[0] *= 0.5
    dw[-1] *= 0.5
    eta = 1 + Gamma / (2 * np.sqrt(wv ** 2 + Delta ** 2))
    return wv * eta, Delta * eta, dw


def josephson_exact(x1, x2, T2, ww, PH, wgrid, chunk=5):
    """I(phi) of the band-projected bilayer: sum ww (1/pi) int dw 2 D~^2 |T|^2 sin(phi)/(w~^4 + w~^2 B + det)."""
    wt, Dt, dw = wgrid
    keep = ww * T2 > 0
    x1, x2, T2, ww = x1[keep], x2[keep], T2[keep], ww[keep]
    I = np.zeros(len(PH))
    for j0 in range(0, len(wt), chunk):
        wt_, Dt_, dw_ = wt[None, j0:j0 + chunk], Dt[None, j0:j0 + chunk], dw[None, j0:j0 + chunk]
        Bq = x1[:, None] ** 2 + x2[:, None] ** 2 + 2 * T2[:, None] + 2 * Dt_ ** 2
        base = (x1[:, None] * x2[:, None] - T2[:, None]) ** 2 + Dt_ ** 2 * (x1[:, None] ** 2 + x2[:, None] ** 2) + Dt_ ** 4
        for i, ph in enumerate(PH):
            det = base + Dt_ ** 2 * 2 * T2[:, None] * np.cos(ph)
            I[i] += np.sum(ww[:, None] * 2 * Dt_ ** 2 * T2[:, None] * np.sin(ph) / (wt_ ** 4 + wt_ ** 2 * Bq + det) * dw_) / np.pi
    return I


def phase_max(PH, tot):
    i = int(np.argmax(tot))
    if 0 < i < len(PH) - 1:
        c = np.polyfit(PH[i - 1:i + 2], tot[i - 1:i + 2], 2)
        return float(np.polyval(c, -c[1] / (2 * c[0]))), i
    return float(tot[i]), i


class TBJunction:
    def __init__(self, mu=0.0, lam=1.0, nsurf=2000, r_ip=0.4, k0=1.0, dgap=3.0, pockets=('G', 'K', 'Kp')):
        self.mu, self.lam, self.r_ip, self.k0, self.dgap = mu, lam, r_ip, k0, dgap
        self.pock = {}
        for name, c, rmax in POCKETS:
            if name in pockets:
                for s in (1, -1):
                    pF, n, v, ds = tb.fermi_contour(c, s, mu, nsurf, rmax, lam)
                    self.pock[(name, s)] = dict(pF=pF, n=n, v=v, ds=ds, u1F=tb.lowest(pF, s, lam)[1])
        self.grids = {}
        P = self.pock.get(('G', 1))
        if P is not None:                          # coupling scale: mean |T| on the Gamma contour (spin up) at theta = 0
            T0 = np.einsum('ni,ni,ni->n', np.conj(P['u1F']), self.W(P['pF']), P['u1F'])
            self.avgT = float(np.sum(P['ds'] * np.abs(T0)) / np.sum(P['ds']))

    def W(self, p):
        f = face(np.linalg.norm(p, axis=-1), self.k0, self.dgap)
        return np.stack([f, self.r_ip * f, self.r_ip * f], -1)

    def layer2(self, p, th, s):
        E, U = tb.lowest(p @ tb.rot(-th).T, s, self.lam)
        return E - self.mu, U @ Dorb(th).T

    def grid(self, xi):
        """contour-normal grids p = p_F + n xi/v for the energy grid xi = (scale, xmax, n): xi = scale sinh(u),
        |xi| <= xmax, with weights scale cosh(u) du."""
        key = tuple(xi)
        if key not in self.grids:
            scale, xmax, n = xi
            u = np.linspace(-np.arcsinh(xmax / scale), np.arcsinh(xmax / scale), n)
            xi, wxi = scale * np.sinh(u), scale * np.cosh(u) * (u[1] - u[0])
            G = {}
            for (name, s), P in self.pock.items():
                p = P['pF'][:, None] + P['n'][:, None] * (xi[None, :, None] / P['v'][:, None, None])
                E, U = tb.lowest(p, s, self.lam)
                G[(name, s)] = dict(p=p, xi1=E - self.mu, u1=U, w=(P['ds'] / P['v'])[:, None] * wxi[None])
            self.grids[key] = G
        return self.grids[key]

    # ---------------------------------------------------------------------------------------------- weak tunnelling
    def gn_weak(self, th, gamma, parts=False):
        """G_N: sum over pockets and spins of int ds/v1 |T|^2 (1/pi) g(1+a)/(b^2 + g^2 (1+a)^2), g = gamma/2,
        b = xi2(p_F1), a = |grad xi2 . n1|/v1."""
        g, out = gamma / 2, {}
        for (name, s), P in self.pock.items():
            b, u2 = self.layer2(P['pF'], th, s)
            g2 = tb.grad(P['pF'] @ tb.rot(-th).T, s, self.lam) @ tb.rot(-th)
            a = np.abs(np.einsum('ni,ni->n', g2, P['n'])) / P['v']
            T = np.einsum('ni,ni,ni->n', np.conj(u2), self.W(P['pF']), P['u1F'])
            out[(name, s)] = float(np.sum(P['ds'] / P['v'] * np.abs(T) ** 2 * g * (1 + a) / (b ** 2 + g ** 2 * (1 + a) ** 2) / np.pi))
        return out if parts else sum(out.values())

    def channel(self, th, kern, xi, mode='sc'):
        """sum over pockets and spins of int w |T|^2 kern(xi1, xi2) on the contour-normal grid xi; mode 'sc': forward
        channel, 'umk': first-shell umklapp p -> p + G1 with Gamma-pocket partners (per unit rho_U^2)."""
        Gs = 4 * np.pi / (np.sqrt(3.0) * tb.A)
        G1 = np.array([[Gs * np.cos(np.pi / 6 + m * np.pi / 3), Gs * np.sin(np.pi / 6 + m * np.pi / 3)] for m in range(6)])
        out = {}
        for (name, s), P in self.grid(xi).items():
            if mode == 'umk' and name != 'G':
                continue
            tot = 0.0
            for g in (G1 if mode == 'umk' else [np.zeros(2)]):
                q = (P['p'] + g) @ tb.rot(-th).T
                E, U = tb.lowest(q, s, self.lam)
                u2 = U @ Dorb(th).T
                T = np.einsum('...i,...i,...i->...', np.conj(u2), self.W(P['p']), P['u1'])
                val = P['w'] * np.abs(T) ** 2 * kern(P['xi1'], E - self.mu)
                if mode == 'umk':
                    centers = np.vstack([G1, [[0.0, 0.0]]])
                    val = val[np.min(np.linalg.norm(q[..., None, :] - centers[None, None], axis=-1), axis=-1) < 0.75]
                tot += float(val.sum())
            out[(name, s)] = tot
        return out

    # ---------------------------------------------------------------------------------------------- exact in T
    def gn_exact(self, th, t, gamma, xi):
        """Caroli conductance for coupling t (mean |T| on the Gamma contour at theta = 0, eV)."""
        sc, tot = (t / self.avgT) ** 2, 0.0
        for (name, s), P in self.grid(xi).items():
            xi2, u2 = self.layer2(P['p'], th, s)
            T2 = np.abs(np.einsum('...i,...i,...i->...', np.conj(u2), self.W(P['p']), P['u1'])) ** 2
            tot += float(np.sum(P['w'] * caroli(P['xi1'], xi2, T2 * sc, gamma)))
        return tot

    def ic_exact(self, th, t, Gamma, xi, Delta=0.45e-3, phis=(0.5, 0.62, 0.74, 0.86), nw=90):
        sc, PH = (t / self.avgT) ** 2, np.pi * np.array(phis)
        wg, tot = omega_grid(Delta, Gamma, nw), np.zeros(len(phis))
        for (name, s), P in self.grid(xi).items():
            xi2, u2 = self.layer2(P['p'], th, s)
            T2 = np.abs(np.einsum('...i,...i,...i->...', np.conj(u2), self.W(P['p']), P['u1'])) ** 2
            tot += josephson_exact(P['xi1'].ravel(), xi2.ravel(), (T2 * sc).ravel(), P['w'].ravel(), PH, wg)
        return phase_max(PH, tot)[0]
