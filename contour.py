"""Contour model of Gamma-pocket tunnelling between two twisted 1H layers.

Each layer has spin-resolved Fermi contours k_s(phi) = A(phi) + s B(phi) and radial Fermi velocities
v_s(phi) = vA(phi) + s vB(phi) (cosine series in the azimuth phi from Gamma-K; A, vA: harmonics 6n; B, vB: 6n+3).
Layer 2 is layer 1 rotated by theta. With Lorentzian spectral functions of full width gamma (energy) and a linear radial
dispersion the radial integral is exact, and the weak-tunnelling conductance and critical current become integrals over
the azimuth (constant tunnelling amplitude):
    G_N ~ sum_s int dphi k1 ve/(v1 v2) (1/pi) gamma/((ve dk)^2 + gamma^2),
    I_c ~ sum_s int dphi k1 ve/(v1 v2) K1(ve dk),
with dk = k1_s(phi) - k2_s(phi), ve = 2 v1 v2/(v1 + v2) and the Abrikosov-Gor'kov kernel
K1(b) = int dw D^2 E/(s^2 (b^2 + 4 E^2)), s^2 = w^2 + D^2, E = s + gamma/2. I_cR_N is given in units of pi D/(2e).
Also: the contact geometry, the scaling function of the broadening law, the angular current density, and the device
perturbations (band offset, Rashba texture, heterostrain, CDW self-energy, contact size, twist spread, incoherent current,
opposite-branch leakage) separately and combined.
"""
import json
import numpy as np

NPHI = 7200
PHI = (np.arange(NPHI) + 0.5) * 360.0 / NPHI          # azimuth grid (deg)
DPH = np.radians(360.0 / NPHI)
DELTA = 0.45e-3                                       # gap (eV)
TRUNC = {3: ((0, 6, 12), (3, 9)), 4: ((0, 6, 12, 18), (3, 9, 15)), 5: ((0, 6, 12, 18, 24), (3, 9, 15, 21)),
         6: ((0, 6, 12, 18, 24, 30), (3, 9, 15, 21, 27))}
trapz = getattr(np, 'trapezoid', None) or np.trapz


# ------------------------------------------------------------------------------------------------ contours
class Contour:
    def __init__(self, a, b, va, vb, label='', a_lat=3.44):
        self.a, self.b, self.va, self.vb = ({int(k): float(v) for k, v in d.items()} for d in (a, b, va, vb))
        self.label, self.a_lat = label, a_lat

    @staticmethod
    def ser(c, phi):
        p = np.radians(phi)
        return sum(v * np.cos(m * p) for m, v in c.items())

    def k(self, phi, s):
        return self.ser(self.a, phi) + s * self.ser(self.b, phi)

    def v(self, phi, s):
        return self.ser(self.va, phi) + s * self.ser(self.vb, phi)

    def A(self, phi):
        return self.ser(self.a, phi)

    def B(self, phi):
        return self.ser(self.b, phi)

    def vbar(self, phi):
        return self.ser(self.va, phi)


def fit_series(phis, y, ms):
    p = np.radians(phis)
    c = np.linalg.lstsq(np.stack([np.cos(m * p) for m in ms], 1), y, rcond=None)[0]
    return dict(zip(ms, c))


def bic_table(ph, A, B):
    """BIC of the joint least-squares fit of A and B for each truncation n (A up to 6(n-1), B up to 6(n-2)+3)."""
    out = {}
    p = np.radians(ph)
    for n, (ma, mb) in TRUNC.items():
        a, b = fit_series(ph, A, ma), fit_series(ph, B, mb)
        res = np.concatenate([A - sum(c * np.cos(m * p) for m, c in a.items()), B - sum(c * np.cos(m * p) for m, c in b.items())])
        out[n] = len(res) * np.log(np.mean(res ** 2)) + (len(ma) + len(mb)) * np.log(len(res))
    return out


def load_rays(path, trunc=None):
    """Contour from the Fermi crossings written by dft_elk.py (truncation by BIC unless `trunc` is given)."""
    d = json.load(open(path)) if isinstance(path, str) else path
    ph = np.array(d['phis'])
    ko, ki, vo, vi = (np.array(d[k]) for k in ('k_out', 'k_in', 'v_out', 'v_in'))
    if trunc is None:
        bic = bic_table(ph, (ko + ki) / 2, (ko - ki) / 2)
        trunc = min(bic, key=bic.get)
    ma, mb = TRUNC[trunc]
    C = Contour(fit_series(ph, (ko + ki) / 2, ma), fit_series(ph, (ko - ki) / 2, mb),
                fit_series(ph, (vo + vi) / 2, ma), fit_series(ph, (vo - vi) / 2, mb), d.get('material', ''), d.get('a_lat', 3.44))
    C.trunc, C.split_max_meV = trunc, d.get('split_max_meV')
    return C


def tb_contour(mu=0.0, nphi=720):
    """Gamma-pocket contour of the tight-binding model at Fermi level mu."""
    import tb
    phi = np.arange(nphi) * 360.0 / nphi
    d = np.stack([np.cos(np.radians(phi)), np.sin(np.radians(phi))], -1)
    ks, vs = {}, {}
    for s in (1, -1):
        k = tb.kF_gamma(phi, s, mu)
        e = lambda r: tb.lowest(r[:, None] * d, s)[0]
        ks[s], vs[s] = k, np.abs(e(k + 1e-4) - e(k - 1e-4)) / 2e-4
    ft = lambda y: np.fft.rfft(y) / nphi * 2
    cA, cB = ft((ks[1] + ks[-1]) / 2), ft((ks[1] - ks[-1]) / 2)
    cVA, cVB = ft((vs[1] + vs[-1]) / 2), ft((vs[1] - vs[-1]) / 2)
    cA[0] /= 2
    cVA[0] /= 2
    ma, mb = range(0, 61, 6), range(3, 64, 6)
    return Contour({m: cA[m].real for m in ma}, {m: cB[m].real for m in mb}, {m: cVA[m].real for m in ma},
                   {m: cVB[m].real for m in mb}, f'TB mu={mu:+.3f}', 3.44)


def tb_mu_for_radius(k_mean):
    """tight-binding Fermi level whose Gamma pocket has the mean radius k_mean."""
    import tb
    from scipy.optimize import brentq
    phi = np.arange(360) * 1.0
    f = lambda mu: 0.5 * (tb.kF_gamma(phi, 1, mu) + tb.kF_gamma(phi, -1, mu)).mean() - k_mean
    return brentq(f, -0.12, 0.08, xtol=1e-4)


# ------------------------------------------------------------------------------------------------ transport
def K1_table(D, G, bmax=0.5, nb=5001):
    b = np.linspace(0, bmax, nb)
    u = np.linspace(-np.arcsinh(4000.0), np.arcsinh(4000.0), 4001)
    w, dw = D * np.sinh(u), D * np.cosh(u) * (u[1] - u[0])
    s = np.sqrt(w ** 2 + D ** 2)
    E = s + G / 2
    K = np.array([np.sum(dw * D ** 2 * E / (s ** 2 * (bb ** 2 + 4 * E ** 2))) for bb in b])
    return lambda x: np.interp(np.abs(x), b, K, right=0.0)


def gn_ic(C, thetas, gamma, D=DELTA, dmu=0.0, want_ic=True, K1=None, opposite=False):
    """weak-tunnelling G_N(theta), I_c(theta) (arbitrary units) and I_cR_N in units of pi D/2e.
    dmu: relative band offset of layer 2 (its hole contour grows by dmu/v); opposite: couple spin s to spin -s."""
    K1 = K1 or (K1_table(D, gamma) if want_ic else None)
    G, I = [], []
    for th in np.atleast_1d(thetas):
        g = i = 0.0
        for s in (1, -1):
            s2 = -s if opposite else s
            k1, v1 = C.k(PHI, s), C.v(PHI, s)
            k2, v2 = C.k(PHI - th, s2), C.v(PHI - th, s2)
            k2 = k2 + dmu / v2
            ve = 2 * v1 * v2 / (v1 + v2)
            dk = k1 - k2
            w = k1 * ve / (v1 * v2) * DPH
            g += np.sum(w * gamma / ((ve * dk) ** 2 + gamma ** 2)) / np.pi
            if want_ic:
                i += np.sum(w * K1(ve * dk))
        G.append(g)
        I.append(i)
    G, I = np.array(G), np.array(I)
    return G, I, (I / G / np.pi / (np.pi * D / 2) if want_ic else None)


def peak(th, y, lo, hi):
    """position and height of the maximum of y(th) in [lo, hi] (parabolic refinement)."""
    th, y = np.asarray(th), np.asarray(y)
    m = (th >= lo) & (th <= hi)
    t, v = th[m], y[m]
    i = int(np.argmax(v))
    if 0 < i < len(v) - 1:
        c = np.polyfit(t[i - 1:i + 2] - t[i], v[i - 1:i + 2], 2)
        x = -c[1] / (2 * c[0])
        return float(t[i] + x), float(np.polyval(c, x))
    return float(t[i]), float(v[i])


def current_density(C, th, gamma):
    """angular density j_s(phi) of the weak-tunnelling current for each spin, normalised so that
    sum_s int dphi j_s = G_N(theta)/G_N(60 deg); also the same-spin mismatch dk_s(phi)."""
    out = {}
    for s in (1, -1):
        k1, v1, k2, v2 = C.k(PHI, s), C.v(PHI, s), C.k(PHI - th, s), C.v(PHI - th, s)
        ve = 2 * v1 * v2 / (v1 + v2)
        dk = k1 - k2
        out[s] = dict(j=k1 * ve / (v1 * v2) * gamma / ((ve * dk) ** 2 + gamma ** 2) / np.pi, dk=dk)
    tot = sum(np.sum(out[s]['j']) * DPH for s in (1, -1))
    G = gn_ic(C, [th, 60.0], gamma, want_ic=False)[0]
    for s in (1, -1):
        out[s]['j'] *= G[0] / G[1] / tot
    return out


# ------------------------------------------------------------------------------------------------ contact geometry
def contact(C):
    """exact contact angle eps_c (deg; f'(phi*) = 0 with f(phi) = k_+(phi) - k_-(phi - eps), phi* = 30 + eps/2), the odd
    expansion f(phi* + x) = lin x + c3 x^3 + ... at eps_c (x in rad), kappa = d lin/d eps (per rad), B'(30), A''(30),
    the leading-order angle 2|B'(30)|/A''(30) and the local velocity at the contact."""
    def f(x, eps):
        ps = 30 + eps / 2
        return C.k(ps + x, 1) - C.k(ps + x - eps, -1)

    def fp(eps, h=0.01):
        return (f(h, eps) - f(-h, eps)) / np.radians(2 * h)
    es = np.arange(0.2, 25.0, 0.1)
    vals = np.array([fp(e) for e in es])
    j = np.nonzero(np.diff(np.sign(vals)))[0]
    if len(j) == 0:
        return None
    a, b = es[j[0]], es[j[0] + 1]
    for _ in range(50):
        m = 0.5 * (a + b)
        if np.sign(fp(m)) == np.sign(fp(a)):
            a = m
        else:
            b = m
    ec = 0.5 * (a + b)
    out = dict(eps_c=float(ec))
    for xmax in (4.0, 8.0, 12.0):
        x = np.linspace(-xmax, xmax, 401)
        xr = np.radians(x)
        c = np.linalg.lstsq(np.stack([xr, xr ** 3, xr ** 5, xr ** 7], 1), f(x, ec), rcond=None)[0]
        out[f'fit_{xmax:g}deg'] = dict(lin=float(c[0]), c3=float(c[1]), c5=float(c[2]), c7=float(c[3]))
    out['kappa'] = float((fp(ec + 0.05) - fp(ec - 0.05)) / np.radians(0.1))
    ps = 30 + ec / 2
    out['phi_star'], out['v_contact'] = float(ps), float(C.v(np.array([ps]), 1)[0])
    out['B_prime_30'] = float((C.B(np.array([30.01])) - C.B(np.array([29.99])))[0] / np.radians(0.02))
    out['A_pp_30'] = float((C.A(np.array([30.01])) - 2 * C.A(np.array([30.0])) + C.A(np.array([29.99])))[0] / np.radians(0.01) ** 2)
    out['eps_lead'] = float(np.degrees(2 * abs(out['B_prime_30']) / abs(out['A_pp_30'])))
    k30, v30 = C.A(np.array([30.0]))[0], C.vbar(np.array([30.0]))[0]
    out['v_ising'] = float(v30 * abs(out['B_prime_30']) / k30)            # hbar v_Ising = v |B'(30)|/k_F at the mirror line
    return out


def scaling_function(u, shape='L'):
    """F(u) = int dy g(u y + y^3) for a Lorentzian (L) or Gaussian (G, same half width) line shape."""
    y = np.linspace(-12, 12, 240001)
    g = (lambda z: 1 / (z ** 2 + 1)) if shape == 'L' else (lambda z: np.exp(-np.log(2) * z ** 2))
    return np.array([trapz(g(a * y + y ** 3), y) for a in u])


# ------------------------------------------------------------------------------------------------ device perturbations
def rashba_gn(C, thetas, gamma, a1, a2):
    """G_N with interface Rashba coupling alpha_i k_F (eV) in layer i (spin texture; transmission (1 + n1.n2)/2)."""
    out = []
    ph = PHI
    yx, yy = -np.sin(np.radians(ph)), np.cos(np.radians(ph))

    def layer(th, al):
        A, B, v = C.A(ph - th), C.B(ph - th), C.vbar(ph - th)
        lam = np.sqrt((v * B) ** 2 + al ** 2)
        n = np.stack([al * yx, al * yy, v * B], -1) / lam[:, None]
        return [(A + lam / v, v, n), (A - lam / v, v, -n)]
    for th in np.atleast_1d(thetas):
        g = 0.0
        for k1, v1, n1 in layer(0.0, a1):
            for k2, v2, n2 in layer(th, a2):
                w = 0.5 * (1 + np.sum(n1 * n2, -1))
                ve = 2 * v1 * v2 / (v1 + v2)
                g += np.sum(k1 * ve / (v1 * v2) * DPH * w * gamma / ((ve * (k1 - k2)) ** 2 + gamma ** 2)) / np.pi
        out.append(g)
    return np.array(out)


def strained(C, u, th, s, nfine=14400):
    """radius and velocity at the azimuths PHI of layer 2 rotated by th and strained, p -> (1 - u) p."""
    p0 = (np.arange(nfine) + 0.5) * 360.0 / nfine
    k, v = C.k(p0, s), C.v(p0, s)
    ang = np.radians(p0 + th)
    P = np.stack([k * np.cos(ang), k * np.sin(ang)], -1)
    P = P - P @ np.asarray(u).T
    lab, rad = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360, np.hypot(P[:, 0], P[:, 1])
    o = np.argsort(lab)
    lab, rad, v = lab[o], rad[o], v[o]
    lab, rad, v = np.concatenate([lab - 360, lab, lab + 360]), np.tile(rad, 3), np.tile(v, 3)
    return np.interp(PHI, lab, rad), np.interp(PHI, lab, v)


def strain_gn_ic(C, thetas, gamma, u, K1):
    G, I = [], []
    for th in thetas:
        g = i = 0.0
        for s in (1, -1):
            k1, v1 = C.k(PHI, s), C.v(PHI, s)
            k2, v2 = strained(C, u, th, s)
            ve = 2 * v1 * v2 / (v1 + v2)
            dk = k1 - k2
            w = k1 * ve / (v1 * v2) * DPH
            g += np.sum(w * gamma / ((ve * dk) ** 2 + gamma ** 2)) / np.pi
            i += np.sum(w * K1(ve * dk))
        G.append(g)
        I.append(i)
    return np.array(G), np.array(I)


def cdw_sigma(C, s, V, gamma, mu_tb):
    """second-order 3x3 CDW self-energy on the layer-1 contour of spin s: sum_j V^2/(-xi(p + Q_j) + i gamma/2), Q_j along
    Gamma-M, |Q| = |b|/3; xi from the contour for |q| < 0.8/A, from the tight-binding band beyond."""
    import tb
    Q = 4 * np.pi / (3 * np.sqrt(3) * C.a_lat)
    k = C.k(PHI, s)
    P = np.stack([k * np.cos(np.radians(PHI)), k * np.sin(np.radians(PHI))], -1)
    sig = np.zeros(NPHI, complex)
    for j in range(6):
        ang = np.radians(30 + 60 * j)
        q = P + Q * np.array([np.cos(ang), np.sin(ang)])
        r, phq = np.hypot(q[:, 0], q[:, 1]), np.degrees(np.arctan2(q[:, 1], q[:, 0]))
        xi = -C.v(phq, s) * (r - C.k(phq, s))
        far = r >= 0.8
        if far.any():
            xi[far] = tb.lowest(q[far], s)[0] - mu_tb
        sig += V ** 2 / (-xi + 1j * gamma / 2)
    return sig


def cdw_gn(C, thetas, gamma, V, mu_tb):
    """G_N with the CDW self-energy in both layers."""
    step = 360.0 / NPHI
    sig = {s: cdw_sigma(C, s, V, gamma, mu_tb) for s in (1, -1)}
    out = []
    for th in thetas:
        n = int(round(th / step))
        g = 0.0
        for s in (1, -1):
            k1, v1, k2, v2 = C.k(PHI, s), C.v(PHI, s), C.k(PHI - th, s), C.v(PHI - th, s)
            s1, s2 = sig[s], np.roll(sig[s], n)
            g1, g2 = gamma / 2 - s1.imag, gamma / 2 - s2.imag
            k1e, k2e = k1 + s1.real / v1, k2 + s2.real / v2
            wid = g1 / v1 + g2 / v2
            g += np.sum(k1 * DPH / (v1 * v2) * (wid / np.pi) / ((k1e - k2e) ** 2 + wid ** 2))
        out.append(g)
    return np.array(out), sig


def size_kernel(L_um, nq=4001):
    """radial projection of the momentum-resolution kernel |FT(disk of diameter L)|^2 and its FWHM (1/A)."""
    from scipy.special import j1
    R = 0.5 * L_um * 1e4
    q, qt = np.linspace(-40 / R, 40 / R, nq), np.linspace(-40 / R, 40 / R, 2001)
    X = np.hypot(q[:, None], qt[None, :]) * R
    airy = np.where(X > 1e-9, (2 * j1(X) / np.where(X > 1e-9, X, 1)) ** 2, 1.0)
    w = airy.sum(1)
    w /= trapz(w, q)
    half = q[w >= w.max() / 2]
    return q, w, float(half.max() - half.min())


def size_gn(C, thetas, gamma, L_um):
    """G_N for a contact of diameter L_um (radial Lorentzian convolved with the contact kernel)."""
    vm = float(np.mean(C.vbar(PHI)))
    q, w, fwhm = size_kernel(L_um)
    b = np.linspace(-0.3, 0.3, 6001)
    prof = np.array([trapz(w * (gamma / np.pi) / ((bb - vm * q) ** 2 + gamma ** 2), q) for bb in b])
    G = []
    for th in thetas:
        acc = 0.0
        for s in (1, -1):
            k1, v1, k2, v2 = C.k(PHI, s), C.v(PHI, s), C.k(PHI - th, s), C.v(PHI - th, s)
            ve = 2 * v1 * v2 / (v1 + v2)
            acc += np.sum(k1 * DPH / (v1 * v2) * ve * np.interp(ve * (k1 - k2), b, prof))
        G.append(acc)
    return np.array(G), vm * fwhm


def spread(TH, y, sig):
    """Gaussian spread of twist angles (deg) applied to a curve on TH (up to 60 deg), mirrored about 60 deg."""
    tt = np.concatenate([TH, 120 - TH[::-1][1:]])
    yy = np.concatenate([y, y[::-1][1:]])
    if sig <= 0:
        return tt, yy
    x = np.arange(-4 * sig, 4 * sig + 1e-9, TH[1] - TH[0])
    w = np.exp(-x ** 2 / (2 * sig ** 2))
    return tt, np.convolve(yy, w / w.sum(), mode='same')


# ------------------------------------------------------------------------------------------------ all imperfections at once
def kernel_tables(gammas_p, D=DELTA, L_um=None, vm=2.1, bmax=0.3, db=1e-4):
    """G_N and I_c kernels as functions of (energy mismatch, pair width), optionally convolved with the contact kernel."""
    b = np.arange(-bmax, bmax + db / 2, db)
    u = np.linspace(-np.arcsinh(4000.0), np.arcsinh(4000.0), 4001)
    w, dw = D * np.sinh(u), D * np.cosh(u) * (u[1] - u[0])
    sq = np.sqrt(w ** 2 + D ** 2)
    if L_um:
        q, wq, _ = size_kernel(L_um)
        e = np.arange(-0.03, 0.03 + db / 2, db)
        ker = np.interp(e / vm, q, wq, left=0, right=0) / vm
        ker /= ker.sum()
    TL, TK = [], []
    for gp in gammas_p:
        lor = (gp / np.pi) / (b ** 2 + gp ** 2)
        E = sq + gp / 2
        Kb = np.array([np.sum(dw * D ** 2 * E / (sq ** 2 * (x ** 2 + 4 * E ** 2))) for x in np.abs(b)[: len(b) // 2 + 1]])
        Kb = np.concatenate([Kb, Kb[-2::-1]])
        if L_um:
            lor, Kb = np.convolve(lor, ker, mode='same'), np.convolve(Kb, ker, mode='same')
        TL.append(lor)
        TK.append(Kb)
    return b, np.array(gammas_p), np.array(TL), np.array(TK)


def interp2(b, gp, T, x, y):
    ix = np.clip((x - b[0]) / (b[1] - b[0]), 0, len(b) - 1.000001)
    i0 = ix.astype(int)
    fx = ix - i0
    iy = np.clip(np.interp(y, gp, np.arange(len(gp))), 0, len(gp) - 1.000001)
    j0 = iy.astype(int)
    fy = iy - j0
    return ((1 - fy) * ((1 - fx) * T[j0, i0] + fx * T[j0, i0 + 1]) + fy * ((1 - fx) * T[j0 + 1, i0] + fx * T[j0 + 1, i0 + 1]))


def joint_curves(C, TH, gamma, tables, dmu=0.0, alpha=0.0, V=0.0, mu_tb=None, eta=0.0):
    """coherent G_N(theta), I_c(theta) with band offset, equal-sign Rashba alpha k_F, CDW self-energy, contact kernel (in
    the tables) and an opposite-branch leakage probability eta (spinor weight w -> w + eta (1 - w))."""
    ph = PHI
    yx, yy = -np.sin(np.radians(ph)), np.cos(np.radians(ph))
    sig = {s: (cdw_sigma(C, s, V, gamma, mu_tb) if V > 0 else np.zeros(NPHI, complex)) for s in (1, -1)}
    step = 360.0 / NPHI
    b, gp, TL, TK = tables

    def layer(th):
        A, B, va, vb = C.A(ph - th), C.B(ph - th), C.vbar(ph - th), C.ser(C.vb, ph - th)
        lam = np.sqrt((va * B) ** 2 + alpha ** 2 + 1e-30)
        sg = np.where(B >= 0, 1.0, -1.0)
        n_out = np.stack([alpha * yx, alpha * yy, va * B], -1) / lam[:, None]
        n = int(round(th / step))
        sp, sm = np.roll(sig[1], n), np.roll(sig[-1], n)
        bands = []
        for sgn, nn in ((1, n_out), (-1, -n_out)):
            S = 0.5 * (1 + nn[:, 2]) * sp + 0.5 * (1 - nn[:, 2]) * sm
            bands.append((A + sgn * lam / va, va + sgn * sg * vb, nn, S))
        return bands
    L1 = layer(0.0)
    G, I = [], []
    for th in TH:
        g = i = 0.0
        for k1, v1, n1, S1 in L1:
            for k2, v2, n2, S2 in layer(th):
                k2 = k2 + dmu / v2
                w = 0.5 * (1 + np.sum(n1 * n2, -1))
                w = w + eta * (1 - w)
                k1e, k2e = k1 + S1.real / v1, k2 + S2.real / v2
                g1, g2 = gamma / 2 - S1.imag, gamma / 2 - S2.imag
                ve = 2 * v1 * v2 / (v1 + v2)
                gpair, beta = ve * (g1 / v1 + g2 / v2), ve * (k1e - k2e)
                pre = k1 * ve / (v1 * v2) * DPH * w
                g += np.sum(pre * interp2(b, gp, TL, beta, gpair))
                i += np.sum(pre * interp2(b, gp, TK, beta, gpair))
        G.append(g)
        I.append(i)
    return np.array(G), np.array(I)
