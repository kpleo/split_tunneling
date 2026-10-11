"""Gamma-pocket Fermi contours of 1H monolayers from all-electron DFT (Elk, PBE + spin-orbit coupling).

For one material: (1) self-consistent ground state; (2) band structures along radial rays from Gamma, one Elk task-20 run
per azimuth phi = 0..30 deg (from Gamma-K; the symmetry of the layer fixes the contours everywhere else); (3) the two
spin-split Gamma-pocket crossings of E_F and the radial Fermi velocities on every ray, also at small shifts of the Fermi
level. The result, contours/NAME.json, is read by contour.load_rays.

usage:  python dft_elk.py NAME --species DIR [--elk elk] [--threads 2] [--rgkmax 8] [--nk 30] [--swidth 0.0005] [--tag T]
NAME: NbSe2, TaS2, NbSe2_h1.656 (chalcogen height where the PBE force vanishes), NbS2, TaSe2.
DIR is the species folder of the Elk installation. Work files go to runs/NAME[_T]/, the contours to contours/NAME[_T].json
(use a tag for convergence checks with other parameters).
"""
import argparse
import json
import os
import subprocess
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

BOHR = 0.529177210903          # Angstrom
HA = 27.211386245988           # eV
VACUUM = 20.0                  # cell height (Angstrom)
# metal, chalcogen, lattice constant a (A), chalcogen height above/below the metal plane (A), ray range r1, r2 (1/A), points
STRUCT = {'NbSe2': ('Nb', 'Se', 3.44, 1.68, 0.36, 0.64, 41), 'NbSe2_h1.656': ('Nb', 'Se', 3.44, 1.656, 0.36, 0.64, 41),
          'TaS2': ('Ta', 'S', 3.315, 1.58, 0.30, 0.72, 57), 'NbS2': ('Nb', 'S', 3.31, 1.58, 0.30, 0.72, 57),
          'TaSe2': ('Ta', 'Se', 3.436, 1.68, 0.30, 0.72, 57)}
PHIS = [float(x) for x in range(0, 20, 2)] + [20 + 0.5 * i for i in range(21)]      # azimuths of the rays (deg)
SHIFTS = (-0.05, -0.025, -0.01, 0.01, 0.025, 0.05)                                  # Fermi-level shifts (eV)


def elk_input(M, X, a, h, species, nk, rgkmax, swidth, tasks=0, plot1d=None):
    A, C, f = a / BOHR, VACUUM / BOHR, h / VACUUM
    src = f"""tasks
  {tasks}

xctype
  20

spinorb
  .true.

avec
  {A:.8f}  0.0  0.0
  {-A / 2:.8f}  {A * np.sqrt(3) / 2:.8f}  0.0
  0.0  0.0  {C:.8f}

sppath
  '{species.rstrip('/')}/'

autormt
  .true.

atoms
  2
  '{M}.in'
  1
  0.0  0.0  0.5
  '{X}.in'
  2
  0.333333333333  0.666666666667  {0.5 + f:.10f}
  0.333333333333  0.666666666667  {0.5 - f:.10f}

ngridk
  {nk}  {nk}  1

rgkmax
  {rgkmax}

nempty
  10

stype
  3

swidth
  {swidth}

epspot
  1.d-7

epsengy
  1.d-5

maxscl
  120
"""
    if plot1d is not None:
        verts, npts = plot1d
        src += f'\nplot1d\n  {len(verts)}  {npts}\n' + ''.join(f'  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}\n' for v in verts)
    return src


def frac(k, a):
    """Cartesian in-plane k (1/A) -> fractional coordinates of the reciprocal lattice (a1 = a(1, 0), a2 = a(-1/2, sqrt3/2))."""
    b1 = 2 * np.pi / a * np.array([1.0, 1 / np.sqrt(3)])
    b2 = 2 * np.pi / a * np.array([0.0, 2 / np.sqrt(3)])
    x = np.linalg.solve(np.stack([b1, b2], 1), k)
    return [x[0], x[1], 0.0]


def parse_band(path):
    blocks, cur = [], []
    for line in open(path):
        if line.strip():
            cur.append([float(x) for x in line.split()[:2]])
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return np.array(blocks)[:, :, 1] * HA            # (bands, points), eV relative to E_F


def crossings(r, E):
    """(radius, dE/dr) at the zeros of E(r) (cubic interpolation on six points)."""
    out = []
    s = np.sign(E)
    for i in np.nonzero(s[:-1] != s[1:])[0]:
        lo, hi = max(i - 2, 0), min(i + 4, len(r))
        cs = CubicSpline(r[lo:hi], E[lo:hi])
        k = brentq(cs, r[i], r[i + 1])
        out.append((k, float(cs(k, 1))))
    return out


def gamma_crossings(r, E, e0=0.0):
    """outer and inner Gamma-pocket radii and |velocities| on every ray at energy e0, and the splitting at k_F."""
    ko, ki, vo, vi, split = [], [], [], [], []
    for j in range(E.shape[0]):
        cr = []
        for b in range(E.shape[1]):
            c = crossings(r, E[j, b] - e0)
            if len(c) == 1:
                cr.append((c[0][0], abs(c[0][1]), b))
        if len(cr) != 2:
            return None
        (k1, v1, b1), (k2, v2, b2) = sorted(cr)
        ki.append(k1); ko.append(k2); vi.append(v1); vo.append(v2)
        km = 0.5 * (k1 + k2)
        split.append(abs(np.interp(km, r, E[j, b1]) - np.interp(km, r, E[j, b2])))
    return dict(k_out=ko, k_in=ki, v_out=vo, v_in=vi, split_meV=list(1e3 * np.array(split)))


def run(name, species, elk, threads, rgkmax, nk, swidth, tag=''):
    M, X, a, h, r1, r2, npts = STRUCT[name]
    label = f'{name}_{tag}' if tag else name
    wd = os.path.join('runs', label)
    os.makedirs(wd, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    log = lambda tag: open(os.path.join(wd, f'{tag}.log'), 'w')
    if not os.path.exists(os.path.join(wd, 'STATE.OUT')):
        open(os.path.join(wd, 'elk.in'), 'w').write(elk_input(M, X, a, h, species, nk, rgkmax, swidth))
        print(f'{name}: ground state', flush=True)
        subprocess.run([elk], cwd=wd, env=env, check=True, stdout=log('scf'), stderr=subprocess.STDOUT)
    rays = os.path.join(wd, 'rays.npz')
    if not os.path.exists(rays):
        Es = []
        for ph in PHIS:
            u = np.array([np.cos(np.radians(ph)), np.sin(np.radians(ph))])
            open(os.path.join(wd, 'elk.in'), 'w').write(
                elk_input(M, X, a, h, species, nk, rgkmax, swidth, tasks=20, plot1d=([frac(r1 * u, a), frac(r2 * u, a)], npts)))
            subprocess.run([elk], cwd=wd, env=env, check=True, stdout=log(f'ray{ph:g}'), stderr=subprocess.STDOUT)
            Es.append(parse_band(os.path.join(wd, 'BAND.OUT')))
            print(f'{name}: ray phi = {ph:g} deg done', flush=True)
        np.savez(rays, phis=np.array(PHIS), r=np.linspace(r1, r2, npts), E=np.array(Es))
    z = np.load(rays)
    out = dict(material=label, a_lat=a, phis=z['phis'].tolist(), **gamma_crossings(z['r'], z['E']))
    out['split_max_meV'] = float(max(out['split_meV']))
    out['shifts'] = {f'{1e3 * e0:g}': gamma_crossings(z['r'], z['E'], e0) for e0 in SHIFTS}
    os.makedirs('contours', exist_ok=True)
    json.dump(out, open(os.path.join('contours', f'{label}.json'), 'w'), indent=1)
    print(f'{label}: wrote contours/{label}.json', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', choices=sorted(STRUCT))
    ap.add_argument('--species', required=True, help='species folder of the Elk installation')
    ap.add_argument('--elk', default='elk', help='Elk executable (default: elk)')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--rgkmax', type=float, default=8.0)
    ap.add_argument('--nk', type=int, default=30)
    ap.add_argument('--swidth', type=float, default=0.0005, help='smearing width (Ha)')
    ap.add_argument('--tag', default='', help='suffix of the run and contour names (e.g. rgk9)')
    a = ap.parse_args()
    run(a.name, a.species, a.elk, a.threads, a.rgkmax, a.nk, a.swidth, a.tag)
