"""Reproduce the calculations of the split tunneling resonance in twisted monolayer Ising superconductors.

usage: python reproduce.py COMMAND [options]      (python reproduce.py -h lists the commands)

Contour-model commands read the DFT contours written by dft_elk.py (contours/NbSe2.json, contours/TaS2.json, ...) and
always include the tight-binding contour (TB, mu = 0). Every command prints a summary and writes results/COMMAND.json.
"""
import argparse
import glob
import json
import os
import time
import numpy as np
import contour as cm

GAMMAS = (2e-3, 5e-3, 10e-3)
D = cm.DELTA


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def save(name, obj):
    os.makedirs('results', exist_ok=True)
    json.dump(jsonable(obj), open(os.path.join('results', f'{name}.json'), 'w'))


def dft_files(files):
    return files or sorted(glob.glob(os.path.join('contours', '*.json')))


def systems(files, tb=True):
    S = {'TB mu=0': cm.tb_contour(0.0)} if tb else {}
    for f in dft_files(files):
        S['DFT ' + os.path.splitext(os.path.basename(f))[0]] = cm.load_rays(f)
    return S


def mu_tb(path):
    """tight-binding Fermi level matching the mean Gamma-pocket radius of a DFT contour (used beyond the Gamma pocket)."""
    d = json.load(open(path))
    A = (np.array(d['k_out']) + np.array(d['k_in'])) / 2
    return cm.tb_mu_for_radius(cm.fit_series(np.array(d['phis']), A, cm.TRUNC[4][0])[0])


# ------------------------------------------------------------------------------------------------ contour model
def cmd_contact(a):
    out = {}
    for lab, C in systems(a.files).items():
        g = cm.contact(C)
        out[lab] = dict(contact=g, a=C.a, b=C.b)
        print(f"{lab}: delta_c = {g['eps_c']:.3f} deg (leading order {g['eps_lead']:.3f}), hbar v_Ising = {g['v_ising']:.3f} eV A, "
              f"<k_F> = {C.a[0]:.4f} 1/A, a6 = {1e3 * C.a[6]:.2f}, a12 = {1e3 * C.a[12]:.3f}, b3 = {1e3 * C.b[3]:.2f}, "
              f"b9 = {1e3 * C.b[9]:.3f} (1e-3/A); kappa = {g['kappa']:.4f}, c3 = {g['fit_4deg']['c3']:+.4f}, v = {g['v_contact']:.3f} eV A")
    for f in dft_files(a.files):
        lab = 'DFT ' + os.path.splitext(os.path.basename(f))[0]
        d = json.load(open(f))
        ph = np.array(d['phis'])
        ko, ki = np.array(d['k_out']), np.array(d['k_in'])
        bic = cm.bic_table(ph, (ko + ki) / 2, (ko - ki) / 2)
        best = min(bic, key=bic.get)
        adm = [n for n in bic if bic[n] - bic[best] < 10]
        ecs = {n: cm.contact(cm.load_rays(d, n))['eps_c'] for n in adm}
        shifts = {k: cm.contact(cm.load_rays(dict(d, **v), best))['eps_c'] for k, v in d.get('shifts', {}).items() if v}
        out[lab].update(split_max_meV=d['split_max_meV'], truncation=best, admissible=adm, eps_c_fits=ecs, eps_c_shift=shifts)
        print(f"{lab}: Delta_Gamma = {d['split_max_meV']:.0f} meV; truncation {best} (admissible {adm}): delta_c "
              + ', '.join(f'{v:.2f}' for v in ecs.values()) + '; E_F shift (meV -> delta_c): '
              + ', '.join(f'{k}: {v:.2f}' for k, v in shifts.items()))
    save('contact', out)


def cmd_peaks(a):
    TH = np.round(np.concatenate([[0.0, 10.0, 15.0, 20.0], np.arange(30.0, 60.0001, 0.05)]), 3)
    i0, i30, i60 = 0, int(np.argmin(np.abs(TH - 30))), len(TH) - 1
    out = {}
    for lab, C in systems(a.files).items():
        out[lab] = []
        for g in GAMMAS + (20e-3,):
            G, I, R = cm.gn_ic(C, TH, g)
            tG, gp = cm.peak(TH, G, 30.5, 59.95)
            tI, ip = cm.peak(TH, I, 30.5, 59.95)
            Rp = float(np.interp(tI, TH, R))
            out[lab].append(dict(gamma=g, G_peak=tG, G_contrast=gp / G[i60], Ic_peak=tI, Ic_contrast=ip / I[i60], G_0_60=G[i0] / G[i60],
                                 Ic_0_60=I[i0] / I[i60], IcRN_AB_0=R[i0], IcRN_AB_30=R[i30], IcRN_AB_peak=Rp, IcRN_AB_60=R[i60]))
            print(f"{lab:16s} gamma {g * 1e3:4.0f} meV: G_N peak {tG:6.2f} deg (x{gp / G[i60]:.2f}), I_c peak {tI:6.2f} deg "
                  f"(x{ip / I[i60]:.2f}); G_N 0/60 {G[i0] / G[i60]:5.1f}; I_cR_N/AB 0: {R[i0]:.2f} 30: {R[i30]:.2f} "
                  f"peak: {Rp:.2f} 60: {R[i60]:.2f}")
    if a.fits:                                     # peak positions over the admissible harmonic truncations
        TH2 = np.round(np.arange(40.0, 60.0001, 0.05), 3)
        for f in dft_files(a.files):
            d = json.load(open(f))
            ph, ko, ki = np.array(d['phis']), np.array(d['k_out']), np.array(d['k_in'])
            bic = cm.bic_table(ph, (ko + ki) / 2, (ko - ki) / 2)
            best = min(bic, key=bic.get)
            for n in [n for n in bic if bic[n] - bic[best] < 10]:
                C = cm.load_rays(d, n)
                for g in GAMMAS:
                    G, I, _ = cm.gn_ic(C, TH2, g)
                    print(f"{os.path.basename(f)} truncation {n}, gamma {g * 1e3:g} meV: G_N peak {cm.peak(TH2, G, 40, 59.95)[0]:.3f}, "
                          f"I_c peak {cm.peak(TH2, I, 40, 59.95)[0]:.3f} deg")
    save('peaks', out)


def cmd_law(a):
    us = np.linspace(-4, 1, 501)
    FL, FG = cm.scaling_function(us, 'L'), cm.scaling_function(us, 'G')
    ustar = float(us[np.argmax(FL)])
    out = dict(F=dict(u=us, L=FL, G=FG, u_star_L=ustar, u_star_G=float(us[np.argmax(FG)])), systems={})
    print(f"F(u): Lorentzian u* = {ustar:.3f}, F(u*) = {FL.max():.4f}; Gaussian u* = {us[np.argmax(FG)]:.3f}")
    for lab, C in systems(a.files).items():
        geo = cm.contact(C)
        ec, rows = geo['eps_c'], []
        for g in (0.125e-3, 0.25e-3, 0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3):
            coarse = np.arange(60 - ec - 6, 60.0001, 0.05)
            t0, _ = cm.peak(coarse, cm.gn_ic(C, coarse, g, want_ic=False)[0], 40, 59.95)
            fine = np.arange(t0 - 0.1, t0 + 0.1001, 0.002)
            tp, gp = cm.peak(fine, cm.gn_ic(C, fine, g, want_ic=False)[0], fine[0], fine[-1])
            g60 = cm.gn_ic(C, [60.0], g, want_ic=False)[0][0]
            law = np.degrees(abs(ustar) * abs(geo['fit_4deg']['c3']) ** (1 / 3) / abs(geo['kappa']) * (g / geo['v_contact']) ** (2 / 3))
            rows.append(dict(gamma=g, theta_peak=tp, offset=60 - tp - ec, offset_cubic_law=law, height=gp, contrast=gp / g60))
        gam, off, hgt = (np.array([r[k] for r in rows]) for k in ('gamma', 'offset', 'height'))
        for r, x, y in zip(rows, np.gradient(np.log(np.abs(off)), np.log(gam)), np.gradient(np.log(hgt), np.log(gam))):
            r['exp_offset'], r['exp_height'] = float(x), float(y)
        out['systems'][lab] = dict(contact=geo, rows=rows)
        print(f"{lab}: delta_c {ec:.3f} deg, kappa {geo['kappa']:.4f}, c3 {geo['fit_4deg']['c3']:+.4f}, v {geo['v_contact']:.3f} eV A")
        for r in rows:
            print(f"   gamma {r['gamma'] * 1e3:6.3f} meV: peak {r['theta_peak']:.3f} deg, offset {r['offset']:.4f} deg (cubic law "
                  f"{r['offset_cubic_law']:.4f}), contrast {r['contrast']:.2f}, exponents offset {r['exp_offset']:.3f} height {r['exp_height']:.3f}")
    save('law', out)


def cmd_window(a):
    f = a.files[0] if a.files else os.path.join('contours', 'NbSe2.json')
    C, mtb = cm.load_rays(f), mu_tb(f)
    TH = np.round(np.arange(40.0, 60.0001, 0.05), 3)
    i60 = len(TH) - 1
    contr = lambda y: (lambda t, p: (t, p / y[i60]))(*cm.peak(TH, y, 40, 59.95))
    out = {}
    base = {g: cm.gn_ic(C, TH, g)[:2] for g in GAMMAS}
    for g in GAMMAS:                                                          # relative band offset
        K1 = cm.K1_table(D, g)
        for dmu in (0.0, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3):
            G, I, _ = cm.gn_ic(C, TH, g, dmu=dmu, K1=K1)
            (tG, cG), (tI, cI) = contr(G), contr(I)
            out[f'dmu|{g * 1e3:g}|{dmu * 1e3:g}'] = dict(G_peak=tG, G_contrast=cG, Ic_peak=tI, Ic_contrast=cI)
            print(f'band offset {dmu * 1e3:4.1f} meV, gamma {g * 1e3:g} meV: G_N peak {tG:6.2f} x{cG:.2f}; I_c peak {tI:6.2f} x{cI:.2f}')
    for g in GAMMAS:                                                          # interface Rashba texture
        c0 = contr(cm.rashba_gn(C, TH, g, 0.0, 0.0))[1]
        for al in (0.42e-3, 1e-3, 2e-3, 4e-3):
            for mode, (a1, a2) in (('equal', (al, al)), ('opposite', (al, -al)), ('one layer', (al, 0.0))):
                t, c = contr(cm.rashba_gn(C, TH, g, a1, a2))
                out[f'rashba|{g * 1e3:g}|{al * 1e3:g}|{mode}'] = dict(peak=t, contrast=c, ref=c0)
                print(f'Rashba {mode:9s} alpha k_F {al * 1e3:4.2f} meV, gamma {g * 1e3:g} meV: peak {t:6.2f}, x{c:.2f} ({100 * (c / c0 - 1):+.1f}%)')
    g = 5e-3
    K1 = cm.K1_table(D, g)
    for kind, e, psi in (('uniaxial', 1e-3, 0), ('uniaxial', 1e-3, 30), ('uniaxial', 1e-3, 90), ('uniaxial', 2e-3, 0),
                         ('uniaxial', 2e-3, 30), ('biaxial', 1e-3, 0), ('biaxial', 2e-3, 0)):          # heterostrain of layer 2
        n = np.array([np.cos(np.radians(psi)), np.sin(np.radians(psi))])
        t_ = np.array([-n[1], n[0]])
        u = e * (np.outer(n, n) - 0.3 * np.outer(t_, t_)) if kind == 'uniaxial' else e * np.eye(2)
        G, I = cm.strain_gn_ic(C, TH, g, u, K1)
        G2, I2 = cm.strain_gn_ic(C, 120 - TH, g, u, K1)
        (tG, cG), (tG2, cG2) = contr(G), contr(G2)
        out[f'strain|{kind}|{e * 1e2:g}|{psi}'] = dict(G_peak=tG, G_contrast=cG, G_peak_hi=120 - tG2, G_contrast_hi=cG2,
                                                       Ic_contrast=contr(I)[1], Ic_contrast_hi=contr(I2)[1])
        print(f'heterostrain {kind} {e * 1e2:.1f}% at {psi} deg (gamma 5 meV): G_N peaks {tG:.2f}/{120 - tG2:.2f} x{cG:.2f}/{cG2:.2f}')
    for g in GAMMAS:                                                          # twist spread and incoherent current
        G30, I30, _ = cm.gn_ic(C, [30.0], g)
        for key, y, y30 in (('G_N', base[g][0], G30[0]), ('I_c', base[g][1], I30[0])):
            sp = []
            for sig in (0.25, 0.5, 1.0, 2.0):
                tt, ys = cm.spread(TH, y, sig)
                m, i6 = (tt >= 44) & (tt <= 59.95), int(np.argmin(np.abs(tt - 60)))
                out[f'spread|{g * 1e3:g}|{key}|{sig:g}'] = sp_c = float(ys[m].max() / ys[i6])
                sp.append(f'{sig:g} deg x{sp_c:.2f}')
            p = cm.peak(TH, y, 40, 59.95)[1]
            bg = []
            for x in (0.25, 0.5, 1.0, 2.0):
                out[f'background|{g * 1e3:g}|{key}|{x:g}'] = c = float((p + x * y30) / (y[i60] + x * y30))
                bg.append(f'{x:g} x{c:.2f}')
            print(f'gamma {g * 1e3:g} meV {key}: twist spread ' + ', '.join(sp) + '; incoherent current x_inc ' + ', '.join(bg))
    for Lum in (0.1, 0.2, 0.5, 1.0, 2.0):                                     # contact size
        for g in GAMMAS:
            G, dE = cm.size_gn(C, TH, g, Lum)
            t, c = contr(G)
            out[f'size|{Lum:g}|{g * 1e3:g}'] = dict(peak=t, contrast=c, dE_fwhm=dE)
            print(f'contact {Lum:g} um (v dq = {1e3 * dE:.1f} meV), gamma {g * 1e3:g} meV: peak {t:.2f}, x{c:.2f}')
    for g in GAMMAS:                                                          # 3x3 CDW
        for V in (0.0, 2e-3, 4e-3):
            G, sig = cm.cdw_gn(C, TH, g, V, mtb)
            t, c = contr(G)
            out[f'cdw|{g * 1e3:g}|{V * 1e3:g}'] = dict(peak=t, contrast=c)
            print(f'CDW V = {V * 1e3:g} meV, gamma {g * 1e3:g} meV: peak {t:.2f}, x{c:.2f}')
    save('window', out)


def cmd_leak(a):
    TH = np.round(np.arange(40.0, 60.0001, 0.05), 3)
    out = {}
    for f in dft_files(a.files):
        C, lab = cm.load_rays(f), os.path.splitext(os.path.basename(f))[0]
        for g in GAMMAS:
            K1 = cm.K1_table(D, g)
            ss = np.array(cm.gn_ic(C, TH, g, K1=K1)[:2]).T
            op = np.array(cm.gn_ic(C, TH, g, K1=K1, opposite=True)[:2]).T
            for j, key in ((0, 'G_N'), (1, 'I_c')):
                row = []
                for eta in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04):
                    y = ss[:, j] + eta * op[:, j]
                    t, p = cm.peak(TH, y, 40, 59.95)
                    out[f'{lab}|{g * 1e3:g}|{key}|{eta:g}'] = dict(peak=t, contrast=p / y[-1])
                    row.append(f'{100 * eta:g}%: x{p / y[-1]:.2f}')
                print(f'{lab} gamma {g * 1e3:g} meV {key}: leakage ' + ', '.join(row))
    save('leak', out)


def cmd_joint(a):
    f = a.files[0] if a.files else os.path.join('contours', 'NbSe2.json')
    C, mtb = cm.load_rays(f), mu_tb(f)
    cfg = dict(dmu=2e-3, alpha=0.42e-3, V=2e-3, L_um=1.0, sigma_theta=0.5, x_inc=0.25, eta=0.01)
    TH = np.round(np.arange(40.0, 60.0001, 0.05), 3)
    vm = float(np.mean(C.vbar(cm.PHI)))
    out = dict(config=cfg)
    for g in GAMMAS:
        gps = np.linspace(g, g + 0.02, 41)
        tab0, tabL = cm.kernel_tables(gps, vm=vm), cm.kernel_tables(gps, L_um=cfg['L_um'], vm=vm)
        G0, I0 = cm.joint_curves(C, TH, g, tab0)
        res = {}
        for eta in (0.0, cfg['eta']):
            kw = dict(dmu=cfg['dmu'], alpha=cfg['alpha'], V=cfg['V'], mu_tb=mtb, eta=eta)
            G1, I1 = cm.joint_curves(C, TH, g, tabL, **kw)
            G30, I30 = cm.joint_curves(C, [30.0], g, tabL, **kw)
            for key, y0, y1, y30 in (('G_N', G0, G1, G30[0]), ('I_c', I0, I1, I30[0])):
                tt, ys = cm.spread(TH, y1, cfg['sigma_theta'])
                tot = ys + cfg['x_inc'] * y30
                m, i6 = (tt >= 44) & (tt <= 59.95), int(np.argmin(np.abs(tt - 60)))
                tp, pv = cm.peak(tt[m], tot[m], 44, 59.95)
                t0, p0 = cm.peak(TH, y0, 40, 59.95)
                res[f'{key}|eta={eta:g}'] = dict(ideal_peak=t0, ideal_contrast=p0 / y0[-1], peak=tp, contrast=pv / tot[i6])
        out[f'{g * 1e3:g}'] = res
        g0, ge, i0 = res['G_N|eta=0'], res['G_N|eta=%g' % cfg['eta']], res['I_c|eta=0']
        print(f"gamma {g * 1e3:g} meV: G_N ideal x{g0['ideal_contrast']:.2f}, combined x{g0['contrast']:.2f} at {g0['peak']:.2f} deg, "
              f"with eta = {100 * cfg['eta']:g}% x{ge['contrast']:.2f}; I_c ideal x{i0['ideal_contrast']:.2f}, combined x{i0['contrast']:.2f}")
    save('joint', out)


def cmd_maps(a):
    f = a.files[0] if a.files else os.path.join('contours', 'NbSe2.json')
    C = cm.load_rays(f)
    dc = cm.contact(C)['eps_c']
    out = dict(phi=cm.PHI[::4], delta_c=dc)
    for g in GAMMAS:
        ths = np.arange(45.0, 60.0001, 0.02)
        tp = cm.peak(ths, cm.gn_ic(C, ths, g, want_ic=False)[0], 45, 59.9)[0]
        for key, th in (('60', 60.0), ('contact', 60.0 - dc), ('peak', tp)):
            d = cm.current_density(C, th, g)
            ju, jd = d[1]['j'], d[-1]['j']
            share = float(jd.sum() / (ju.sum() + jd.sum()))
            out[f'{g * 1e3:g}|{key}'] = dict(theta=th, G_rel=float((ju.sum() + jd.sum()) * cm.DPH), j_up=ju[::4], j_dn=jd[::4],
                                             dk_up=d[1]['dk'][::4], dk_dn=d[-1]['dk'][::4], share_dn=share)
            print(f"gamma {g * 1e3:g} meV, {key:7s} theta = {th:7.3f} deg: G_N/G_N(60) = {(ju.sum() + jd.sum()) * cm.DPH:.3f}, "
                  f"spin-down share {share:.3f}")
    save('maps', out)


# ------------------------------------------------------------------------------------------------ tight-binding junction
ANG = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 20, 25, 28, 30, 32, 35, 40, 45, 48, 50, 51, 52, 52.5, 53, 53.5,
       54, 54.5, 55, 55.5, 56, 56.5, 57, 57.5, 58, 58.5, 59, 59.5, 60]


def _tb_task(args):
    import junction as jn
    lam, g = args
    Jn, Jc = jn.TBJunction(0.0, lam, nsurf=4000), jn.TBJunction(0.0, lam, nsurf=2000)
    kern = jn.ag_kernel(D, g)
    rows = []
    for a_ in ANG:
        th = np.radians(a_)
        ic, gn = Jc.channel(th, kern, (D, 300 * D, 161)), Jn.gn_weak(th, g, parts=True)
        rows.append((sum(ic.values()), ic[('G', 1)] + ic[('G', -1)], sum(gn.values())))
    return lam, g, np.array(rows)


def cmd_tb_scan(a):
    from multiprocessing import Pool
    gams = [x * 1e-3 for x in a.gammas]
    tasks = [(lam, g) for lam in (1.0, 0.0) for g in gams]
    t0 = time.time()
    res = Pool(a.jobs).map(_tb_task, tasks) if a.jobs > 1 else [_tb_task(t) for t in tasks]
    out, i0, i30, i60 = {}, ANG.index(0), ANG.index(30), ANG.index(60)
    sub = [i for i, x in enumerate(ANG) if 48 <= x < 60]
    for lam, g, r in res:
        Ic, IcG, Gn = r.T
        R = Ic / Gn / np.pi / (np.pi * D / 2)
        ip, jp = sub[int(np.argmax(Ic[sub]))], sub[int(np.argmax(Gn[sub]))]
        out[f'lam{lam:g}|{g * 1e3:g}'] = dict(ang=ANG, Ic=Ic, Gn=Gn, IcRN_AB=R)
        print(f"SOC {'on ' if lam else 'off'} gamma {g * 1e3:4.1f} meV: G_N peak {ANG[jp]} deg x{Gn[jp] / Gn[i60]:.2f}, I_c peak {ANG[ip]} deg "
              f"x{Ic[ip] / Ic[i60]:.2f}; 0/60: G_N {Gn[i0] / Gn[i60]:.2f}, I_c {Ic[i0] / Ic[i60]:.2f}; I_cR_N/AB 0: {R[i0]:.2f} "
              f"30: {R[i30]:.2f} peak: {R[jp]:.2f} 60: {R[i60]:.2f}; K-pocket share of I_c at the peak {1 - IcG[ip] / Ic[ip]:.4f}")
    print(f'({time.time() - t0:.0f} s)')
    save('tb_scan', out)


def cmd_tb_law(a):
    import junction as jn
    TC = cm.tb_contour(0.0)
    geo = cm.contact(TC)
    us = np.linspace(-4, 1, 501)
    ustar = float(us[np.argmax(cm.scaling_function(us, 'L'))])
    J = jn.TBJunction(0.0, 1.0, nsurf=24000, pockets=('G',))
    TH = np.round(np.arange(48.0, 60.0001, 0.05), 4)
    rows = []
    for g in (0.25e-3, 0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3):
        G = np.array([J.gn_weak(np.radians(x), g) for x in TH])
        i = int(np.argmax(G[:-1]))
        c = np.polyfit(TH[i - 1:i + 2], G[i - 1:i + 2], 2)
        tp = -c[1] / (2 * c[0])
        law = np.degrees(abs(ustar) * abs(geo['fit_4deg']['c3']) ** (1 / 3) / abs(geo['kappa']) * (g / geo['v_contact']) ** (2 / 3))
        rows.append(dict(gamma=g, theta_peak=float(tp), offset=float(60 - tp - geo['eps_c']), height=float(np.polyval(c, tp)),
                         contrast=float(np.polyval(c, tp) / G[-1]), cubic_law=float(law), G=G))
        r = rows[-1]
        print(f"gamma {g * 1e3:5.2f} meV: peak {tp:.3f} deg, offset {r['offset']:.4f} deg, cubic law (local velocity) {law:.4f} deg "
              f"({100 * (law / r['offset'] - 1):+.1f}%), contrast {r['contrast']:.2f}")
    gs, off, hp = (np.array([r[k] for r in rows]) for k in ('gamma', 'offset', 'height'))
    e_off, e_h = np.polyfit(np.log(gs[:4]), np.log(off[:4]), 1)[0], np.polyfit(np.log(gs[:4]), np.log(hp[:4]), 1)[0]
    print(f"delta_c = {geo['eps_c']:.3f} deg, u* = {ustar:.3f}; exponents for gamma <= 2 meV: offset {e_off:.3f}, height {e_h:.3f}")
    save('tb_law', dict(theta=TH, rows=rows, exponent_offset=e_off, exponent_height=e_h))


def cmd_tb_exact(a):
    import junction as jn
    Jall, JG = jn.TBJunction(0.0, 1.0, nsurf=2000), jn.TBJunction(0.0, 1.0, nsurf=2000, pockets=('G',))
    xi = (2e-4, 0.15, 241)
    angs = [0.0, 30.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 60.0]
    out = {}
    for g in [x * 1e-3 for x in a.gammas]:
        for t in [x * 1e-3 for x in a.couplings]:
            Gn, Ic = [], []
            for x in angs:
                J = Jall if x in (0.0, 30.0) else JG         # the K pockets are spin-blocked near 60 deg
                Gn.append(J.gn_exact(np.radians(x), t, g, xi))
                Ic.append(J.ic_exact(np.radians(x), t, g, xi, D))
            Gn, Ic = np.array(Gn), np.array(Ic)
            sub = slice(2, 9)
            out[f'{g * 1e3:g}|{t * 1e3:g}'] = dict(ang=angs, Gn=Gn, Ic=Ic)
            print(f'gamma {g * 1e3:g} meV, t {t * 1e3:g} meV: G_N peak/60 {Gn[sub].max() / Gn[-1]:.2f}, I_c peak/60 {Ic[sub].max() / Ic[-1]:.2f}; '
                  f'0/60: G_N {Gn[0] / Gn[-1]:.2f}, I_c {Ic[0] / Ic[-1]:.2f}', flush=True)
    save('tb_exact', out)


def cmd_tb_umklapp(a):
    import junction as jn
    g = 5e-3
    J = jn.TBJunction(0.0, 1.0, nsurf=2000)
    xi = (D, 300 * D, 161)
    L = lambda x: (g / 2) / np.pi / (x ** 2 + (g / 2) ** 2)
    kern = dict(I_c=jn.ag_kernel(D, g), G_N=lambda x1, x2: L(x1) * L(x2))
    angs = [0.0, 30.0, 48.0, 50.0, 51.0, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5, 56.0, 57.0, 58.0, 59.0, 60.0]
    sub = [i for i, x in enumerate(angs) if 48 <= x < 60]
    out = {}
    for key, k in kern.items():
        sc = np.array([sum(J.channel(np.radians(x), k, xi, 'sc').values()) for x in angs])
        um = np.array([sum(J.channel(np.radians(x), k, xi, 'umk').values()) for x in angs])
        out[key] = dict(ang=angs, sc=sc, umk=um)
        row = [f'{rho:g}: x{((sc + rho ** 2 * um)[sub]).max() / (sc + rho ** 2 * um)[-1]:.2f}' for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)]
        print(f'{key} (gamma 5 meV), split-peak contrast against the umklapp ratio rho_U: ' + ', '.join(row))
    save('tb_umklapp', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    help_ = dict(contact='contact angle, Ising slope and harmonics (TB and DFT contours)',
                 peaks='split-peak positions and contrasts of G_N and I_c, I_cR_N (contour model)',
                 law='broadening law: peak offset and height against gamma, scaling function',
                 window='device imperfections one at a time (DFT NbSe2 by default)',
                 leak='opposite-branch leakage eta (all DFT contours)',
                 joint='all imperfections combined (DFT NbSe2 by default)',
                 maps='angular current density on the Gamma contour (DFT NbSe2 by default)',
                 tb_scan='tight-binding junction: G_N, I_c and I_cR_N from 0 to 60 deg, spin-orbit coupling on and off',
                 tb_law='tight-binding junction: broadening law of the split peak',
                 tb_exact='tight-binding junction: exact in the tunnelling amplitude (strong coupling)',
                 tb_umklapp='tight-binding junction: first-shell umklapp channel')
    for name, h in help_.items():
        p = sp.add_parser(name.replace('_', '-'), help=h)
        p.add_argument('files', nargs='*', help='contour files written by dft_elk.py (default: contours/*.json)')
        if name == 'peaks':
            p.add_argument('--fits', action='store_true', help='also loop over the admissible harmonic truncations')
        if name in ('tb_scan', 'tb_exact'):
            p.add_argument('--gammas', type=float, nargs='+', default=[2.0, 5.0, 10.0] if name == 'tb_exact' else [2.0, 5.0, 10.0, 20.0],
                           help='broadening gamma (meV)')
        if name == 'tb_scan':
            p.add_argument('--jobs', type=int, default=1, help='parallel processes')
        if name == 'tb_exact':
            p.add_argument('--couplings', type=float, nargs='+', default=[1.0, 2.0, 5.0, 10.0, 20.0], help='coupling t (meV)')
    a = ap.parse_args()
    globals()['cmd_' + a.cmd.replace('-', '_')](a)
