#!/usr/bin/env python
# coding: utf-8
# helpers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq
from scipy.special import factorial
from scipy.stats import norm
from pathlib import Path
from collections import Counter
import glob, re

def parse_tau_o(path):
    """Extract the tau_o value from a filename like esctimes_av_tO=0.123.txt"""
    return float(re.search(r'tO=([0-9.eE+\-]+)\.txt$', path).group(1))
def parse_tau_e(path):
    """Extract the tau_o value from a filename like esctimes_av_tO=0.123.txt"""
    return float(re.search(r'tE=([0-9.eE+\-]+)\.txt$', path).group(1))    


# ============================================================
# 11. FPT HELPER FUNCTIONS
# ============================================================

def pooled(D,kind, idx):

    S = D[f'sums_{kind}'][idx].sum(axis=0)
    H = D[f'hist_{kind}'][idx].sum(axis=0)

    return S, H


def raw_moments(sums, dt):
    """
    sums[k] = sum of (passage time in sample steps)^k

    Returns physical raw moments:
    [<t>, <t²>, <t³>, ...]
    """

    k = np.arange(len(sums))

    with np.errstate(invalid='ignore', divide='ignore'):
        return (
            sums / sums[0] * dt**k
        )[1:]


def exponential_bin_pdf(edges, mu):

    widths = np.diff(edges)

    return (
        np.exp(-edges[:-1] / mu)
        * (-np.expm1(-widths / mu))
        / widths
    )

# ============================================================
# 10. LOAD AND POOL ALL FPT FILES FROM PARALLEL JOBs
# MULTIFOLDER!
# ============================================================
def load_fpt_parallel(folders, U0):

    file_info = []

    for date, folder in zip(RUN_DATES, folders):

        pattern = f'U0_{U0:.1f}_fpt_*_job*.npz'

        files = list(folder.glob(pattern))

        print(f'{date}: found {len(files)} FPT files')

        for f in files:

            m = re.search(
                r'fpt_(\d+)_job(\d+)\.npz$',
                f.name
            )

            if m is None:
                raise ValueError(f'Unexpected filename: {f.name}')

            patch_id = int(m.group(1))
            job_id = int(m.group(2))

            file_info.append((date, job_id, patch_id, f))

    if not file_info:
        raise FileNotFoundError('No FPT files found!')

    # Sort by date, job ID, patch ID
    file_info.sort(key=lambda x: (x[0], x[1], x[2]))

    print(f'\nTotal FPT files: {len(file_info)}')

    # Check number of patches per individual job
    job_counts = Counter(
        (date, job_id)
        for date, job_id, patch_id, f in file_info
    )

    print('\nPatches per job:')

    for (date, job_id), count in sorted(job_counts.items()):

        if count != expected_patches_per_job:

            print(
                f'WARNING: {date}, job {job_id}: '
                f'{count} patches '
                f'(expected {expected_patches_per_job})'
            )

    # --------------------------------------------------------
    # Initialize and incrementally accumulate
    # --------------------------------------------------------

    out = {}

    metadata_values = {}

    for i, (date, job_id, patch_id, fn) in enumerate(file_info):

        with np.load(fn) as d:

            if i == 0:

                out['pairs'] = d['pairs'].copy()
                out['dt'] = float(d['dt_sample'])
                out['log_edges'] = d['log_edges'].copy()

                for key in ('tau_m', 'tau_o', 'tau_phi', 'U0'):
                    if key in d.files:
                        metadata_values[key] = []

                for kind in ('all', 'first'):

                    out[f'sums_{kind}'] = np.zeros_like(
                        d[f'sums_{kind}'], dtype=float
                    )

                    out[f'hist_{kind}'] = np.zeros_like(
                        d[f'hist_{kind}'], dtype=np.int64
                    )

            else:

                if not np.array_equal(out['pairs'], d['pairs']):
                    raise ValueError(f'Pair mismatch: {fn}')

                if not np.isclose(out['dt'], float(d['dt_sample'])):
                    raise ValueError(f'dt mismatch: {fn}')

                if not np.allclose(out['log_edges'], d['log_edges']):
                    raise ValueError(f'log_edges mismatch: {fn}')

            # Pool raw data
            for kind in ('all', 'first'):

                out[f'sums_{kind}'] += d[f'sums_{kind}']
                out[f'hist_{kind}'] += d[f'hist_{kind}']

            for key in metadata_values:

                if key in d.files:
                    metadata_values[key].append(float(d[key]))

    # Metadata
    for key, values in metadata_values.items():

        values = np.asarray(values)

        if not np.allclose(values, values[0]):
            print(f'WARNING: {key} varies across files!')

        out[key] = float(values[0])

    print('\nFPT pooling completed.')

    return out


# ============================================================
# 6. STATISTICS OF MARGINAL DISTRIBUTIONS
# ============================================================

def marginal_stats(p, x, dx):

    mean = np.sum(p * x) * dx

    variance = np.sum(
        p * (x - mean)**2
    ) * dx

    return mean, np.sqrt(variance)


def plot_marginal(ax, x, p, dx, mu_ref, sigma_ref,
                  xlabel, ylabel, title, reference_label):

    mean, sigma = marginal_stats(p, x, dx)

    # Simulation
    ax.bar(
        x,
        p,
        width=dx,
        align='center',
        alpha=0.6,
        label='Crossing simulation'
    )

    # Theoretical reference
    xx = np.linspace(x[0], x[-1], 600)

    ax.plot(
        xx,
        norm.pdf(xx, mu_ref, sigma_ref),
        'k--',
        lw=1.5,
        label=reference_label
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(
        f'{title}\n'
        rf'$\langle x\rangle={mean:+.3f}$, '
        rf'$\sigma={sigma:.3f}$'
    )

    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)



RUN_DATES = ['261109', '261110']   # Add as many dates as needed

BASE_PATH = Path(
    '...'
)
RUN_SUFFIX = '...'
path_output = Path('...')

run_dirs = [
    BASE_PATH / f'{date}{RUN_SUFFIX}'
    for date in RUN_DATES
]

pathFPTS_list = [p / 'fpts' for p in run_dirs]
pathPots_list = [p / 'pots' for p in run_dirs]

# Check directories
for p in run_dirs:
    if not p.exists():
        raise FileNotFoundError(f'Missing directory: {p}')

# Save combined analysis in the first run directory
analysis_dir = run_dirs[0] / 'combined_analysis'
analysis_dir.mkdir(exist_ok=True)

tau_m = 0.01
tau_phi = 1.0
tau_o = 1.0
gamma_o = 1.0
U0 = 3.0

expected_jobs = 500
expected_patches_per_job = 100

# Exclude last FPT bin because overflow events are clipped there
exclude_last_fpt_bin = True



# 1. LOAD AND POOL AUXILIARY HISTOGRAMS - takes 10min
# ============================================================

# ============================================================
# 1. LOAD AUXILIARY HISTOGRAMS FROM ALL DATES
# ============================================================

aux_files = []

for date, folder in zip(RUN_DATES, pathPots_list):

    pattern = (
        f'{date}P_zvw_x_pm1_'
        f'tau_m={tau_m:.3f},tau_phi={tau_phi:.3f},'
        f'tau_o={tau_o:.3f}_job*.npz'
    )

    files = list(folder.glob(pattern))

    print(f'{date}: found {len(files)} auxiliary files')

    aux_files.extend(files)


# Sort by date and job ID
aux_files.sort(
    key=lambda f: (
        f.parent.parent.name[:6],
        int(re.search(r'_job(\d+)\.npz$', f.name).group(1))
    )
)

if not aux_files:
    raise FileNotFoundError('No auxiliary files found!')

print(f'\nTotal auxiliary files: {len(aux_files)}')

Hm_raw = None
Hp_raw = None

bz = bv = bw = None

job_weights = []

for fn in aux_files:

    with np.load(fn) as d:

        # Only accept raw histograms!
        if 'hist_is_raw' not in d.files or not bool(d['hist_is_raw']):
            raise RuntimeError(
                f'{fn.name} does not contain a raw histogram. '
                'Per-job normalized histograms cannot be pooled '
                'correctly without their original total weights.'
            )

        Hm_i = np.asarray(d['Hm'], dtype=float)
        Hp_i = np.asarray(d['Hp'], dtype=float)

        bz_i = d['bz']
        bv_i = d['bv']
        bw_i = d['bw']

        # First job: initialize
        if Hm_raw is None:

            Hm_raw = np.zeros_like(Hm_i)
            Hp_raw = np.zeros_like(Hp_i)

            bz = bz_i.copy()
            bv = bv_i.copy()
            bw = bw_i.copy()

        else:

            # Ensure identical grids across jobs
            if not (
                np.allclose(bz, bz_i)
                and np.allclose(bv, bv_i)
                and np.allclose(bw, bw_i)
            ):
                raise ValueError(
                    f'Histogram bin edges differ in {fn.name}'
                )

            if Hm_i.shape != Hm_raw.shape:
                raise ValueError(
                    f'Histogram shape mismatch in {fn.name}'
                )

        # THE IMPORTANT PART:
        # Add RAW weighted histograms across jobs
        Hm_raw += Hm_i
        Hp_raw += Hp_i

        job_weights.append((
            float(Hm_i.sum()),
            float(Hp_i.sum())
        ))


# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

job_ids = [
    int(re.search(r'_job(\d+)\.npz$', f.name).group(1))
    for f in aux_files
]

print(f'Job IDs: {min(job_ids)} ... {max(job_ids)}')
print(f'Missing job files: {expected_jobs - len(aux_files)}')

print(f'Raw accumulated weight, x=-1: {Hm_raw.sum():.8e}')
print(f'Raw accumulated weight, x=+1: {Hp_raw.sum():.8e}')


# ============================================================
# 2. NORMALIZE ONLY AFTER POOLING ALL JOBS
# ============================================================

dz_all = np.diff(bz)
dv_all = np.diff(bv)
dw_all = np.diff(bw)

# Check uniform grid spacing
assert np.allclose(dz_all, dz_all[0])
assert np.allclose(dv_all, dv_all[0])
assert np.allclose(dw_all, dw_all[0])

dz = dz_all[0]
dv = dv_all[0]
dw = dw_all[0]

dV = dz * dv * dw

norm_m = Hm_raw.sum() * dV
norm_p = Hp_raw.sum() * dV

if norm_m <= 0 or norm_p <= 0:
    raise RuntimeError('One of the pooled histograms is empty.')

Hm = Hm_raw / norm_m
Hp = Hp_raw / norm_p

print('\nNormalization after pooling:')
print(f'Integral Hm = {Hm.sum()*dV:.10f}')
print(f'Integral Hp = {Hp.sum()*dV:.10f}')


# ============================================================
# 3. BIN CENTERS
# ============================================================

zc = 0.5 * (bz[1:] + bz[:-1])
vc = 0.5 * (bv[1:] + bv[:-1])
wc = 0.5 * (bw[1:] + bw[:-1])


# ============================================================
# 4. 1D MARGINAL DISTRIBUTIONS
# ============================================================

# x = -1
pz_m = Hm.sum(axis=(1, 2)) * dv * dw
pv_m = Hm.sum(axis=(0, 2)) * dz * dw
pw_m = Hm.sum(axis=(0, 1)) * dz * dv

# x = +1
pz_p = Hp.sum(axis=(1, 2)) * dv * dw
pv_p = Hp.sum(axis=(0, 2)) * dz * dw
pw_p = Hp.sum(axis=(0, 1)) * dz * dv


# ============================================================
# 5. THEORETICAL INITIAL DISTRIBUTIONS
# ============================================================

m_o = gamma_o * tau_o / 2.0

k_y = m_o * (
    1.0 / tau_o**2 +
    1.0 / tau_phi**2
)

sig_v = np.sqrt(1.0 / tau_m)
sig_w = np.sqrt(1.0 / m_o)
sig_z = np.sqrt(1.0 / k_y)

mu_z_m = -1.0
mu_z_p = +1.0

##PLOT - aux distr.
# =========== =================================================
# 7. MARGINALS FOR x = -1
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

plot_marginal(
    axes[0], zc, pz_m, dz,
    mu_z_m, sig_z,
    'z', r'$P(z\mid x=-1)$',
    r'$z$',
    rf'$\mathcal{{N}}(-1,{sig_z:.2f}^2)$ -- initial reference'
)

plot_marginal(
    axes[1], vc, pv_m, dv,
    0.0, sig_v,
    'v', r'$P(v\mid x=-1)$',
    r'$v$',
    rf'$\mathcal{{N}}(0,{sig_v:.2f}^2)$ -- initial reference'
)

plot_marginal(
    axes[2], wc, pw_m, dw,
    0.0, sig_w,
    'w', r'$P(w\mid x=-1)$',
    r'$w$',
    rf'$\mathcal{{N}}(0,{sig_w:.2f}^2)$ -- initial reference'
)

fig.suptitle(
    rf'Crossing distributions at $x=-1$ | '
    rf'$\tau_m={tau_m}$, $\tau_o={tau_o}$'
)

plt.tight_layout()
plt.savefig(
    path_output / 'marginals_xm_all_jobs.png',
    dpi=180
)
plt.show()


# ============================================================
# 8. MARGINALS FOR x = +1
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

plot_marginal(
    axes[0], zc, pz_p, dz,
    mu_z_p, sig_z,
    'z', r'$P(z\mid x=+1)$',
    r'$z$',
    rf'$\mathcal{{N}}(+1,{sig_z:.2f}^2)$ -- local-well reference'
)

plot_marginal(
    axes[1], vc, pv_p, dv,
    0.0, sig_v,
    'v', r'$P(v\mid x=+1)$',
    r'$v$',
    rf'$\mathcal{{N}}(0,{sig_v:.2f}^2)$ -- initial reference'
)

plot_marginal(
    axes[2], wc, pw_p, dw,
    0.0, sig_w,
    'w', r'$P(w\mid x=+1)$',
    r'$w$',
    rf'$\mathcal{{N}}(0,{sig_w:.2f}^2)$ -- initial reference'
)

fig.suptitle(
    rf'Crossing distributions at $x=+1$ | '
    rf'$\tau_m={tau_m}$, $\tau_o={tau_o}$'
)

plt.tight_layout()
plt.savefig(
    path_output / 'marginals_xp_all_jobs.png',
    dpi=180
)
plt.show()


# ============================================================
# 9. VELOCITY COMPARISON: x=-1 AND x=+1
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, pv, tag in [
    (axes[0], pv_m, 'x=-1'),
    (axes[1], pv_p, 'x=+1')
]:

    mean, sigma = marginal_stats(pv, vc, dv)

    ax.bar(
        vc, pv,
        width=dv,
        align='center',
        alpha=0.6,
        label='Crossing simulation'
    )

    ax.plot(
        vc,
        norm.pdf(vc, 0, sig_v),
        'k--',
        lw=1.5,
        label=rf'$\mathcal{{N}}(0,{sig_v:.2f}^2)$'
    )

    ax.set_xlabel('v')

    ax.set_title(
        f'P(v | {tag})\n'
        rf'$\langle v\rangle={mean:+.3f}$, '
        rf'$\sigma={sigma:.3f}$'
    )

    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_ylabel('P(v)')
axes[0].set_yscale('log')

plt.tight_layout()
plt.savefig(
    path_output / 'marginals_v_xpm_all_jobs.png',
    dpi=180
)
plt.show()


#FPT dists - GAMMA_OSC
D_OSC = load_fpt_parallel(pathFPTS_list, U0)

# ============================================================
# 12. FPT TIME BINNING
# ============================================================

dt = D_OSC['dt']
log_edges = D_OSC['log_edges']

t_edges = dt * np.exp(log_edges)

t_c = np.sqrt(
    t_edges[:-1] * t_edges[1:]
)

bin_widths = np.diff(t_edges)

dlnt = np.diff(log_edges)


# ============================================================
# 13. SELECT TRANSITIONS
# ============================================================

pairs = [
    tuple(p) for p in D_OSC['pairs']
]

print('\nFPT transition pairs:')
print(pairs)

transition_idx = [
    i for i, (s, e) in enumerate(pairs)
    if np.sign(s) != np.sign(e)
]

recross_idx = [
    i for i, (s, e) in enumerate(pairs)
    if np.sign(s) == np.sign(e)
]

groups = {
    'transition ±1→∓1': transition_idx,
    'recrossings ±1→±1': recross_idx
}

kinds = (
    ('all', 'C0'),
    ('first', 'C1')
)

labels_FP = {
    'all': r'$\exp\{-t/\tau_{\mathrm{MAFP}}\}/\tau_{\mathrm{MAFP}}$',
    'first': r'$\exp\{-t/\tau_{\mathrm{MFFP}}\}/\tau_{\mathrm{MFFP}}$'
}


# ============================================================
# 14. FPT PDF + EXPONENTIAL REFERENCES
# ============================================================

fig, axs = plt.subplots(2, 2, figsize=(10, 8.5))

pdfs = {}

for ax, (name, gidx) in zip(axs[0], groups.items()):

    for kind, c in kinds:

        S, H = pooled(D_OSC,kind, gidx)

        N_fpt = S[0]

        if N_fpt <= 0:
            continue

        # Exact PDF per unit physical time
        pdf = H / (N_fpt * bin_widths)

        err = np.sqrt(H) / (N_fpt * bin_widths)

        # Mean passage time
        mn = raw_moments(S, dt)
        mu = mn[0]

        pdfs[(kind, name)] = pdf

        # Exclude clipped overflow bin
        valid = H > 0

        if exclude_last_fpt_bin:
            valid[-1] = False

        ax.errorbar(
            t_c[valid],
            pdf[valid],
            yerr=err[valid],
            fmt='o-',
            ms=3,
            capsize=2,
            c=c,
            label=f'{kind}-FPT'
        )

        # Exponential reference
        pdf_exp = exponential_bin_pdf(t_edges, mu)

        ref_mask = np.ones(len(t_c), dtype=bool)

        if exclude_last_fpt_bin:
            ref_mask[-1] = False

        ax.loglog(
            t_c[ref_mask],
            pdf_exp[ref_mask],
            '--',
            c=c,
            lw=1,
            label=labels_FP[kind]
        )

        print(
            f'{name}, {kind}: '
            f'N={N_fpt:.0f}, mean={mu:.5g}'
        )

    ax.set_title(name)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$p(t)$')

    ax.set_ylim([1e-13, 1e2])

    ax.grid(alpha=0.25, which='both')


axs[0, 0].legend(fontsize=8)


# ============================================================
# 15. RAW MOMENTS + EXPONENTIAL REFERENCE
# ============================================================

idx = transition_idx

for kind, c in kinds:

    S, H = pooled(D_OSC,kind, idx)

    mn = raw_moments(S, dt)

    ns = np.arange(1, len(mn) + 1)

    # <t^n>^(1/n)
    axs[1, 0].semilogy(
        ns,
        mn**(1.0 / ns),
        'o-',
        c=c,
        label=f'{kind}'
    )

    # Exponential reference:
    # <t^n> = n! * mu^n
    mu = mn[0]

    exp_moments = np.array([
        factorial(int(n))**(1.0 / n) * mu
        for n in ns
    ])

    axs[1, 0].semilogy(
        ns,
        exp_moments,
        ':',
        c=c,
        label=rf'Exponential {kind}'
    )


axs[1, 0].set_xlabel('n')
axs[1, 0].set_ylabel(
    r'$\langle t^n\rangle^{1/n}$'
)

axs[1, 0].legend(fontsize=8)
axs[1, 0].grid(alpha=0.25)


# ============================================================
# 16. INTEGRAND PER ln(t)
# ============================================================

for n_, c in zip(
    (1, 2, 3, 4),
    ('C0', 'C1', 'C2', 'C3')
):

    for kind, ls in (
        ('first', '-'),
        ('all', '--')
    ):

        pdf = pdfs[(kind, 'transition ±1→∓1')]

        y = t_c**(n_ + 1) * pdf

        mask = y > 0

        if exclude_last_fpt_bin:
            mask[-1] = False

        axs[1, 1].loglog(
            t_c[mask],
            y[mask],
            c=c,
            ls=ls,
            label=f'{kind}, n={n_}'
        )


axs[1, 1].set_xlabel(r'$t$')
axs[1, 1].set_ylabel(
    r'$t^{n+1}p(t)$'
)

axs[1, 1].legend(fontsize=8)
axs[1, 1].grid(alpha=0.25, which='both')


# ============================================================
# 17. FIGURE FORMATTING AND SAVING
# ============================================================

fig.suptitle(
    rf'FPT distributions | '
    rf'$\tau_m={D_OSC.get("tau_m", tau_m)}$, '
    rf'$\tau_\phi={D_OSC.get("tau_phi", tau_phi)}$, '
    rf'$\tau_o={D_OSC.get("tau_o", tau_o)}$, '
    rf'$U_0={U0}$',
    fontsize=13
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    path_output / 'FPT_analysis_all_jobs.png',
    dpi=180,
    bbox_inches='tight'
)

plt.show()
