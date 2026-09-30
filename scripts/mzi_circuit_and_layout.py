"""
Passive unbalanced MZI — circuit-level simulation (sax) + layout (gdsfactory).

Uses the waveguide properties measured by mzi_smoke_test.py (MPB mode solve,
500 x 220 nm SOI strip, TE0, 1550 nm):

    n_eff(1550 nm) = 2.4445
    dn_eff/dlambda = -1.0392 /um      ->  n_g = n_eff - lambda * dn_eff/dlambda = 4.0552

The straight-waveguide model uses a *dispersive* n_eff(lambda) (linear about
1550 nm), so the simulated fringe spacing is governed by n_g, not n_eff.
That is the physics being tested: the simulated FSR should match
FSR = lambda^2 / (n_g * dL).

Idealizations (see README "Scope and assumptions"):
    - couplers are ideal lossless 50/50 (2x2, 90-degree cross-coupling phase)
    - arms are lossless
    - constant material indices were used in the upstream mode solve

Outputs (written to ../results/ relative to this script):
    mzi_spectrum_dL<dL>um.png     bar/cross transmission, full sweep + zoom
    mzi_fsr_vs_dL.png             simulated vs analytic FSR for each dL
    mzi_dL<dL>um.gds              gdsfactory layout (generic PDK), open in KLayout
    mzi_circuit_results.txt       summary table (same as console output)

Run:
    conda activate pymeep
    python scripts/mzi_circuit_and_layout.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import sax

# ----------------------------------------------------------------------------
# Measured waveguide properties (from mzi_smoke_test.py)
# ----------------------------------------------------------------------------
LAMBDA0 = 1.550  # um
NEFF0 = 2.4445  # effective index at 1550 nm
DNEFF_DLAM = -1.0392  # per um
NG = NEFF0 - LAMBDA0 * DNEFF_DLAM  # group index, ~4.0552

# ----------------------------------------------------------------------------
# Device and sweep parameters
# ----------------------------------------------------------------------------
DELTA_L_UM = [50, 100, 200]  # arm-length differences to simulate
L_REF = 100.0  # reference (short) arm length, um; cancels out of the physics
WL = np.linspace(1.500, 1.600, 20001)  # um, dense enough to resolve every fringe

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Component models (S-parameters)
# ----------------------------------------------------------------------------
def neff(wl):
    """Dispersive effective index, linear about 1550 nm (from the MPB solve)."""
    return NEFF0 + DNEFF_DLAM * (wl - LAMBDA0)


def straight(wl=LAMBDA0, length=10.0):
    """Lossless straight waveguide: pure propagation phase."""
    phase = 2 * jnp.pi * neff(wl) * length / wl
    return sax.reciprocal({("o1", "o2"): jnp.exp(1j * phase)})


def coupler(wl=LAMBDA0):
    """Ideal lossless 50/50 2x2 coupler.
    Straight-through amplitude 1/sqrt2, cross amplitude j/sqrt2
    (the 90-degree cross phase is what makes the coupler lossless/unitary).
    Ports: o1, o2 = inputs (left); o3, o4 = outputs (right).
    """
    t = 2**-0.5
    k = 1j * 2**-0.5
    return sax.reciprocal(
        {
            ("o1", "o4"): t,  # o1 -> upper output, straight through
            ("o1", "o3"): k,  # o1 -> lower output, crossed
            ("o2", "o3"): t,
            ("o2", "o4"): k,
        }
    )


def mzi_netlist(delta_l):
    """Splitter -> two arms (L and L + dL) -> combiner."""
    return {
        "instances": {
            "split": "coupler",
            "arm_short": {"component": "straight", "settings": {"length": L_REF}},
            "arm_long": {"component": "straight", "settings": {"length": L_REF + delta_l}},
            "combine": "coupler",
        },
        "connections": {
            "split,o4": "arm_short,o1",
            "split,o3": "arm_long,o1",
            "arm_short,o2": "combine,o2",
            "arm_long,o2": "combine,o1",
        },
        "ports": {
            "in": "split,o1",
            "out_a": "combine,o4",
            "out_b": "combine,o3",
        },
    }


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------
def peak_wavelengths(wl, t):
    """Local maxima of a transmission curve (strictly interior points)."""
    idx = np.where((t[1:-1] > t[:-2]) & (t[1:-1] >= t[2:]))[0] + 1
    return wl[idx]


def fsr_at(wl0, peaks):
    """Spacing of the two adjacent peaks that straddle wl0, and their midpoint."""
    right = np.searchsorted(peaks, wl0)
    p1, p2 = peaks[right - 1], peaks[right]
    return p2 - p1, 0.5 * (p1 + p2)


def to_db(t):
    return 10 * np.log10(np.maximum(t, 1e-12))


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
def write_layouts():
    """Generic-PDK MZI layout for each dL. Isolated so a layout/PDK problem
    cannot block the circuit results."""
    try:
        import gdsfactory as gf

        gf.gpdk.PDK.activate()  # the fix identified in the smoke test
        paths = []
        for dl in DELTA_L_UM:
            c = gf.components.mzi(delta_length=float(dl))
            path = RESULTS / f"mzi_dL{dl}um.gds"
            c.write_gds(path)
            paths.append(path.name)
        return f"OK  ({', '.join(paths)})"
    except Exception as exc:  # report and carry on
        return f"SKIPPED  ({type(exc).__name__}: {exc})"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 68)
    log("  Passive unbalanced MZI — sax circuit simulation + layout")
    log("=" * 68)
    log(f"  n_eff(1550) = {NEFF0:.4f}   dn_eff/dlambda = {DNEFF_DLAM:.4f} /um   n_g = {NG:.4f}")
    log(f"  sweep {WL[0]*1e3:.0f}-{WL[-1]*1e3:.0f} nm, {WL.size} points; ideal 50/50 couplers, lossless arms")
    log("-" * 68)

    models = {"coupler": coupler, "straight": straight}
    rows = []

    for dl in DELTA_L_UM:
        circuit, _ = sax.circuit(netlist=mzi_netlist(dl), models=models)
        s = circuit(wl=WL)
        t_a = np.asarray(jnp.abs(s["in", "out_a"]) ** 2)
        t_b = np.asarray(jnp.abs(s["in", "out_b"]) ** 2)

        # Sanity: energy conservation for a lossless circuit
        energy_err = np.max(np.abs(t_a + t_b - 1.0))

        # Simulated FSR from peak spacing around 1550 nm
        peaks = peak_wavelengths(WL, t_a)
        fsr_sim, wl_mid = fsr_at(LAMBDA0, peaks)
        fsr_pred = wl_mid**2 / (NG * dl)  # analytic, evaluated at the same wavelength
        fsr_wrong = wl_mid**2 / (NEFF0 * dl)  # what n_eff would (wrongly) predict
        diff_pct = (fsr_sim - fsr_pred) / fsr_pred * 100

        # Idealized ER and IL (expected: very large ER, ~0 dB IL)
        in_band = np.abs(WL - LAMBDA0) < 2 * fsr_sim
        er_db = to_db(t_a[in_band].max()) - to_db(t_a[in_band].min())
        il_db = -to_db(t_a[in_band].max())

        rows.append((dl, fsr_sim, fsr_pred, diff_pct, fsr_wrong))
        log(f"  dL = {dl:3d} um")
        log(f"    FSR simulated (sax)       = {fsr_sim*1e3:7.3f} nm   (peaks around {wl_mid*1e3:.2f} nm)")
        log(f"    FSR analytic  (n_g)       = {fsr_pred*1e3:7.3f} nm   difference {diff_pct:+.2f} %")
        log(f"    FSR if n_eff were used    = {fsr_wrong*1e3:7.3f} nm   (wrong formula, for contrast)")
        log(f"    ER (ideal)                = {er_db:7.1f} dB    IL (ideal) = {il_db:.3f} dB")
        log("      (ideal ER is infinite; the finite value is set by where sweep samples fall near the nulls)")
        log(f"    max |T_a + T_b - 1|       = {energy_err:.1e}   (energy conservation)")

        # Spectrum plot: full sweep + zoom around 1550 nm
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5))
        for ax in (ax1, ax2):
            ax.plot(WL * 1e3, to_db(t_a), lw=1, label="port A (bar)")
            ax.plot(WL * 1e3, to_db(t_b), lw=1, label="port B (cross)", alpha=0.8)
            ax.set_ylabel("Transmission (dB)")
            ax.set_ylim(-40, 2)
            ax.grid(alpha=0.3)
        ax1.set_title(f"Passive unbalanced MZI, ΔL = {dl} µm — sax, ideal couplers")
        ax1.legend(loc="lower right")
        half = 1.5 * fsr_sim * 1e3
        ax2.set_xlim(LAMBDA0 * 1e3 - half, LAMBDA0 * 1e3 + half)
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_title(
            f"Zoom: FSR simulated {fsr_sim*1e3:.3f} nm vs analytic λ²/(n_g·ΔL) {fsr_pred*1e3:.3f} nm",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(RESULTS / f"mzi_spectrum_dL{dl}um.png", dpi=150)
        plt.close(fig)

    # FSR vs dL summary plot
    dls = np.array([r[0] for r in rows], float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dl_fine = np.linspace(dls.min() * 0.8, dls.max() * 1.1, 200)
    ax.plot(dl_fine, LAMBDA0**2 / (NG * dl_fine) * 1e3, label="analytic λ²/(n_g·ΔL)")
    ax.plot(dl_fine, LAMBDA0**2 / (NEFF0 * dl_fine) * 1e3, "--", label="λ²/(n_eff·ΔL) (wrong)")
    ax.plot(dls, [r[1] * 1e3 for r in rows], "o", ms=8, label="simulated (sax)")
    ax.set_xlabel("Arm-length difference ΔL (µm)")
    ax.set_ylabel("FSR (nm)")
    ax.set_title("FSR vs ΔL — the fringe spacing follows n_g, not n_eff")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "mzi_fsr_vs_dL.png", dpi=150)
    plt.close(fig)

    log("-" * 68)
    log(f"  Layout (gdsfactory, generic PDK): {write_layouts()}")
    log(f"  Plots and summary written to: {RESULTS}")
    log("=" * 68)

    (RESULTS / "mzi_circuit_results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
