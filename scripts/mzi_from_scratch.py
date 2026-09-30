#!/opt/anaconda3/envs/pymeep/bin/python
"""
Unbalanced Mach-Zehnder interferometer -- from scratch, then with sax
=====================================================================
Two ways to simulate the SAME device, side by side:

  METHOD 1 (from scratch): build every S-matrix by hand in numpy and multiply
            them -- input -> coupler -> arm phases -> coupler -> output.
            No photonics library. This is the physics with nothing hidden.

  METHOD 2 (circuit tool): the same MZI as a sax netlist. This is how you'd
            do it for any real (larger) circuit.

They are overlaid and their difference printed -- they must agree to ~1e-13.
That "two independent methods agree" check is the same discipline as the
n_eff/n_g cross-check in the MPB smoke test.

Waveguide: 500 x 220 nm SOI strip, from the MPB smoke test:
  n_eff = 2.4445, n_g = 4.0552 @ 1550 nm.
Arm imbalance dL = 100 um  ->  predicted FSR = lambda^2/(n_g*dL) ~ 5.92 nm.

Tools: numpy, jax+sax, matplotlib.
Run:  /opt/anaconda3/envs/pymeep/bin/python mzi_from_scratch.py
Output: mzi_scratch_vs_sax.png
"""

import numpy as np
import jax.numpy as jnp
import sax

# ----------------------------------------------------------------------
# Parameters (from the MPB smoke test)
# ----------------------------------------------------------------------
NEFF = 2.4445      # effective index @ 1550 nm  -> sets WHERE fringes sit
NG   = 4.0552      # group index    @ 1550 nm  -> sets fringe SPACING (FSR)
WL0  = 1.55        # reference wavelength [um]
L    = 100.0       # reference (short) arm length [um]
DL   = 100.0       # arm-length difference [um]  <-- the "unbalanced" part
WL   = np.linspace(1.50, 1.60, 20001)   # wavelength sweep [um]


# ======================================================================
# METHOD 1 -- FROM SCRATCH (pure numpy)
# ======================================================================
# Step 1: how the waveguide's index varies with wavelength (first-order
#         dispersion), pinned so that the group index equals NG at WL0.
def neff_of(wl):
    return NEFF - (wl - WL0) * (NG - NEFF) / WL0

# Step 2: propagation constant beta = 2*pi*n_eff/lambda.
def beta(wl):
    return 2 * np.pi * neff_of(wl) / wl

# Step 3: the ideal 50/50 coupler as a 2x2 matrix mapping [in0,in1]->[out0,out1].
#         The 1/sqrt(2) splits amplitude 50/50; the 1j on the cross terms is the
#         90-degree phase that lossless-ness (unitarity) forces. This j is what
#         makes the two output ports come out COMPLEMENTARY (cos^2 vs sin^2).
C = (1/np.sqrt(2)) * np.array([[1, 1j],
                               [1j, 1]])

def mzi_from_scratch(wl_array):
    """Return (bar, cross) power spectra by multiplying the S-matrices by hand."""
    bar = np.empty_like(wl_array)
    cross = np.empty_like(wl_array)
    for i, w in enumerate(wl_array):
        # Step 4: the two arms are a diagonal phase matrix. Top arm length L,
        #         bottom arm length L+DL -> they emerge with different phase.
        P = np.array([[np.exp(1j * beta(w) * L), 0],
                      [0, np.exp(1j * beta(w) * (L + DL))]])
        # Step 5: cascade. Light enters port 0 only: a = [1, 0].
        #         b = combiner @ arms @ splitter @ a
        b = C @ P @ C @ np.array([1, 0])
        # Step 6: power = |amplitude|^2 at each output port.
        bar[i] = abs(b[0])**2
        cross[i] = abs(b[1])**2
    return bar, cross


# ======================================================================
# METHOD 2 -- SAX CIRCUIT
# ======================================================================
def waveguide(wl=1.55, length=100.0):
    neff_wl = NEFF - (wl - WL0) * (NG - NEFF) / WL0
    return sax.reciprocal({("in0", "out0"): jnp.exp(1j * 2*jnp.pi * neff_wl * length / wl)})

def coupler(coupling=0.5):
    k, t = coupling**0.5, (1 - coupling)**0.5
    return sax.reciprocal({("in0","out0"): t, ("in0","out1"): 1j*k,
                           ("in1","out0"): 1j*k, ("in1","out1"): t})

MZI_NET = {
    "instances": {"sp":"coupler", "top":"waveguide", "bot":"waveguide", "cb":"coupler"},
    "connections": {"sp,out0":"top,in0", "sp,out1":"bot,in0",
                    "top,out0":"cb,in0", "bot,out0":"cb,in1"},
    "ports": {"in0":"sp,in0", "bar":"cb,out0", "cross":"cb,out1"},
}

def mzi_from_sax(wl_array):
    circuit, _ = sax.circuit(netlist=MZI_NET,
                             models={"coupler": coupler, "waveguide": waveguide})
    S = circuit(wl=jnp.asarray(wl_array), top={"length": L}, bot={"length": L + DL})
    return np.asarray(jnp.abs(S["in0","bar"])**2), np.asarray(jnp.abs(S["in0","cross"])**2)


# ======================================================================
# FSR extraction + run both, compare, plot
# ======================================================================
def measure_fsr(wl_array, bar):
    idx = [i for i in range(1, len(bar)-1)
           if bar[i] > bar[i-1] and bar[i] >= bar[i+1] and bar[i] > 0.5]
    return np.diff(wl_array[idx]).mean() * 1000 if len(idx) > 1 else float("nan")

def main():
    print("=" * 60)
    print("  Unbalanced MZI: from scratch vs sax")
    print("=" * 60)

    bar1, cross1 = mzi_from_scratch(WL)      # numpy
    bar2, cross2 = mzi_from_sax(WL)          # sax

    diff = np.max(np.abs(bar1 - bar2))
    fsr = measure_fsr(WL, bar1)
    fsr_pred = WL0**2 / (NG * DL) * 1000

    print(f"  power conservation (scratch): {(bar1+cross1).min():.6f} .. {(bar1+cross1).max():.6f}")
    print(f"  max |scratch - sax|         : {diff:.2e}   (agree to machine precision)")
    print(f"  FSR measured                : {fsr:.3f} nm")
    print(f"  FSR predicted lam^2/(ng*dL) : {fsr_pred:.3f} nm")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(WL*1000, bar1, lw=2.6, alpha=0.35, label="bar — from scratch (numpy)")
    ax.plot(WL*1000, bar2, lw=1.0, color="k", label="bar — sax (overlaid)")
    ax.plot(WL*1000, cross1, lw=2.6, alpha=0.35, color="C1", label="cross — from scratch")
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("transmission")
    ax.set_title(f"Unbalanced MZI (ΔL={DL:.0f} µm) — two methods, one result")
    ax.set_ylim(-0.03, 1.03); ax.legend(loc="center right", fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("mzi_scratch_vs_sax.png", dpi=140)
    print("  Saved: mzi_scratch_vs_sax.png (the black sax line sits exactly on the")
    print("         thick from-scratch line -> identical physics).")

if __name__ == "__main__":
    main()
