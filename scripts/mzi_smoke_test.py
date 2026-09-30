#!/opt/anaconda3/envs/pymeep/bin/python
"""
MZI pipeline smoke test
=======================
Goal: prove the whole silicon-photonics stack talks to itself end to end, and
get the ONE number the MZI spectrum hangs off -- the group index n_g -- from
YOUR waveguide rather than a textbook value.

What it exercises:
  gdsfactory  -> lay out a straight strip waveguide, write GDS
  KLayout     -> read that GDS back (round-trip sanity)
  MPB         -> solve the fundamental mode, get n_eff (and n_g)

n_g is computed TWO independent ways and cross-checked:
  (A) finite difference of n_eff over wavelength  ->  n_g = n_eff - lambda * dn_eff/dlambda
      (this is literally the interview formula; only needs the mode solve)
  (B) MPB's native group velocity                 ->  n_g = c / v_g = 1 / v_g[c-units]
If (A) and (B) agree to ~1%, every piece is trustworthy.

Geometry: 500 nm x 220 nm silicon strip, fully oxide-clad, TE-like fundamental,
around 1550 nm. Edit WIDTH/THICKNESS/N_SI/N_SIO2 to match your PDK.

Run:  /opt/anaconda3/envs/pymeep/bin/python mzi_smoke_test.py
"""

import numpy as np

# ----------------------------------------------------------------------
# Waveguide + sim parameters  (edit these to match your platform)
# ----------------------------------------------------------------------
WIDTH      = 0.50     # core width  [um]
THICKNESS  = 0.22     # core height [um]
N_SI       = 3.48     # silicon index @ ~1550 nm
N_SIO2     = 1.44     # oxide cladding index
LAMBDA0    = 1.55     # target vacuum wavelength [um]
DLAMBDA    = 0.01     # step for the finite-difference n_g [um]

WINDOW     = 3.0      # cross-section sim window (x = y) [um]
RESOLUTION = 64       # MPB pixels per um
DELTA_L_UM = [50, 100, 200]   # arm length differences to report FSR for [um]

C0 = 299792458.0      # m/s, only used for a friendly comment; MPB uses c = 1


# ----------------------------------------------------------------------
# 1. MPB mode solve: n_eff at a given wavelength (+ native group velocity)
# ----------------------------------------------------------------------
def solve_mode(wavelength_um, want_vg=False):
    """Return n_eff (real) of the fundamental mode; optionally n_g from MPB v_g."""
    import meep as mp
    from meep import mpb

    fcen = 1.0 / wavelength_um            # MPB frequency in c/a units, a = 1 um

    lattice = mp.Lattice(size=mp.Vector3(WINDOW, WINDOW, 0))
    geometry = [mp.Block(mp.Vector3(WIDTH, THICKNESS, mp.inf),
                         material=mp.Medium(index=N_SI))]

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.Medium(index=N_SIO2),   # oxide fills the rest
        resolution=RESOLUTION,
        num_bands=1,
    )

    # Find the propagation constant k (along z) that yields frequency fcen.
    kmags = ms.find_k(
        mp.NO_PARITY,             # fundamental band; no parity constraint
        fcen,                     # solve for this frequency
        1, 1,                     # band_min, band_max
        mp.Vector3(0, 0, 1),      # k direction: propagate along z
        1e-4,                     # tolerance
        fcen * 2.5,               # kmag guess (~ n_eff * f)
        fcen * 1.3,               # kmag min  (n_eff > cladding)
        fcen * 3.5,               # kmag max  (n_eff < core)
    )
    kmag = kmags[0]
    neff = kmag / fcen            # n_eff = beta / k0 = k_mpb / f_mpb

    ng_mpb = None
    if want_vg:
        try:
            vg = ms.compute_group_velocities()   # list of Vector3, units of c
            vgz = vg[0].z
            if vgz != 0:
                ng_mpb = 1.0 / vgz               # n_g = c / v_g
        except Exception as e:
            print(f"   (MPB group-velocity call unavailable: {e})")

    return neff, ng_mpb


# ----------------------------------------------------------------------
# 2. gdsfactory layout + KLayout round-trip (non-fatal if it hiccups)
# ----------------------------------------------------------------------
def layout_and_roundtrip():
    try:
        import gdsfactory as gf
        xs = gf.cross_section.strip(width=WIDTH)
        c = gf.components.straight(length=10.0, cross_section=xs)
        gdspath = c.write_gds("strip_smoke.gds")
        print(f"[gdsfactory] wrote {gdspath}  ({c.name})")

        try:
            import klayout.db as kdb
            ly = kdb.Layout()
            ly.read(str(gdspath))
            top = ly.top_cell()
            print(f"[KLayout]    read back GDS: top cell '{top.name}', "
                  f"{top.bbox().width()/ly.dbu:.0f} x {top.bbox().height()/ly.dbu:.0f} dbu bbox")
        except Exception as e:
            print(f"[KLayout]    readback skipped: {e}")
    except Exception as e:
        print(f"[gdsfactory] layout step skipped: {e}")


# ----------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  MZI pipeline smoke test")
    print("=" * 60)
    print(f"  strip {WIDTH*1e3:.0f} x {THICKNESS*1e3:.0f} nm, "
          f"n_Si={N_SI}, n_SiO2={N_SIO2}, lambda0={LAMBDA0*1e3:.0f} nm")
    print("-" * 60)

    layout_and_roundtrip()
    print("-" * 60)

    # n_eff at three wavelengths -> finite-difference n_g (method A)
    print("[MPB] solving fundamental mode at 3 wavelengths ...")
    neff_lo, _        = solve_mode(LAMBDA0 - DLAMBDA)
    neff_0,  ng_mpb   = solve_mode(LAMBDA0, want_vg=True)   # method B here too
    neff_hi, _        = solve_mode(LAMBDA0 + DLAMBDA)

    dneff_dlam = (neff_hi - neff_lo) / (2 * DLAMBDA)        # per um
    ng_fd = neff_0 - LAMBDA0 * dneff_dlam                   # n_g = n_eff - lam*dn/dlam

    print("-" * 60)
    print(f"  n_eff (@ {LAMBDA0*1e3:.0f} nm)      = {neff_0:.4f}")
    print(f"  dn_eff/dlambda           = {dneff_dlam:.4f} /um")
    print(f"  n_g  (A, finite diff)    = {ng_fd:.4f}")
    if ng_mpb is not None:
        disagree = abs(ng_fd - ng_mpb) / ng_fd * 100
        print(f"  n_g  (B, MPB v_g)        = {ng_mpb:.4f}")
        print(f"  cross-check             -> {disagree:.2f}% apart "
              f"({'OK' if disagree < 1.5 else 'CHECK RESOLUTION / WINDOW'})")
    print("-" * 60)

    # FSR each ΔL would produce, using the measured n_g  (FSR = lambda^2 / (n_g * dL))
    ng = ng_fd
    lam_m = LAMBDA0 * 1e-6
    print("  FSR for a passive unbalanced MZI built on THIS waveguide:")
    for dL_um in DELTA_L_UM:
        fsr_m = lam_m**2 / (ng * dL_um * 1e-6)
        print(f"    ΔL = {dL_um:4d} um  ->  FSR = {fsr_m*1e9:6.2f} nm")
    print("=" * 60)
    print("  If n_g (A) and (B) agree, the pipeline is closed: geometry -> mode")
    print("  -> real number. That n_g is what your sax spectrum sweep will reproduce.")
    print("=" * 60)


if __name__ == "__main__":
    main()
