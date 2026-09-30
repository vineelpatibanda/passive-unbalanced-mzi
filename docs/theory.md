# Theory — Passive Unbalanced MZI

## 1. Guided mode and the two indices

A guided mode propagates as exp(−jβz) with propagation constant

    β = (2π/λ) · n_eff

- **n_eff** (phase index) sets the phase velocity, v_p = c/n_eff — and therefore *where* interference fringes sit.
- **n_g** (group index) sets the group velocity, v_g = c/n_g — and therefore *how far apart* fringes are.

They differ because n_eff depends on wavelength (dispersion):

    n_g = n_eff − λ · dn_eff/dλ

For a 500 × 220 nm Si strip at 1550 nm the gap is large (2.44 vs 4.06): the mode is tightly confined in a sub-wavelength, high-contrast core, so n_eff shifts strongly with λ.

## 2. Transfer function

Input light is split by a 50/50 coupler, travels two arms of length L and L + ΔL, and is recombined.

An ideal lossless 50/50 coupler has transfer matrix

    C = (1/√2) · [ 1   j ]
                 [ j   1 ]

The j (90° cross-coupling phase) is required for the coupler to be lossless (unitary); it is why the two output ports are complementary.

The arms add phase exp(−jβL) and exp(−jβ(L+ΔL)). The common L factors out, leaving the phase difference

    Δφ = β · ΔL = 2π · n_eff · ΔL / λ

Multiplying C · Arms · C gives

    T_bar   = cos²(Δφ/2)
    T_cross = sin²(Δφ/2)
    T_bar + T_cross = 1        (energy conservation)

(Which port is labelled "bar" depends on the coupler convention; the physics is unchanged.)

## 3. Free spectral range — why n_g, not n_eff

Transmission peaks occur when Δφ = 2πm, i.e.

    n_eff(λ_m) · ΔL = m · λ_m

For the neighbouring order m+1 at λ_m − FSR, differentiate the phase with respect to λ:

    dΔφ/dλ = 2πΔL · d/dλ [n_eff/λ]
           = −(2πΔL/λ²) · (n_eff − λ · dn_eff/dλ)
           = −(2πΔL/λ²) · n_g

Setting |dΔφ/dλ| · FSR = 2π:

    FSR = λ² / (n_g · ΔL)

The derivative pulls in dn_eff/dλ, and that combination is exactly n_g.

## 4. Consequences

- FSR ∝ 1/ΔL — doubling ΔL halves the fringe spacing.
- ΔL = 0 (balanced MZI) → Δφ independent of λ → no filtering.
- Inverting the formula, n_g = λ² / (FSR · ΔL): an unbalanced MZI on a test chip is a standard way to **measure** a waveguide's group index.

## 5. What the ideal model leaves out

- **Extinction ratio**, ER = 10·log₁₀(T_max/T_min): infinite for perfect 50/50 couplers and equal arm loss; in practice limited by splitter imbalance and unequal arm loss.
- **Insertion loss**: 0 dB in the ideal model; in practice set by propagation loss (~1–3 dB/cm for Si strips, dominated by sidewall roughness), bend loss and coupler excess loss.
