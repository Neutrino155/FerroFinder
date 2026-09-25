# Endpoint-only parent and branch reconstruction

## Input and target

The inference input is one polar endpoint: species, positions, unit cell, and
periodicity. No known nonpolar parent or intermediate path structure is used.
MACE-Field supplies Cartesian polarization $P$, reduced polarization $p$,
and Born effective charges $Z^*_{i,ab}$.

With ASE's row-vector cell matrix $H$ and volume $\Omega=|\det H|$, the
dimensionless reduced polarization is

$$
p=\Omega H^{-T}P.
$$

The candidate targets are the inversion-invariant formal classes in reduced
polarization space, reduced to symmetry-distinct orbits of the polar endpoint.
The BEC update moves the atoms (and, by default, the cell) toward each target.
Same-composition structures refined by spglib provide additional parent
hypotheses. Candidate scores combine parent distortion, cell change, target
residual, and point-group polarity.

## Generalized inverse distortion

The generalized coordinate is $q=(R,\eta)$, with Cartesian atomic
positions $R$ and six symmetric logarithmic-strain coordinates $\eta$.
The cell is updated as

$$
H'=H\exp(\eta),\qquad
J=\begin{bmatrix}J_R&J_\eta\end{bmatrix},\qquad
J_R=\frac{\partial p}{\partial R},\quad
J_\eta=\frac{\partial p}{\partial\eta}.
$$

At fixed cell the BEC block is

$$
\frac{\partial p_a}{\partial R_{ib}}
=\left(H^{-T}\right)_{ac}Z^*_{i,cb}.
$$

When variable-cell backprojection is enabled, its lattice block is computed
as six central differences of reduced polarization at fixed fractional
coordinates. This is a branch-aware clamped-ion strain response, analogous to
the piezoelectric contribution but kept in reduced-polarization coordinates
for direct use in the inverse update. Each displaced polarization is matched
to the reference Berry branch before differencing:

$$
(J_\eta)_{ak}\approx
\frac{p_a(H\exp(\delta E_k))-p_a(H\exp(-\delta E_k))}{2\delta}.
$$

For polarization residual $\Delta p=p_\mathrm{target}-p$, the minimum-metric
step is

$$
\Delta q=W^{-1}J^T(JW^{-1}J^T)^+\Delta p,
$$

where atomic coordinates use unit metric weight and logarithmic strains use
weight 25 by default. The pseudoinverse handles rank-deficient response.
Each atomic step is capped at 0.12 Å and each strain component at 0.02. The
default cumulative cell bounds are 0.85–1.15 for principal stretches and
0.70–1.30 for volume ratio.

The screening CLI enables this block by default. This is a bounded
inverse-polarization update; it does not minimize stress or energy with
respect to the cell.

## Branch construction

The best parent candidates are aligned to the polar endpoint by a
species-aware periodic assignment. Nine images interpolate positions and
log-strain from the endpoint to the candidate. The atomic displacement is
then reflected about the parent to construct the opposite polar branch.
MACE-Field evaluates energy and polarization on the sampled images. Reduced
polarization is continuously unwrapped by integer polarization quanta and
folded for display.

## Evidence and limits

The MP-Ferroelectric screen is evaluated against held-out parents and sampled
paths only after inference completes. Its strict parent criterion requires
both atom-matched RMSD and cell RMSD to be at most 0.25 Å. Wider cutoffs are
reported as approximate structural matches. The [screen report](../screenings/mp_ferroelectric_250/reports/method_and_results.md)
contains fixed-cell and variable-cell comparisons.

Recovered candidates are not automatically stable phases. The sampled paths
are geometric reconstructions, not constrained minima, relaxed switching
paths, or activation barriers. MACE-Field predictions do not prove that a
candidate is insulating.
