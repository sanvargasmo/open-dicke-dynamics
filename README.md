# Open Dicke dynamics

Numerical exploration of the Dicke model with cavity photon loss: finite-system Lindblad evolution, nonlinear mean-field dynamics, stationary points, and linearized collective modes. This project extends my study of the closed Dicke model to an open quantum system.

## Model and conventions

With $\hbar=1$, $N$ two-level emitters in the collective spin sector $j=N/2$ interact with one cavity mode:

$$H=\omega a^\dagger a+\omega_0J_z+\frac{\lambda}{\sqrt N}(a+a^\dagger)(J_++J_-).$$

The counter-rotating terms are retained. Cavity loss is described by

$$\dot\rho=-i[H,\rho]+L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\},\qquad L=\sqrt\kappa\,a.$$

Here $\kappa$ is the photon-number decay rate; the field amplitude decays at $\kappa/2$. There is no additional pump, atomic decay, or dephasing in this model. Frequencies, coupling, and loss are expressed in consistent frequency units, with time in the corresponding inverse units.

## Contents

- [`notebooks/dicke_disipacion_corregido.ipynb`](notebooks/dicke_disipacion_corregido.ipynb): self-contained notebook with sparse quantum evolution, adaptive photon truncation, density-matrix diagnostics, mean-field evolution, fixed points, and coupling/loss sweeps.
- [`open_dicke_mean_field.py`](open_dicke_mean_field.py): reusable mean-field functions extracted from the corrected notebook. Function names and notebook explanations retain the original Spanish notation.
- [`tests/test_mean_field.py`](tests/test_mean_field.py): checks of fixed points, the Jacobian, analytic limits, symmetry, and degenerate-mode handling.

The notebook and standalone module currently contain the same mean-field functions. Keep them synchronized when editing.

## Run

Use Python 3.11 or newer. From the repository directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name open-dicke-dynamics
python -m pytest -q
```

Open the notebook in VS Code or Jupyter and select the `open-dicke-dynamics` kernel. The first code cell asks for $N$, $\lambda$, $\kappa$, and the initial spin and photon occupations. Run cells in order. Start with small $N$ and a small photon population: a Hilbert-space dimension $d$ requires a $d^2$-dimensional Liouville space, and increasing the photon cutoff can become expensive. Review the cutoff-convergence history and boundary population before interpreting quantum results. A finite-time stationary-state estimate must also pass its residual and time-stability checks.

The standalone module requires only NumPy. For example:

```python
import numpy as np
from open_dicke_mean_field import (
    puntos_y_modos_campo_medio,
    barrer_lambda_campo_medio,
)

result = puntos_y_modos_campo_medio(lam=0.6, kappa=0.1)
point = result["seleccionado"]
print("Critical coupling:", result["lambda_c"])
print("Stability:", point["estabilidad"])
print("All four poles:", point["autovalores"])

curves = barrer_lambda_campo_medio(np.linspace(0.05, 1.0, 200), kappa=0.1)
# curves["omega_LP"], curves["omega_UP"]: angular frequencies
# curves["gamma_LP"], curves["gamma_UP"]: amplitude damping rates
```

## Mean-field analysis

Define $\langle a\rangle/\sqrt N=u+iv$, and $(x,y,z)=\langle(J_x,J_y,J_z)\rangle/N$. Factorizing spin-field products gives

$$\dot u=-\frac\kappa2u+\omega v,\qquad
\dot v=-\omega u-\frac\kappa2v-2\lambda x,$$
$$\dot x=-\omega_0y,\qquad
\dot y=\omega_0x-4\lambda uz,\qquad
\dot z=4\lambda uy.$$

These mean-field equations conserve $r=\sqrt{x^2+y^2+z^2}$. The default is $r=1/2$; smaller nonzero spin lengths are supported. For positive frequencies,

$$\lambda_c=\sqrt{\frac{\omega_0[\omega^2+(\kappa/2)^2]}{8r\omega}}.$$

The code evaluates the lower normal, inverted normal, and (when present) two symmetry-related superradiant stationary points. The `auto` option selects the lower normal point below the threshold and a superradiant branch above it.

Linearizing around a stationary point gives a five-variable Jacobian. Restricting spin perturbations to the tangent plane at fixed spin length leaves four independent variables. The reduced matrix has four poles $s$, with perturbations proportional to $e^{st}$:

$$s=-\gamma\pm i\Omega.$$

For two distinct oscillatory conjugate pairs, the lower and higher positive frequencies define the local LP and UP labels. Negative real parts imply decay; positive real parts imply instability. The notebook plots frequencies and damping versus $\lambda$ and $\kappa$, and compares nonlinear trajectories with small-perturbation evolution.

## Interpretation and limitations

- Equal frequencies do not determine a unique LP/UP damping label. The code leaves those damping labels as `NaN`, retaining the rates in `gamma_degeneradas` and the complete modes.
- Real poles describe nonoscillatory relaxation or growth. Distinct real decay rates are retained rather than averaged into an artificial single rate. When all poles are real, inspect `autovalores` directly.
- LP/UP labels are assigned by local frequency ordering, not by eigenvector tracking through crossings.
- A vanishing oscillation frequency alone does not identify a phase transition: also examine the complete pole, stability, and stationary order parameter.
- These poles describe linear response around a mean-field stationary point. They are not exact finite-$N$ Hamiltonian level spacings, a full quantum noise calculation, or automatically the peaks of an emission spectrum.
- Mean-field results do not capture finite-$N$ quantum fluctuations. At fixed collective coupling and normalized spin length, changing $N$ rescales mean observables but does not add finite-size corrections to the mean-field poles.
- At $\kappa=0$, a stationary point need not be an attracting state. A phenomenological bare-cavity Lindblad loss model also does not automatically constitute a microscopic dressed-basis bath treatment at arbitrary coupling.

## Background reference

The closed-system study that motivated this extension is C. Emary and T. Brandes, *Chaos and the quantum phase transition in the Dicke model*, Physical Review E **67**, 066203 (2003), [doi:10.1103/PhysRevE.67.066203](https://doi.org/10.1103/PhysRevE.67.066203). The dissipative calculations here are an extension of that background, rather than a reproduction of dissipative results from that closed-system paper.
