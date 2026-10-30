"""Exercise the notebook's actual solver without interactive or costly cells."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def test_notebook_syntax_and_uncoupled_loss():
    path = Path(__file__).resolve().parents[1]/"notebooks/dicke_disipacion_corregido.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index+1}", "exec")
    namespace = dict(np=np, sparse=sparse, expm_multiply=expm_multiply)
    exec("".join(notebook["cells"][5]["source"]), namespace)
    result = namespace["resolver_dicke"](1, 3, 3, -0.5, 1., 1., 0., 0.2, 10., 101)
    np.testing.assert_allclose(result["n_medio"], 3*np.exp(-0.2*result["t"]), atol=1e-10)
    np.testing.assert_allclose(result["Jz_medio"], -0.5, atol=1e-12)
    rho = result["rho"]
    np.testing.assert_allclose(np.trace(rho, axis1=1, axis2=2), 1, atol=1e-12)
    np.testing.assert_allclose(rho, rho.conj().transpose(0, 2, 1), atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_standalone_functions_match_notebook():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/"notebooks/dicke_disipacion_corregido.ipynb").read_text(encoding="utf-8"))
    assert (root/"open_dicke_mean_field.py").read_text(encoding="utf-8") == "".join(notebook["cells"][23]["source"])
