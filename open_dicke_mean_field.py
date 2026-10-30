"""Funciones independientes para el notebook corregido de Dicke abierto."""
import itertools
import numpy as np


def _rhs_mf_punto(q, omega, omega_0, lam, kappa):
    u, v, x, y, z = q
    return np.array([
        -kappa*u/2 + omega*v,
        -omega*u - kappa*v/2 - 2*lam*x,
        -omega_0*y,
        omega_0*x - 4*lam*u*z,
        4*lam*u*y,
    ])


def analizar_punto_campo_medio(q, omega, omega_0, lam, kappa):
    """Analiza fluctuaciones a longitud de espín fija; no inventa etiquetas
    de decaimiento LP/UP cuando las frecuencias son degeneradas.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (5,) or not np.all(np.isfinite(q)):
        raise ValueError("El punto debe contener cinco valores finitos.")
    u, _, x, y, z = q
    espin = q[2:]
    if np.linalg.norm(espin) < 1e-12:
        raise ValueError("La reducción tangente requiere espín medio no nulo.")

    jacobiano = np.array([
        [-kappa/2, omega, 0, 0, 0],
        [-omega, -kappa/2, -2*lam, 0, 0],
        [0, 0, 0, -omega_0, 0],
        [-4*lam*z, 0, omega_0, 0, -4*lam*u],
        [4*lam*y, 0, 0, 4*lam*u, 0],
    ])
    # SVD: dos direcciones ortonormales perpendiculares al espín.
    _, _, vh = np.linalg.svd(espin.reshape(1, 3), full_matrices=True)
    base = np.zeros((5, 4))
    base[:2, :2] = np.eye(2)
    base[2:, 2:] = vh[1:].T
    matriz = base.T @ jacobiano @ base
    valores, vectores = np.linalg.eig(matriz)
    tol = 1e-8 * max(1.0, np.max(np.abs(valores)))
    if np.any(valores.real > tol):
        estabilidad = "inestable"
    elif np.all(valores.real < -tol):
        estabilidad = "estable"
    else:
        estabilidad = "marginal"

    positivos = np.flatnonzero(valores.imag > tol)
    negativos = np.flatnonzero(valores.imag < -tol)
    pares = []
    if len(positivos) != len(negativos):
        raise RuntimeError("No se pudieron agrupar los polos conjugados.")
    if len(positivos):
        # Asignación conjunta: cada polo negativo se utiliza una sola vez.
        emparejamiento = min(
            itertools.permutations(negativos),
            key=lambda perm: sum(abs(valores[i].conjugate()-valores[j])
                                 for i, j in zip(positivos, perm)),
        )
        pares.extend(valores[[i, j]] for i, j in zip(positivos, emparejamiento))
    reales = valores[np.abs(valores.imag) <= tol]
    if len(reales) == 2:
        pares.append(reales[np.argsort(reales.real)])

    modos = []
    for par in pares:
        oscilatorio = bool(np.max(np.abs(par.imag)) > tol)
        gammas = np.sort(-par.real)
        gammas[np.abs(gammas) <= tol] = 0.0
        modos.append({
            "omega": float(np.max(np.abs(par.imag))) if oscilatorio else 0.0,
            "gamma": float(gammas.mean()) if oscilatorio else np.nan,
            "gammas": gammas,
            "oscilatorio": oscilatorio,
            "autovalores": par,
            "ambigua": False,
        })

    ramas = {}
    degenerada = False
    if len(modos) == 2:
        modos.sort(key=lambda modo: modo["omega"])
        tol_degeneracion = 1e-7 * max(1.0, modos[1]["omega"])
        degenerada = abs(modos[1]["omega"]-modos[0]["omega"]) <= tol_degeneracion
        if degenerada:
            # Frecuencia común válida; el nombre LP/UP no determina un gamma.
            frecuencia_comun = float(np.mean([m["omega"] for m in modos]))
            for nombre in ("LP", "UP"):
                ramas[nombre] = {
                    "omega": frecuencia_comun, "gamma": np.nan,
                    "gammas": np.full(2, np.nan),
                    "oscilatorio": modos[0]["oscilatorio"],
                    "ambigua": True,
                }
        else:
            ramas = dict(zip(("LP", "UP"), modos))

    return {
        "estado": q.copy(),
        "residuo": float(np.linalg.norm(_rhs_mf_punto(q, omega, omega_0, lam, kappa))),
        "jacobiano": jacobiano,
        "base": base,
        "matriz_reducida": matriz,
        "autovalores": valores,
        "autovectores": base @ vectores,
        "estabilidad": estabilidad,
        "modos": modos,
        "ramas": ramas,
        "frecuencia_degenerada": degenerada,
        "asignacion_disponible": len(modos) == 2,
    }


def puntos_y_modos_campo_medio(lam, omega=1.0, omega_0=1.0,
                              kappa=0.1, N=20, fase="auto", signo=1,
                              longitud_espin=0.5):
    """Puntos fijos con longitud de espín conservada r=longitud_espin.

    'auto' selecciona normal inferior o superradiante según el umbral.
    En degeneraciones, consulta 'modos': gamma_LP/UP queda indefinido.
    Con kappa=0 se calculan puntos fijos, no se asegura relajación.
    """
    if not np.all(np.isfinite([lam, omega, omega_0, kappa, longitud_espin])):
        raise ValueError("Los parámetros deben ser finitos.")
    if omega <= 0 or omega_0 <= 0 or lam < 0 or kappa < 0:
        raise ValueError("Frecuencias positivas; lambda y kappa >= 0.")
    if not 0 < longitud_espin <= 0.5:
        raise ValueError("Se requiere 0 < longitud_espin <= 1/2.")
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N debe ser un entero positivo.")
    if fase not in ("auto", "normal", "superradiante") or signo not in (-1, 1):
        raise ValueError("fase: auto/normal/superradiante; signo: +1/-1.")

    r = longitud_espin
    D = omega**2 + (kappa/2)**2
    critica = np.sqrt(omega_0*D/(8*r*omega))
    estados = {
        "normal": np.array([0., 0., 0., 0., -r]),
        "normal_invertido": np.array([0., 0., 0., 0., r]),
    }
    if lam > critica:
        mu = (critica/lam)**2
        for s in (-1, 1):
            x = s*r*np.sqrt(max(0., 1-mu**2))
            u = -2*lam*omega*x/D
            estados[f"superradiante_{s:+d}"] = np.array([
                u, kappa*u/(2*omega), x, 0., -r*mu,
            ])
    if fase == "superradiante" and lam <= critica:
        raise ValueError("La rama superradiante no nula exige lambda > lambda_c.")
    elegida = "normal"
    if fase == "superradiante" or (fase == "auto" and lam > critica):
        elegida = f"superradiante_{signo:+d}"
    puntos = {}
    for nombre, q in estados.items():
        info = analizar_punto_campo_medio(q, omega, omega_0, lam, kappa)
        info.update({"a_medio": np.sqrt(N)*(q[0]+1j*q[1]),
                     "espin_medio": N*q[2:]})
        puntos[nombre] = info
    return {"lambda": lam, "kappa": kappa, "lambda_c": critica,
            "longitud_espin": r, "puntos": puntos,
            "rama_elegida": elegida, "seleccionado": puntos[elegida]}


def _barrer_parametro_mf(valores, parametro, parametros):
    valores = np.asarray(valores, dtype=float)
    if valores.ndim != 1 or valores.size == 0 or not np.all(np.isfinite(valores)):
        raise ValueError("El barrido requiere un vector finito no vacío.")
    resultados = []
    for valor in valores:
        argumentos = dict(parametros)
        argumentos[parametro] = float(valor)
        resultados.append(puntos_y_modos_campo_medio(**argumentos))
    curvas = {"lambda" if parametro == "lam" else "kappa": valores,
              "lambda_c": np.array([r["lambda_c"] for r in resultados]),
              "resultados": resultados,
              "degenerada": np.array([r["seleccionado"]["frecuencia_degenerada"]
                                      for r in resultados]),
              "gamma_degeneradas": np.full((valores.size, 2), np.nan)}
    for nombre in ("LP", "UP"):
        curvas[f"omega_{nombre}"] = np.full(valores.size, np.nan)
        curvas[f"gamma_{nombre}"] = np.full(valores.size, np.nan)
        curvas[f"gammas_{nombre}"] = np.full((valores.size, 2), np.nan)
        curvas[f"no_oscilatoria_{nombre}"] = np.zeros(valores.size, dtype=bool)
    for i, resultado in enumerate(resultados):
        info = resultado["seleccionado"]
        if info["frecuencia_degenerada"]:
            curvas["gamma_degeneradas"][i] = sorted(m["gamma"] for m in info["modos"])
        for nombre, modo in info["ramas"].items():
            curvas[f"omega_{nombre}"][i] = modo["omega"]
            curvas[f"gamma_{nombre}"][i] = modo["gamma"]
            curvas[f"gammas_{nombre}"][i] = modo["gammas"]
            curvas[f"no_oscilatoria_{nombre}"][i] = not modo["oscilatorio"]
    return curvas


def barrer_lambda_campo_medio(lambdas, **parametros):
    curvas = _barrer_parametro_mf(lambdas, "lam", parametros)
    curvas["lambda_c"] = float(curvas["lambda_c"][0])
    return curvas


def barrer_kappa_campo_medio(kappas, lam, **parametros):
    parametros = dict(parametros, lam=lam)
    curvas = _barrer_parametro_mf(kappas, "kappa", parametros)
    omega = parametros.get("omega", 1.0)
    omega_0 = parametros.get("omega_0", 1.0)
    r = parametros.get("longitud_espin", 0.5)
    argumento = 8*r*lam**2*omega/omega_0 - omega**2
    curvas["kappa_c"] = 2*np.sqrt(argumento) if argumento > 0 else None
    curvas["lambda_fija"] = lam
    return curvas


def malla_con_critico(minimo, maximo, critico=None, num=501):
    if not np.all(np.isfinite([minimo, maximo])) or not 0 <= minimo < maximo:
        raise ValueError("Se requiere 0 <= minimo < maximo finitos.")
    valores = np.linspace(minimo, maximo, num)
    if critico is not None and minimo <= critico <= maximo:
        cerca = critico*np.linspace(0.98, 1.02, 161)
        cerca = cerca[(cerca >= minimo) & (cerca <= maximo)]
        valores = np.unique(np.r_[valores, cerca, critico])
    return valores


def avisos_barrido_mf(curvas):
    if np.any(curvas["degenerada"]):
        print("Frecuencias coincidentes: gamma_LP/UP queda sin asignar. "
              "Las dos tasas están en gamma_degeneradas y en los modos completos.")
    if any(np.any(~np.isfinite(curvas[f"omega_{n}"])) for n in ("LP", "UP")):
        print("Hay puntos sin clasificación local LP/UP; consulta los autovalores completos.")
