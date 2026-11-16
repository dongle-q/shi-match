import re

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from jax.scipy.linalg import expm
from scipy.optimize import minimize


def parse_target_lumos(tokens, n_alpha_elec, n_beta_elec, nbas):
    """Read targets counted from the LUMO of each spin: a0 (= a, alumo, alumo0), a1, b0, b1, ...
    Tokens may be separated by spaces or commas.
    Return sorted 0-based MO indices of the alpha and beta targets."""
    targets = {"a": [], "b": []}
    for token in " ".join(tokens).replace(",", " ").split():
        match = re.fullmatch(r"([ab])(?:lumo)?(\d*)", token.lower())
        if match is None:
            raise ValueError(f"Cannot read target '{token}'. Examples: a0, b1, alumo, blumo1.")
        spin, number = match.groups()
        n_occ = n_alpha_elec if spin == "a" else n_beta_elec
        idx = n_occ + int(number or 0)
        if idx >= nbas:
            raise ValueError(f"Target '{token}' is beyond NBAS = {nbas}.")
        if idx not in targets[spin]:
            targets[spin].append(idx)
    return sorted(targets["a"]), sorted(targets["b"])


def best_partners(occ_coef, targ_coef, ao_ovlp):
    """Closed form: combinations of occupied MOs (columns of occ_coef) with the largest overlap
    with the target MOs (columns of targ_coef).

    SVD of S_0 = C_occ.T S_ao C_targ = U sigma W.T:
        partner j = C_occ U[:, j],  target j = C_targ W[:, j],  overlap = sigma_j (the maximum)
    """
    U, sigma, Wt = np.linalg.svd(occ_coef.T @ ao_ovlp @ targ_coef, full_matrices=False)
    return U, sigma, Wt.T


def complete_rotation(partners):
    """Orthogonal matrix with the partner vectors (columns of partners, in the occupied MO basis)
    in the slots of the canonical MOs contributing most to them. The other columns are the
    canonical MOs with the partners projected out, Loewdin-orthonormalized (closest to canonical).
    Return the matrix and the partner slots."""
    n_occ, n_partner = partners.shape
    slots = []
    for j in range(n_partner):
        slots.append(next(int(i) for i in np.argsort(-abs(partners[:, j])) if i not in slots))
    others = [i for i in range(n_occ) if i not in slots]

    rest = (np.eye(n_occ) - partners @ partners.T)[:, others]
    eig_val, eig_vec = np.linalg.eigh(rest.T @ rest)
    rotation = np.empty((n_occ, n_occ))
    rotation[:, slots] = partners
    rotation[:, others] = rest @ (eig_vec / np.sqrt(eig_val)) @ eig_vec.T
    return rotation, slots


def minimize_rotation(ovlp, log):
    """Rotation R of the rows of the square overlap S_0 minimizing ||(R.T S_0)^2 - I||^2.
    S_0 is computed once by the caller. Return R and the final objective."""
    n = ovlp.shape[0]
    triu = jnp.triu_indices(n, k=1)
    ovlp = jnp.asarray(ovlp)

    def rotation_from_vec(vec):
        # Rotation matrix R = exp(W), W antisymmetric
        antisym = jnp.zeros((n, n)).at[triu].set(vec)
        return expm(antisym - antisym.T)

    def objective(vec):
        # Squared Frobenius norm of S'^2 - I, S' = R.T S_0
        return jnp.sum(jnp.square((rotation_from_vec(vec).T @ ovlp) ** 2 - jnp.eye(n)))

    # Initial guess: SVD (Procrustes) rotation, then logm to get W.
    # exp(W) is always a proper rotation (det = +1). If U @ Vt has det = -1, flip the sign of
    # one column of U (the objective only depends on S'^2) so that logm stays real.
    U, _, Vt = np.linalg.svd(np.asarray(ovlp))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    init_W = np.real(la.logm(U @ Vt))
    init_guess = init_W[np.triu_indices(n, k=1)]

    grad_fn = jax.grad(objective)
    step = 0

    def callback_func(xk):
        nonlocal step
        log.write(f"{step:<4d}   {objective(xk):>13.8f} \n")
        step += 1

    log.write("====================\n")
    log.write("    CONVERGENCE\n")
    log.write("====================\n")
    log.write("step      obj. func.\n")
    log.write("--------------------\n")

    result = minimize(
        lambda v: np.asarray(objective(v)),
        init_guess,
        jac=lambda v: np.asarray(grad_fn(v)),
        method="CG",
        callback=callback_func,
    )
    final_obj = float(objective(result.x))
    log.write("\n")
    log.write(f"The minimization is done after {result.nit} steps.\n")
    log.write(f"Squared Frobenius norm ||S^2-1||2F  =  {final_obj:.5f} \n")
    log.write("--------------------\n")
    return np.asarray(rotation_from_vec(result.x)), final_obj
