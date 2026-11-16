#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import time
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from jax.scipy.linalg import expm
from scipy.optimize import minimize

from file47_parser import get_dat_from_file47
from gaussian_io import write_fchk
from nwchem_io import write_ascii_movecs
from orbital_matching import (
    best_partners,
    complete_rotation,
    minimize_rotation,
    parse_target_lumos,
)
from printing_utils import print_overlap

start_time = time.time()


VERSION = "0.9.0"

parser = argparse.ArgumentParser(
    prog="SHI-match", description="A tool for SOMO-HOMO Inversion."
)

# Add required arguments
parser.add_argument(
    "file47",
    default=None,
    help="gennbo file (file47) from NWChem (keyword: nbofile) or Gaussian (keyword: gennbo).",
)

# TO-DOs or improve. These options help user to generate cube files
parser.add_argument(
    "-nwchem",
    "--movecs",
    default=None,
    help="movecs file from NWChem. (Note: Please add nwchem to PATH and compile mov2asc NWCHEM_TOP/contrib/mov2asc).",
)
parser.add_argument(
    "-gaussian",
    "-g16",
    "--fch",
    "-fch",
    "-fchk",
    dest="fch",
    default=None,
    help="formated checkpoint file (fch, fchk) from Gaussian.",
)

parser.add_argument(
    "--target",
    nargs="+",
    default=None,
    help="Virtual MOs to match, counted from the LUMO of each spin: a0 (= a, alumo, alumo0) is the alpha LUMO, "
    "a1 the next one, b0 the beta LUMO, ...",
)

parser.add_argument(
    "--match-method",
    choices=["minimize", "closed"],
    default="minimize",
    help="How the occ. MOs are rotated for --target-lumos. minimize: minimize ||S'^2 - 1|| so that all occ. MOs "
    "are matched one-to-one (falls back to closed if the occ. sets cannot be matched one-to-one). closed: build "
    "the best partner of each target directly and keep the other occ. MOs as close to canonical as possible "
    "(default: minimize).",
)

parser.add_argument(
    "--homo-threshold",
    type=float,
    default=0.95,
    help="Warn if |<canonical HOMO|transformed HOMO>| is below this value (default: 0.95).",
)

parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="enable verbose output",
)

parser.add_argument(
    "--version",
    action="version",
    version=f"SHI-match version {VERSION}",
)

parser.add_argument(
    "-hw",
    "--hardware",
    default=None,
    help="Hardware platform to use (cpu, cuda, rocm, tpu). Use cuda for NVIDIA GPU and rocm for AMD GPU.",
)

args = parser.parse_args()


# JAX setup

if args.hardware is None:
    jax.config.update("jax_platforms", "cpu")
else:
    hardware = args.hardware.lower()
    if hardware not in ["cpu", "cuda", "rocm", "tpu"]:
        parser.error(
            "Invalid hardware option. Please choose from 'cpu', 'cuda', 'rocm', or 'tpu'."
        )
    else:
        jax.config.update("jax_platforms", hardware)

# Create a log file and print the header
log = open("shi_match.log", "w")
log.write(f"""
                    777000SPIN-DOWN
                   7770000000000000
                  777
                 777  000    000  IDE      MOLECULAR0000        AU     TRANSFORM  CONJUGATE00  000    000
       ^        777   000    000  NTI      ORBITAL000000       TODI    000000000  GRADIENT000  000    000
     <HHH>     777    HOMO000000  TY0      0000000000000      FFEREN      000     000          000000SUMO
       H      777     SOMO000000  MAT      0000000000000     TIA  TIO     000     000          000000HOMO
       H     777      000    000  100      000 00000 000    N000000000    000     000          000    000
SPIN-UP00000777       000    000  010      000  000  000   0000    0000   000     00000000000  000    000
00000000000777        000    000  001      000   0   000  0000      0000  000     00000000000  000    000
       H
       H     SHI-MATCH VERSION: {VERSION}
""")


# Process input files---------------------------------------------------------
current_directory = os.getcwd()
file47 = args.file47
if args.file47 is None:
    file47s = glob.glob(current_directory + "/*.gen")
    if len(file47s) > 1:
        print("Please specify which file47 file will be used:")
        for gen in file47s:
            print(f"""     {gen.split("/")[-1]}""")
        sys.exit()
    elif len(file47s) == 0:
        print("There is no file47 in the current directory.")
        sys.exit()
    else:
        file47 = file47s[0]
log.write(f"\nGENNBO file  : {file47.split('/')[-1]}\n")
log.write(f"Work directory : {current_directory} \n")
# -----------------------------------------------------------------------------

HARTREE_TO_EV = 27.2113245702

def match_orbital(ovlp_row, threshold=0.97):
    max_idx = np.argmax(abs(ovlp_row))
    max_ovlp = ovlp_row[max_idx]
    if abs(max_ovlp) >= threshold:
        return max_idx, max_ovlp
    return None, None

# Get C, S_ao, Fock matrices, nbas, number of alpha-beta electrons from file47
# MO coefficient matrices are stored row-wise: one row per MO.
(
    a_coef_full,
    b_coef_full,
    ao_ovlp,
    a_fock,
    b_fock,
    nbas,
    n_alpha_elec,
    n_beta_elec,
) = get_dat_from_file47(file47, log)

n_unpaired = n_alpha_elec - n_beta_elec
a_homo_idx = n_alpha_elec - 1
b_homo_idx = n_beta_elec - 1
b_sumo_idx = n_beta_elec

log.write("\n")
log.write(f"""
                 Alpha electrons: {n_alpha_elec}
                  Beta electrons: {n_beta_elec}
                Alpha HOMO index: {a_homo_idx + 1}
                 Beta HOMO index: {b_homo_idx + 1}
          Lowest Beta SUMO index: {b_sumo_idx + 1}
                            2S+1: {n_unpaired + 1}""")

# Overlap between all alpha and all beta MOs
init_ovlp_full = reduce(np.dot, (a_coef_full, ao_ovlp, b_coef_full.T))

log.write("\nINITIAL OVERLAP\n")
log.write("=====================\n")
print_overlap(init_ovlp_full, 10, False, log, n_alpha_elec, a_homo_idx, b_homo_idx)

sumo_a_idx = np.argmax(abs(init_ovlp_full[:, b_sumo_idx]))
sumo_max_ovlp = init_ovlp_full[sumo_a_idx, b_sumo_idx]
log.write(
    f"\nMaximal overlap with lowest beta SUMO:    < b_{b_sumo_idx + 1:>} | a_{sumo_a_idx + 1:<} > = {sumo_max_ovlp:<5.4f}\n"
)

log.write("\n")


# Rotation space: all alpha occupied MOs, and beta occupied MOs + beta SUMOs
n_rot = n_alpha_elec
a_occ_coef = a_coef_full.T[:, :n_rot]
b_occ_sumo_coef = b_coef_full.T[:, :n_rot]
init_ovlp = jnp.dot(a_occ_coef.T, jnp.dot(ao_ovlp, b_occ_sumo_coef))


def check_permuted_identity(ovlp, threshold=0.05):
    abs_ovlp = abs(ovlp)
    n_orbital = abs_ovlp.shape[0]
    strong_ovlp_idx = []
    for orb_idx in range(n_orbital):
        max_idx = np.argmax(abs_ovlp[orb_idx])
        if abs_ovlp[orb_idx, max_idx] < (1.0 - threshold):
            return False  # No element close to 1
        strong_ovlp_idx.append(max_idx)

    # Check if all strong overlap are unique [set in Python will remove duplicates]
    if len(set(strong_ovlp_idx)) != n_orbital:
        return False

    # Sort rows by their strong overlap index --> Create a identity permuted matrix
    permuted_ovlp = abs_ovlp[np.argsort(strong_ovlp_idx)]

    # Verify
    for i in range(n_orbital):
        if permuted_ovlp[i, i] < (1.0 - threshold):
            return False
        for j in range(n_orbital):
            if i != j and permuted_ovlp[i, j] > threshold:
                return False
    return True


def antisymm_mat_from_vec(vec):
    antisym = jnp.zeros((n_rot, n_rot))
    antisym = antisym.at[jnp.triu_indices(n_rot, k=1)].set(vec)
    return antisym - antisym.T


def objective(vec, ovlp):
    # Rotation matrix R = exp(W), W antisymmetric
    rotation = expm(antisymm_mat_from_vec(vec))
    # Compute S' = C'_alpha.T S_ao C_beta
    #            = (C_alpha * R).T S_ao C_beta
    #            = R.T C_alpha.T S_ao C_beta
    rotated_ovlp = jnp.dot(rotation.T, ovlp)
    # Squared Frobenius norm of S'^2 - I
    return jnp.sum(jnp.square(rotated_ovlp**2 - jnp.eye(n_rot)))


# Compute initial guess for optimization
# Use SVD to compute initial rotation matrix, then logm to get initial W matrix
U, _, Vt = np.linalg.svd(init_ovlp)
init_W = np.real(la.logm(U @ Vt))
init_guess = init_W[np.triu_indices_from(init_W, k=1)]


def shi_match():

    grad_fn = jax.grad(objective)

    step = 0

    def callback_func(xk):
        nonlocal step
        log.write(f"{step:<4d}   {objective(xk, init_ovlp):>13.8f} \n")
        step += 1

    log.write("====================\n")
    log.write("    CONVERGENCE\n")
    log.write("====================\n")
    log.write("step      obj. func.\n")
    log.write("--------------------\n")

    result = minimize(
        lambda v, ovlp: np.asarray(objective(v, ovlp)),
        init_guess,
        args=(init_ovlp,),
        jac=lambda v, ovlp: np.asarray(grad_fn(v, ovlp)),
        method="CG",
        callback=callback_func,
    )

    log.write("\n")
    log.write(f"The minimization is done after {result.nit} steps.\n")
    final_obj = objective(result.x, init_ovlp)
    log.write(f"Squared Frobenius norm ||S^2-1||2F  =  {final_obj:.5f} \n")

    # Extract the final result
    rotation = expm(antisymm_mat_from_vec(result.x))
    log.write("--------------------\n")

    a_trans_coef = np.dot(a_occ_coef, rotation)
    a_trans_ener = HARTREE_TO_EV * (a_trans_coef.T @ a_fock @ a_trans_coef).diagonal()
    a_canon_ener = HARTREE_TO_EV * (a_coef_full @ a_fock @ a_coef_full.T).diagonal()
    b_ener = HARTREE_TO_EV * (b_coef_full @ b_fock @ b_coef_full.T).diagonal()

    # Sort transformed alpha occupied MOs by energy
    ener_order = np.argsort(a_trans_ener)
    a_trans_ener = a_trans_ener[ener_order]
    a_trans_coef = a_trans_coef[:, ener_order]

    final_ovlp = a_trans_coef.T @ ao_ovlp @ b_occ_sumo_coef
    log.write("\nFINAL OVERLAP\n")
    log.write("=====================\n")
    print_overlap(final_ovlp, 10, True, log, n_alpha_elec, a_homo_idx, b_homo_idx)

    log.write("\n")
    if final_obj > 0.5:
        log.write("WARNING: There might some orbital swappings.\n")

    log.write("\n")

    if not check_permuted_identity(final_ovlp):
        log.write("WARNING: Overlap matrix is not permuted identity.\n")
        log.write("JOB FAILED\n")
        return False

    log.write("================================\n")
    log.write("    Alpha MO energies (eV)\n")
    log.write("================================\n")
    log.write("i       canonical    transformed  \n")
    log.write("--------------------------------\n")
    for a_idx in range(n_alpha_elec):
        log.write(
            f"""{a_idx + 1:<4d} {a_canon_ener[a_idx]:>12.3f}   {a_trans_ener[a_idx]:>12.3f}   \n"""
        )

    log.write("\n\n")
    log.write("==========================================\n")
    log.write("            MO energies eV\n")
    log.write("==========================================\n")
    log.write("i  transformed ALPHA  canonical BETA  S_ii\n")
    log.write("------------------------------------------\n")
    for a_idx in range(n_alpha_elec):
        ovlp_str = (
            f"{final_ovlp[a_idx, a_idx]:>5.2f}"
            if abs(final_ovlp[a_idx, a_idx]) > 0.01
            else "  0  "
        )
        log.write(
            f"""{a_idx + 1:<4d} {a_trans_ener[a_idx]:>13.3f}   {b_ener[a_idx]:>13.3f}   {ovlp_str}\n"""
        )

    a_somo_indices = []
    b_sumo_indices = []

    log.write("\n\n")
    log.write("===============================\n")
    log.write("Matching transf. alpha and beta\n")
    log.write("===============================\n")
    log.write("alpha(i)  beta(j)  S_ij    swap\n")
    log.write("-------------------------------\n")
    for a_idx in range(n_alpha_elec):
        b_idx, ovlp = match_orbital(final_ovlp[a_idx, :])
        note = "   " if a_idx == b_idx else "yes"
        log.write(
            f"""   {a_idx + 1:>5d}  {b_idx + 1:<5d}    {ovlp:>5.2f}    {note} \n"""
        )
        if b_idx >= n_beta_elec:
            a_somo_indices.append(a_idx)
            b_sumo_indices.append(b_idx)

    log.write("\n\n")
    log.write("=============================================\n")
    log.write("        Alpha SOMO(i) and beta SUMO(j)\n")
    log.write("=============================================\n")
    log.write("   i - j     S_ij        a_SOMO        b_SUMO\n")
    log.write("---------------------------------------------\n")
    for a_idx, b_idx in zip(a_somo_indices, b_sumo_indices):
        log.write(
            f"""{a_idx + 1:>5d} {b_idx + 1:<5d} {final_ovlp[a_idx, b_idx]:>5.2f} {a_trans_ener[a_idx]:>13.3f} {b_ener[b_idx]:>13.3f} \n"""
        )
        # Alpha SOMO matched with the lowest beta SUMO
        if b_idx == b_sumo_idx:
            a_somo_idx = a_idx
            a_somo_ener = a_trans_ener[a_idx]

    b_homo_ener = b_ener[b_homo_idx]
    a_homo_ener = np.max(a_trans_ener)

    shi_gap = a_homo_ener - a_somo_ener
    log.write("\n\n")
    log.write("=====================================\n")
    log.write("              SHI GAP\n")
    log.write("=====================================\n")
    log.write("  alpha                 beta\n\n")
    log.write(
        f"  HOMO: {a_homo_ener:<10.4f}      SUMO: {b_ener[b_sumo_idx]:<10.4f}\n"
    )
    if a_homo_idx != a_somo_idx:
        log.write(f"                        HOMO: {b_homo_ener:<10.4f}\n\n")
        log.write(f"  SOMO: {a_somo_ener:<10.4f}\n")
    else:
        log.write(
            f"  SOMO: {a_somo_ener:<10.4f}      HOMO: {b_homo_ener:<10.4f}\n"
        )
    log.write(f"  ---------------\n  SHI gap: {shi_gap:.4f}\n")
    a_full_ener = np.concatenate((a_trans_ener, a_canon_ener[n_alpha_elec:]))
    a_full_coef = np.vstack((a_trans_coef.T, a_coef_full[n_alpha_elec:, :]))
    log.write("\n")

    if shi_gap == 0.0 and b_homo_ener < a_homo_ener:
        shi_class = "non-SHI"
        log.write("Classification: non-SHI\n")
    elif shi_gap == 0.0 and b_homo_ener > a_homo_ener:
        shi_class = "partial"
        log.write("Classification: partial-SHI")
    elif shi_gap > 0.0 and b_homo_ener > a_homo_ener:
        shi_class = "SHI"
        log.write("Classification: SHI")

    print(shi_gap, shi_class)

    if args.movecs:
        write_ascii_movecs(args.movecs, nbas, a_full_ener, a_full_coef)

    if args.fch:
        write_fchk(args.fch, {"alpha": (a_full_coef, a_full_ener / HARTREE_TO_EV)})  # fchk stores Hartree

    return True


VIRT_PAIR_THRESHOLD = 0.95  # alpha and beta targets with this overlap are paired directly
PAIR_WARN_THRESHOLD = 0.95  # warn if a matched pair has a smaller overlap


def mo_energies(coef, fock):
    # Orbital energies (eV) of the MOs in the columns of coef
    return HARTREE_TO_EV * np.sum(coef * (fock @ coef), axis=0)


def assign_positions(mix, indices):
    # Each (rotated) target MO takes the index of the canonical target contributing most to it
    positions = []
    for j in range(mix.shape[1]):
        order = np.argsort(-abs(mix[:, j]))
        positions.append(next(indices[i] for i in order if indices[i] not in positions))
    return positions


def lumo_label(spin, idx):
    n_occ = n_alpha_elec if spin == "a" else n_beta_elec
    name = "alpha" if spin == "a" else "beta"
    return f"{spin}{idx + 1} ({name} LUMO{f'+{idx - n_occ}' if idx > n_occ else ''})"


def classify_shi(somo_is_homo, shi_gap, own_homo_ener, other_homo_ener):
    # own: spin of the SOMO; other: spin of the SUMO
    if somo_is_homo and other_homo_ener < own_homo_ener:
        return "non-SHI"
    if somo_is_homo and other_homo_ener > own_homo_ener:
        return "partial"
    if shi_gap > 0.0 and other_homo_ener > own_homo_ener:
        return "SHI"
    return "undetermined"


def match_targets(a_targ_idx, b_targ_idx):
    """Match target virtual MOs with occupied MOs of the opposite spin.

    Step 0: alpha and beta targets that overlap each other are paired directly (no occupied
            partner, no SHI).
    Step 1: rotate alpha occ. to match the remaining beta targets. These pairs are frozen.
    Step 2: rotate beta occ. to match the remaining alpha targets, excluding the frozen pairs.
    """
    a_occ = a_coef_full.T[:, :n_alpha_elec]
    b_occ = b_coef_full.T[:, :n_beta_elec]
    a_targ = a_coef_full.T[:, a_targ_idx]
    b_targ = b_coef_full.T[:, b_targ_idx]
    # Mixing of the canonical targets: column j = current target j in the canonical target basis
    a_mix = np.eye(len(a_targ_idx))
    b_mix = np.eye(len(b_targ_idx))

    log.write("====================================\n")
    log.write("          TARGET MATCHING\n")
    log.write("====================================\n")
    log.write(
        "Targets: "
        + ", ".join([lumo_label("a", i) for i in a_targ_idx] + [lumo_label("b", i) for i in b_targ_idx])
        + "\n"
    )
    log.write(f"Method : {args.match_method}\n\n")

    # Step 0: alpha target <-> beta target -------------------------------------------------
    n_vv = 0
    if a_targ_idx and b_targ_idx:
        U, sigma, Wt = np.linalg.svd(a_targ.T @ ao_ovlp @ b_targ)
        if len(a_targ_idx) >= 2:
            a_targ, a_mix = a_targ @ U, a_mix @ U
        if len(b_targ_idx) >= 2:
            b_targ, b_mix = b_targ @ Wt.T, b_mix @ Wt.T
        n_vv = int(np.sum(sigma >= VIRT_PAIR_THRESHOLD))
        log.write(
            "Step 0: overlap between alpha and beta targets (singular values): "
            + ", ".join(f"{value:.4f}" for value in sigma)
            + f"\n        {n_vv} virt.-virt. pair(s) (>= {VIRT_PAIR_THRESHOLD})\n\n"
        )
    n_a_rem = len(a_targ_idx) - n_vv
    n_b_rem = len(b_targ_idx) - n_vv

    # Closed form: best occ. partners of the remaining targets -------------------------------
    R_a, a_slots = np.eye(n_alpha_elec), []
    R_b, b_slots = np.eye(n_beta_elec), []
    if n_b_rem > 0:
        U_a, sigma_b, W_b = best_partners(a_occ, b_targ[:, n_vv:], ao_ovlp)
        if n_b_rem >= 2:
            b_targ[:, n_vv:] = b_targ[:, n_vv:] @ W_b
            b_mix[:, n_vv:] = b_mix[:, n_vv:] @ W_b
        R_a, a_slots = complete_rotation(U_a)
        log.write(
            "Best possible overlap of the beta targets with alpha occ.: "
            + ", ".join(f"{value:.4f}" for value in sigma_b)
            + "\n"
        )
    if n_a_rem > 0:
        U_b, sigma_a, W_a = best_partners(b_occ, a_targ[:, n_vv:], ao_ovlp)
        if n_a_rem >= 2:
            a_targ[:, n_vv:] = a_targ[:, n_vv:] @ W_a
            a_mix[:, n_vv:] = a_mix[:, n_vv:] @ W_a
        R_b, b_slots = complete_rotation(U_b)
        log.write(
            "Best possible overlap of the alpha targets with beta occ.: "
            + ", ".join(f"{value:.4f}" for value in sigma_a)
            + "\n"
        )
    log.write("\n")

    # Minimization: match all occ. MOs one-to-one, partners in the closed-form slots ---------
    if args.match_method == "minimize":
        n_a_left = n_alpha_elec - len(a_slots)
        n_b_left = n_beta_elec - len(b_slots)
        if n_a_left != n_b_left:
            log.write(
                f"WARNING: {n_a_left} alpha occ. and {n_b_left} beta occ. MOs are left after taking out the "
                "partners of the targets, so they cannot be matched one-to-one.\n"
                "         Add targets of the other spin (e.g. the beta SUMOs). Using the closed-form result.\n\n"
            )
        else:
            if n_b_rem > 0:
                # Step 1: rotate alpha occ. Reference: beta targets in the partner slots, beta occ.
                # (without the partners of the alpha targets) in the other slots
                log.write("Step 1: rotate alpha occ. to match the beta targets\n")
                a_others = [i for i in range(n_alpha_elec) if i not in a_slots]
                b_ref = np.empty((a_occ.shape[0], n_alpha_elec))
                b_ref[:, a_slots] = b_targ[:, n_vv:]
                b_ref[:, a_others] = np.delete(b_occ @ R_b, b_slots, axis=1)
                R_a, _ = minimize_rotation(a_occ.T @ ao_ovlp @ b_ref, log)
                log.write("\n")
            if n_a_rem > 0:
                # Step 2: rotate beta occ. Reference: alpha targets in the partner slots, alpha occ.
                # without the frozen partners of the beta targets in the other slots
                log.write("Step 2: rotate beta occ. to match the alpha targets (step 1 pairs frozen)\n")
                b_others = [i for i in range(n_beta_elec) if i not in b_slots]
                a_ref = np.empty((b_occ.shape[0], n_beta_elec))
                a_ref[:, b_slots] = a_targ[:, n_vv:]
                a_ref[:, b_others] = np.delete(a_occ @ R_a, a_slots, axis=1)
                R_b, _ = minimize_rotation(b_occ.T @ ao_ovlp @ a_ref, log)
                log.write("\n")

    a_occ_trans = a_occ @ R_a
    b_occ_trans = b_occ @ R_b
    a_occ_ener = mo_energies(a_occ_trans, a_fock)
    b_occ_ener = mo_energies(b_occ_trans, b_fock)

    # Check of the remaining occ.-occ. pairs
    rest_sigma = np.linalg.svd(
        np.delete(a_occ_trans, a_slots, axis=1).T @ ao_ovlp @ np.delete(b_occ_trans, b_slots, axis=1),
        compute_uv=False,
    )
    n_small = int(np.sum(rest_sigma < PAIR_WARN_THRESHOLD))
    log.write(
        f"Remaining alpha occ. / beta occ. MOs: {n_small} singular value(s) below {PAIR_WARN_THRESHOLD} "
        f"(smallest {rest_sigma.min():.4f})\n"
    )
    if n_small > 0:
        log.write("WARNING: some occ. MOs have no partner of the other spin. Consider more targets.\n")

    # HOMO check: transformed HOMO should still be the canonical HOMO
    a_homo_slot = int(np.argmax(a_occ_ener))
    b_homo_slot = int(np.argmax(b_occ_ener))
    for spin, rotation, homo_slot, canon_homo in (
        ("alpha", R_a, a_homo_slot, a_homo_idx),
        ("beta", R_b, b_homo_slot, b_homo_idx),
    ):
        coef = abs(rotation[canon_homo, homo_slot])
        log.write(f"|<canonical {spin} HOMO|transformed {spin} HOMO>| = {coef:.4f}\n")
        if coef < args.homo_threshold:
            log.write(
                f"WARNING: The transformed {spin} HOMO is not the canonical {spin} HOMO "
                f"(coefficient {coef:.4f} < {args.homo_threshold}).\n"
            )
    log.write("\n")

    # Full transformed MO sets (rows = MOs): occ. sorted by energy, targets at their positions
    a_order = np.argsort(a_occ_ener)
    b_order = np.argsort(b_occ_ener)
    a_pos = assign_positions(a_mix, a_targ_idx) if a_targ_idx else []
    b_pos = assign_positions(b_mix, b_targ_idx) if b_targ_idx else []
    a_full_coef = a_coef_full.copy()
    b_full_coef = b_coef_full.copy()
    a_full_coef[:n_alpha_elec] = a_occ_trans[:, a_order].T
    b_full_coef[:n_beta_elec] = b_occ_trans[:, b_order].T
    a_full_coef[a_pos] = a_targ.T
    b_full_coef[b_pos] = b_targ.T
    a_full_ener = mo_energies(a_full_coef.T, a_fock)
    b_full_ener = mo_energies(b_full_coef.T, b_fock)

    # Prime: transformed MO (alpha occ. is rotated if there are beta targets, and vice versa)
    def a_label(idx):
        rotated = n_b_rem > 0 if idx < n_alpha_elec else len(a_targ_idx) >= 2
        return f"a_{idx + 1}" + ("'" if rotated else "")

    def b_label(idx):
        rotated = n_a_rem > 0 if idx < n_beta_elec else len(b_targ_idx) >= 2
        return f"b_{idx + 1}" + ("'" if rotated else "")

    # Pairs: (type, alpha MO index, beta MO index) in the full transformed sets
    pairs = []
    for j in range(n_vv):
        pairs.append(("virt-virt", a_pos[j], b_pos[j]))
    for j in range(n_b_rem):
        pairs.append(("a-SOMO/b-SUMO", int(np.where(a_order == a_slots[j])[0][0]), b_pos[n_vv + j]))
    for j in range(n_a_rem):
        pairs.append(("b-SOMO/a-SUMO", a_pos[n_vv + j], int(np.where(b_order == b_slots[j])[0][0])))

    n_show = max([n_alpha_elec] + [i + 1 for i in a_targ_idx + b_targ_idx])
    final_ovlp = a_full_coef[:n_show] @ ao_ovlp @ b_full_coef[:n_show].T
    log.write("\nFINAL OVERLAP\n")
    log.write("=====================\n")
    print_overlap(final_ovlp, 10, True, log, n_show, a_homo_idx, b_homo_idx)

    log.write("\n\n")
    log.write("==================================================================\n")
    log.write("                         MATCHED PAIRS\n")
    log.write("==================================================================\n")
    log.write("type             alpha      beta        S_ij   E_alpha(eV)  E_beta(eV)\n")
    log.write("------------------------------------------------------------------\n")
    for kind, a_idx, b_idx in pairs:
        ovlp = a_full_coef[a_idx] @ ao_ovlp @ b_full_coef[b_idx]
        log.write(
            f"{kind:<15s}  {a_label(a_idx):<9s}  {b_label(b_idx):<9s} {ovlp:>7.4f}  "
            f"{a_full_ener[a_idx]:>11.3f}  {b_full_ener[b_idx]:>10.3f}\n"
        )
        if abs(ovlp) < PAIR_WARN_THRESHOLD:
            log.write(f"WARNING: {a_label(a_idx)} and {b_label(b_idx)} overlap only {abs(ovlp):.4f}.\n")

    # SHI gaps of the occ.-virt. pairs
    a_homo_ener = a_occ_ener[a_homo_slot]
    b_homo_ener = b_occ_ener[b_homo_slot]
    log.write("\n\n")
    log.write("=====================================================================\n")
    log.write("                              SHI GAP\n")
    log.write("=====================================================================\n")
    log.write(f"Transformed HOMO (eV): alpha {a_homo_ener:.4f}, beta {b_homo_ener:.4f}\n\n")
    log.write("SOMO        SUMO         E_HOMO     E_SOMO     E_SUMO   SHI gap  class\n")
    log.write("---------------------------------------------------------------------\n")
    for kind, a_idx, b_idx in pairs:
        if kind == "virt-virt":
            continue
        if kind == "a-SOMO/b-SUMO":
            somo_label, sumo_label = a_label(a_idx), b_label(b_idx)
            somo_ener, sumo_ener, homo_ener = a_full_ener[a_idx], b_full_ener[b_idx], a_homo_ener
            somo_is_homo = a_idx == n_alpha_elec - 1
            shi_class = classify_shi(somo_is_homo, homo_ener - somo_ener, a_homo_ener, b_homo_ener)
        else:
            somo_label, sumo_label = b_label(b_idx), a_label(a_idx)
            somo_ener, sumo_ener, homo_ener = b_full_ener[b_idx], a_full_ener[a_idx], b_homo_ener
            somo_is_homo = b_idx == n_beta_elec - 1
            shi_class = classify_shi(somo_is_homo, homo_ener - somo_ener, b_homo_ener, a_homo_ener)
        shi_gap = 0.0 if somo_is_homo else homo_ener - somo_ener
        log.write(
            f"{somo_label:<10s}  {sumo_label:<10s} {homo_ener:>8.4f}  {somo_ener:>8.4f}  {sumo_ener:>8.4f}  "
            f"{shi_gap:>7.4f}  {'partial-SHI' if shi_class == 'partial' else shi_class}\n"
        )
        print(somo_label, sumo_label, shi_gap, shi_class)
    if n_vv > 0:
        log.write("\nvirt.-virt. pairs have no occupied partner: no SHI.\n")

    if args.fch:
        write_fchk(
            args.fch,
            {
                "alpha": (a_full_coef, a_full_ener / HARTREE_TO_EV),  # fchk stores Hartree
                "beta": (b_full_coef, b_full_ener / HARTREE_TO_EV),
            },
        )
        log.write("\nTransformed alpha and beta MOs written to transformed_orbitals.fchk\n")
    if args.movecs:
        write_ascii_movecs(args.movecs, nbas, a_full_ener, a_full_coef)
        log.write("\nOnly the alpha MOs are written to the NWChem movecs file.\n")
    return True


if args.target:
    try:
        a_targets, b_targets = parse_target_lumos(args.target, n_alpha_elec, n_beta_elec, nbas)
    except ValueError as error:
        log.write(f"\nERROR: {error}\n")
        log.close()
        sys.exit(f"ERROR: {error}")
    match_targets(a_targets, b_targets)
else:
    shi_match()

log.write(f"\n\nTotal elapsed time: {time.time() - start_time:.1f} seconds\n")
log.close()
