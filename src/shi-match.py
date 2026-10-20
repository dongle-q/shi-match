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
        write_fchk(args.fch, a_full_coef, a_full_ener)

    return True


shi_match()

log.write(f"\n\nTotal elapsed time: {time.time() - start_time:.1f} seconds\n")
log.close()
