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
from jax import config
from jax.scipy.linalg import expm
from scipy.optimize import minimize

start_time = time.time()

parser = argparse.ArgumentParser(
    description="SHI-match: A tool for SOMO-HOMO Inversion."
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
    "-ascii-movecs",
    dest="movecs",
    default=None,
    help="movecs file from NWChem. (Note: Please add nwchem to PATH and compile mov2asc NWCHEM_TOP/contrib/mov2asc).",
)
parser.add_argument(
    "-gaussian",
    "-g16",
    "-fch",
    "-fchk",
    dest="fch",
    default=None,
    help="formated checkpoint file (fch, fchk) from Gaussian.",
)

parser.add_argument(
    "-extend",
    "-ext",
    "-add",
    dest="ext_mo",
    type=int,
    default=0,
    help="Add more vir. MOs (alpha+beta) for orbital rotation. Non-converged calculations will automatically add orbitals to the rotation space. This option lets users control the calculation more flexibly."
)

parser.add_argument(
    "-hw",
    "--hardware",
    dest="hardware",
    default=None,
    help="Hardware platform to use (cpu, cuda, rocm, tpu). Use cuda for NVIDIA GPU and rocm for AMD GPU.",
)

args = parser.parse_args()


# JAX setup

if args.hardware is None:
    jax.config.update("jax_platforms", "cpu")
else:

    hardware_choice = args.hardware.lower()
    if args.hardware.lower() not in ["cpu", "cuda", "rocm", "tpu"]:
        parser.error("Invalid hardware option. Please choose from 'cpu', 'cuda', 'rocm', or 'tpu'.")
    else:
        jax.config.update("jax_platforms", hardware_choice)

# Create a log file and print the header
log = open("shi_match.log", "w")
log.write(r"""
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
       H
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


toEV = 27.2113245702


def formatted_overlap(number):
    return "       " if abs(number) < 0.01 else f"{number: >+7.4f}".replace("+", " ")

def label_mo(is_alpha, idx):
    homo_idx = a_HOMO_idx if is_alpha else b_HOMO_idx
    return "[*]" if idx <= homo_idx else "[ ]"


def print_overlap(smat, numb):
    print_range = range(max(0,n_alpha_elec + args.ext_mo - numb), n_alpha_elec +  args.ext_mo)
    header = "          " + "".join(f"  b_{b_idx + 1:04d}" for b_idx in print_range)+"\n"
    log.write(header)
    header = "          " + "".join(f"     {label_mo(False, b_idx)}" for b_idx in print_range)+"\n"
    log.write(header)

    log.write("-"*(10+6*len(print_range))+"\n")
    for a_idx in print_range:
        row = f"a_{a_idx + 1:04d} {label_mo(True,a_idx)}"
        for b_idx in range(n_alpha_elec - numb, n_alpha_elec):
            row += f" {formatted_overlap(smat[a_idx][b_idx])}"
        log.write(row + "\n")

    log.write("\nNote: [*] = occupied,      [ ] = virtual\n")


def write_arr_movecs(ascii_output, arr):
    nelems = len(arr)
    nlines = nelems // 3
    ncount = 0
    for _ in range(nlines):
        ascii_output.write(
            "".join(f"   {arr[i]:>22.15e}" for i in range(ncount, ncount + 3)) + "\n"
        )
        ncount += 3
    if nelems % 3 != 0:
        ascii_output.write(
            "".join(f"   {arr[i]:>22.15e}" for i in range(ncount, nelems)) + "\n"
        )
        nlines += 1
    return nlines


def write_ascii_movecs(a_ener, a_coef):
    nbas_ascii = a_coef.shape[0]
    with open(args.movecs, "r") as canonical_ascii:
        with open("transformed_orbitals.ascii", "w") as transformed_ascii:
            while True:
                line = canonical_ascii.readline()
                transformed_ascii.write(line)
                if "ao basis" in line:
                    break
            # Next three lines containing: (1) interger indicates Restricted or Unrestricted calc.
            #                              (2) NBAS
            #                              (3) Numbers of MO: alpha beta
            line = canonical_ascii.readline()
            sys.exit(
                f"Wrong ASCII movecs file. {args.movecs} cotains restricted calculation."
            ) if int(line) == 1 else None
            transformed_ascii.writelines(line)
            line = canonical_ascii.readline()
            transformed_ascii.writelines(line)
            line = canonical_ascii.readline()
            transformed_ascii.writelines(line)

            # Skip lines containing alpha occupation
            nlines = nbas // 3 if nbas % 3 == 0 else nbas // 3 + 1
            for _ in range(nlines):
                line = canonical_ascii.readline()
                transformed_ascii.writelines(line)

            # Write alpha energy
            nlines = write_arr_movecs(transformed_ascii, a_ener)
            # Write alpha coeff
            for iOrb in range(nbas_ascii):
                nlines += write_arr_movecs(transformed_ascii, a_coef[iOrb])
            # Skip these lines
            for _ in range(nlines):
                canonical_ascii.readline()

            # Copy the rest of the file
            line = canonical_ascii.readline()
            while line:
                transformed_ascii.writelines(line)
                line = canonical_ascii.readline()


def get_number(string):
    # Purpose: Sometimes, Fortran code prints a wrong number: e.g 9.1-300 = 9.1E-300
    if "E" in string:
        return float(string)
    else:
        if "+" in string:
            return float(string.split("+")[0]) * 10 ** (int(string.split("+")[1]))
        elif "-" in string:
            return float(string.split("-")[0]) * 10 ** (-int(string.split("-")[1]))


def get_upper_tri_matrix_file47(file47, nmat, nbas):
    # nmat: number of matrices to read
    temp = []
    line = file47.readline()  # Skip the header line
    a_mat = []
    while "$END" not in line.upper():
        numbers = line.split()
        for number in numbers:
            temp.append(get_number(number))
        line = file47.readline()
        if "END" in line.upper():
            break
    if nmat == 2:
        temp_a = temp[: int(nbas * (nbas - 1) / 2 + nbas)]
        temp_b = temp[int(nbas * (nbas - 1) / 2 + nbas) :]
        a_mat = np.zeros((nbas, nbas), dtype=np.float64)
        b_mat = np.zeros((nbas, nbas), dtype=np.float64)
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k = k + 1
                a_mat[i][j] = temp_a[k]
                a_mat[j][i] = temp_a[k]
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k = k + 1
                b_mat[i][j] = temp_b[k]
                b_mat[j][i] = temp_b[k]
        return a_mat, b_mat
    else:
        matrix = np.zeros((nbas, nbas), dtype=np.float64)
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k = k + 1
                matrix[i][j] = temp[k]
                matrix[j][i] = temp[k]
        return matrix


def get_square_matrix_file47(file47, nmat, nbas):
    temp = []
    line = file47.readline()  # Skip the header line
    while "$END" not in line.upper():
        numbers = line.split()
        for number in numbers:
            temp.append(get_number(number))
        line = file47.readline()
    if nmat == 2:
        a_matrix = np.array(temp[: nbas * nbas], dtype=np.float64).reshape(nbas, nbas)
        b_matrix = np.array(temp[nbas * nbas :], dtype=np.float64).reshape(nbas, nbas)
        return a_matrix, b_matrix
    else:
        matrix = np.array(temp, dtype=np.float64).reshape(nbas, nbas)
        return matrix


def get_dat_from_file47(filename):
    with open(filename) as file47:
        line = file47.readline()
        if "OPEN" in line:
            open_shell = True
        else:
            open_shell = False
            log.write(
                "Please check your molecular system (geometry, charge, ab initio input)"
            )
            sys.exit(
                "FILE47 contains the closed shell calculation. SHI-match requires open shell calculation."
            )

        # Get number of basis functions: NBAS
        # The first has a format like that: $GENNBO  UPPER  BODM  BOHR  NATOMS= 3  NBAS=  24 $END
        #  Next to the string "NBAS=" is the number of basis functions
        if "NBAS" in line:
            # If nbas<100, Fortran developer leaves a space after "NBAS= 99"
            # this is different with the case "NBAS=100"
            for token in line.split():
                if token.startswith("NBAS="):
                    nbas_str = token[len("NBAS=") :]
                    # There is a space if nbas_str=="", because line was split by space
                    if nbas_str == "":
                        idx = line.split().index(token)
                        nbas = int(line.split()[idx + 1])
                    # This is the case when nbas>100
                    else:
                        nbas = int(nbas_str)
                    break
            else:
                sys.exit("Failed to parse NBAS from FILE47")
        else:
            sys.exit(
                "NBAS is not found in the first line of FILE47. Please check the FILE47."
            )
        log.write(f"\n\n\nNumber of basis functions (NBAS): {nbas}\n")

        # Parse the rest of the file
        while line:
            if "$LCAOMO" in line.upper():
                a_coef_mat, b_coef_mat = get_square_matrix_file47(file47, 2, nbas)
            elif "$DENSITY" in line.upper():
                a_density_mat, b_density_mat = get_upper_tri_matrix_file47(
                    file47, 2, nbas
                )
            elif "$FOCK" in line.upper():
                a_fock_mat, b_fock_mat = get_upper_tri_matrix_file47(file47, 2, nbas)
            elif "$OVERLAP" in line.upper():
                bf_ovlp_mat = get_upper_tri_matrix_file47(file47, 1, nbas)
            line = file47.readline()

        # Calculate numbers of alpha, beta electrons by density matrix.
        n_alpha_elec = int(round(np.trace(a_density_mat @ bf_ovlp_mat)))
        n_beta_elec = int(round(np.trace(b_density_mat @ bf_ovlp_mat)))

        return (
            a_coef_mat,
            b_coef_mat,
            bf_ovlp_mat,
            a_density_mat,
            b_density_mat,
            a_fock_mat,
            b_fock_mat,
            nbas,
            n_alpha_elec,
            n_beta_elec,
        )


def match_orbital(ovlp_ab, threshold=0.97):
    max_idx = np.argmax(abs(ovlp_ab))
    max_ovlp = ovlp_ab[max_idx]
    if abs(max_ovlp) >= threshold:
        return max_idx, max_ovlp
    return None, None


# Get C, S_ao, DM, Fock matrices, nbas, number of alpha-beta electrons from file47
# density matrix is not used, they are discarded.

(
    c_a_scf_full,
    b_full_coef,
    ovlp_bf,
    _,
    _,
    fock_a_mat,
    fock_b_mat,
    nbas,
    n_alpha_elec,
    n_beta_elec,
) = get_dat_from_file47(file47)

dlt_elec = n_alpha_elec - n_beta_elec
a_HOMO_idx = n_alpha_elec - 1
b_HOMO_idx = n_beta_elec - 1
b_SUMO_idx = n_beta_elec

log.write("\n")
log.write(f'''
                 Alpha electrons: {n_alpha_elec}
                  Beta electrons: {n_beta_elec}
                Alpha HOMO index: {a_HOMO_idx + 1}
                 Beta HOMO index: {b_HOMO_idx + 1}
          Lowest Beta SUMO index: {b_SUMO_idx + 1}
                            2S+1: {0.5 * dlt_elec * 2 + 1:.0f}
             Additional vir. MOs: {args.ext_mo}''')
# Get the MO coefs
# alpha_coef and beta_coef are NOT full coefs matrix. It only include alpha occupied  (all occ)
# and beta occupied + beta SUMO (all occ+1virt)
init_S_ab = reduce(np.dot, (c_a_scf_full, ovlp_bf, b_full_coef.T))


# Alpha set contains orbitals that will be used in rotation
alpha_set = list(range(1, n_alpha_elec + args.ext_mo + 1))
beta_set = alpha_set.copy()


log.write("\nINITIAL OVERLAP\n")
log.write("=====================\n")
print_overlap(init_S_ab, 10)

init_a_maxovlp_idx = np.argmax(abs(init_S_ab[:, n_beta_elec]))
init_maxovlp = init_S_ab[init_a_maxovlp_idx, n_beta_elec]
log.write(
    f"\nMaximal overlap with lowest beta SUMO:    < b_{n_beta_elec + 1:>} | a_{init_a_maxovlp_idx + 1:<} > = {init_maxovlp:<5.4f}\n"
)

log.write("\n")


# Define the shape of rotation matrix and unique antisymmetric elements
n_rot_elem = len(alpha_set)
n_uniq_antisym_elem = (
    n_rot_elem * (n_rot_elem - 1) // 2
)


c_a_occ = c_a_scf_full.T[:, np.array(alpha_set) - 1]
c_b_occ_pl_sumo = b_full_coef.T[:, np.array(beta_set) - 1]
init_ovlp_ab = jnp.dot(c_a_occ.T, jnp.dot(ovlp_bf, c_b_occ_pl_sumo))


def check_permuted_identity(overlap, threshold=0.05):
    abs_overlap = abs(overlap)
    n_orbital = abs_overlap.shape[0]
    strong_ovlp = []
    for orb_idx in range(n_orbital):
        row = abs_overlap[orb_idx]
        # Find index of max element
        max_idx = np.argmax(row)
        max_ovlp = row[max_idx]
        if max_ovlp < (1.0 - threshold):
            return False  # No element close to 1
        strong_ovlp.append(max_idx)

    # Check if all strong overlap are unique [set in Python will remove duplicates]
    if len(set(strong_ovlp)) != n_orbital:
        return False

    # Sort rows by their strong overlap index --> Create a identity permuted matrix
    sorted_rows = sorted(
        range(n_orbital), key=lambda orb_idx: np.argmax(abs_overlap[orb_idx])
    )
    permuted_mat = abs_overlap[sorted_rows]

    # Verify
    for orb_idx in range(n_orbital):
        if permuted_mat[orb_idx, orb_idx] < (1.0 - threshold):
            return False
        for j in range(n_orbital):
            if orb_idx != j and permuted_mat[orb_idx, j] > threshold:
                return False
    return True


def antisymm_mat_from_vec(vec):
    X = jnp.zeros((n_rot_elem, n_rot_elem))
    triu_indices = jnp.triu_indices(n_rot_elem, k=1)
    X = X.at[triu_indices].set(vec)
    X = X - X.T
    return X


def object_func(vec, s_ab_matrix):
    # Form antisymmetric matrix from vec
    W = antisymm_mat_from_vec(vec)
    # Form rotation matrix from X
    Rt = expm(W)
    # Compute S' = C'_alpha.T S_ao C_beta
    #            = (C_alpha * R).T S_ao C_beta
    #            = R.T C_alpha.T S_ao C_beta
    overlap = jnp.dot(Rt.T, s_ab_matrix)
    # Compute penalty by Frobenius Norm of ABS(overlap) subtract I (identity matrix)
    obj_matrix = overlap**2 - jnp.eye(n_rot_elem)
    J2 = jnp.sum(jnp.square(obj_matrix))
    return J2


def write_arr_fchk(arr, fchk_output, fchk_input):
    nelems = len(arr)
    nlines = nelems // 5
    ncount = 0
    for _ in range(nlines):
        fchk_output.write(
            "".join(f" {arr[i]:>15.8e}" for i in range(ncount, ncount + 5)) + "\n"
        )
        ncount += 5
    if nelems % 5 != 0:
        fchk_output.write(
            "".join(f" {arr[i]:>15.8e}" for i in range(ncount, nelems)) + "\n"
        )
        nlines += 1
    # Skip lines to update cursor of fchk_input
    for _ in range(nlines):
        fchk_input.readline()
    line = fchk_input.readline()
    return line


def write_fchk(a_coefs, a_eners):
    with open(args.fch, "r") as fchk_input:
        with open("transformed_orbitals.fchk", "w+") as fchk_output:
            line = fchk_input.readline()
            while line:
                if (
                    "alpha orbital energies" not in line.lower()
                    and "alpha mo coefficients" not in line.lower()
                ):
                    fchk_output.write(line)
                    line = fchk_input.readline()
                elif "alpha orbital energies" in line.lower():
                    # Write the header line
                    fchk_output.write(line)
                    line = write_arr_fchk(a_eners, fchk_output, fchk_input)

                elif "alpha mo coefficients" in line.lower():
                    # Write the header line
                    fchk_output.write(line)
                    line = write_arr_fchk(a_coefs.flatten(), fchk_output, fchk_input)


# Compute initial guess for optimization
# Use SVD to compute initial rotation matrix, then logm to get initial W matrix
U, S, Vt = np.linalg.svd(init_ovlp_ab)
init_Rt = U @ Vt
init_W = np.real(la.logm(init_Rt))
upper_indices = np.triu_indices_from(init_W, k=1)
init_guess = init_W[upper_indices]


def shi_match():

    grad_fn = jax.grad(object_func)

    step = 0

    def callback_func(xk):
        nonlocal step
        J2 = object_func(xk, init_ovlp_ab)
        log.write(f"{step:<4d}   {J2:>13.8f} \n")
        step += 1

    log.write("====================\n")
    log.write("    CONVERGENCE\n")
    log.write("====================\n")
    log.write("step      obj. func.\n")
    log.write("--------------------\n")

    result = minimize(
        lambda v, s_ab_matrix: np.asarray(object_func(v, s_ab_matrix)),
        init_guess,
        args=(init_ovlp_ab),
        jac=lambda v, s_ab_matrix: np.asarray(grad_fn(v, s_ab_matrix)),
        method="CG",
        callback=callback_func,
    )

    log.write("\n")
    log.write(f"The minimization is done after {result.nit} steps.\n")
    final_J2 = object_func(result.x, init_ovlp_ab)
    log.write(f"Squared Frobenius norm ||S^2-1||2F  =  {final_J2:.5f} \n")

    # Extract the final result
    final_rotation_matrix = expm(antisymm_mat_from_vec(result.x))
    log.write("--------------------\n")

    a_transformed_occ_coef = np.dot(c_a_occ, final_rotation_matrix)

    a_transformed_ener = toEV * (
        (a_transformed_occ_coef.T @ fock_a_mat @ a_transformed_occ_coef).diagonal()
    )
    a_canonical_ener = toEV * ((c_a_scf_full @ fock_a_mat @ c_a_scf_full.T).diagonal())

    ### new_alpha_energies: new alpha_occupied energies
    argsort_ener = np.argsort(a_transformed_ener)
    a_transformed_ener = a_transformed_ener[argsort_ener]
    # -------------
    a_transformed_occ_coef = a_transformed_occ_coef[:, argsort_ener]

    final_overlap = a_transformed_occ_coef.T @ ovlp_bf @ c_b_occ_pl_sumo
    log.write("\nFINAL OVERLAP\n")
    log.write("=====================\n")
    print_overlap(final_overlap, 10)

    log.write("\n")
    if final_J2 > 0.5:
        log.write("WARNING: There might some orbital swappings.\n")

    log.write("\n")

    if not check_permuted_identity(final_overlap):
        log.write("WARNING: Overlap matrix is not permuted identity.\n")
        log.write("JOB FAILED\n")
        return False

    b_full_ener = toEV * ((b_full_coef @ fock_b_mat @ b_full_coef.T).diagonal())
    log.write("================================\n")
    log.write("    Alpha MO energies (eV)\n")
    log.write("================================\n")
    log.write("i       canonical    transformed  \n")
    log.write("--------------------------------\n")
    for a_idx in range(n_alpha_elec):
        log.write(
            f"""{a_idx + 1:<4d} {a_canonical_ener[a_idx]:>12.3f}   {a_transformed_ener[a_idx]:>12.3f}   \n"""
        )

    log.write("\n\n")
    log.write("==========================================\n")
    log.write("            MO energies eV\n")
    log.write("==========================================\n")
    log.write("i  transformed ALPHA  canonical BETA  S_ii\n")
    log.write("------------------------------------------\n")
    for a_idx in range(n_alpha_elec):
        ovlp_str = (
            f"{final_overlap[a_idx, a_idx]:>5.2f}"
            if abs(final_overlap[a_idx, a_idx]) > 0.01
            else "  0  "
        )
        log.write(
            f"""{a_idx + 1:<4d} {a_transformed_ener[a_idx]:>13.3f}   {b_full_ener[a_idx]:>13.3f}   {ovlp_str}\n"""
        )

    a_SOMO_idx = []
    b_SUMO_idx = []

    log.write("\n\n")
    log.write("===============================\n")
    log.write("Matching transf. alpha and beta\n")
    log.write("===============================\n")
    log.write("alpha(i)  beta(j)  S_ij    swap\n")
    log.write("-------------------------------\n")
    for a_idx in range(n_alpha_elec):
        b_idx, ovlp = match_orbital(final_overlap[a_idx, :])
        note = "   " if a_idx == b_idx else "yes"
        log.write(
            f"""   {a_idx + 1:>5d}  {b_idx + 1:<5d}    {ovlp:>5.2f}    {note} \n"""
        )
        if b_idx >= n_beta_elec:
            a_SOMO_idx.append(a_idx)
            b_SUMO_idx.append(b_idx)

    log.write("\n\n")
    log.write("=============================================\n")
    log.write("        Alpha SOMO(i) and beta SUMO(j)\n")
    log.write("=============================================\n")
    log.write("   i - j     S_ij        a_SOMO        b_SUMO\n")
    log.write("---------------------------------------------\n")
    for idx in range(dlt_elec):
        a_idx = a_SOMO_idx[idx]
        b_idx = b_SUMO_idx[idx]
        ovlp = final_overlap[a_idx, b_idx]
        log.write(
            f"""{a_idx + 1:>5d} {b_idx + 1:<5d} {ovlp:>5.2f} {a_transformed_ener[a_idx]:>13.3f} {b_full_ener[b_idx]:>13.3f} \n"""
        )
        if b_idx == n_beta_elec:
            matched_a_SOMO = a_transformed_ener[a_idx]
            matched_a_idx = a_idx

    b_HOMO_ener = b_full_ener[b_HOMO_idx]
    a_HOMO_ener = np.max(a_transformed_ener)

    shi_gap = a_HOMO_ener - matched_a_SOMO
    log.write("\n\n")
    log.write("=====================================\n")
    log.write("              SHI GAP\n")
    log.write("=====================================\n")
    log.write("  alpha                 beta\n\n")
    log.write(
        f"  HOMO: {a_HOMO_ener:<10.4f}      SUMO: {b_full_ener[n_beta_elec]:<10.4f}\n"
    )
    if a_HOMO_idx != matched_a_idx:
        log.write(f"                        HOMO: {b_full_ener[b_HOMO_idx]:<10.4f}\n\n")
        log.write(f"  SOMO: {matched_a_SOMO:<10.4f}\n")
    else:
        log.write(
            f"  SOMO: {matched_a_SOMO:<10.4f}      HOMO: {b_full_ener[b_HOMO_idx]:<10.4f}\n"
        )
    log.write(f"  ---------------\n  SHI gap: {shi_gap:.4f}\n")
    a_full_ener = np.concatenate((a_transformed_ener, a_canonical_ener[n_alpha_elec:]))
    a_full_coef = np.vstack((a_transformed_occ_coef.T, c_a_scf_full[n_alpha_elec:, :]))
    log.write("\n")

    if shi_gap == 0.0 and b_HOMO_ener < a_HOMO_ener:
        shi_class = "non-SHI"
        log.write("Classification: non-SHI\n")
    elif shi_gap == 0.0 and b_HOMO_ener > a_HOMO_ener:
        shi_class = "partial"
        log.write("Classification: partial-SHI")
    elif shi_gap > 0.0 and b_HOMO_ener > a_HOMO_ener:
        shi_class = "SHI"
        log.write("Classification: SHI")

    print(shi_gap, shi_class)

    if args.movecs:
        write_ascii_movecs(
            a_full_ener,
            a_full_coef,
        )

    if args.fch:
        write_fchk(a_full_coef, a_full_ener)

    return True


shi_match()

end_time = time.time()
elapsed_time = end_time - start_time
log.write(f"\n\nTotal elapsed time: {elapsed_time:.1f} seconds\n")
log.close()
