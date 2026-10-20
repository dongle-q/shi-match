import sys

import numpy as np


def get_number(string):
    # Fortran output may omit the E in values such as 9.1-300.
    if "E" in string:
        return float(string)
    if "+" in string:
        mantissa, exponent = string.split("+")
        return float(mantissa) * 10 ** int(exponent)
    if "-" in string:
        mantissa, exponent = string.split("-")
        return float(mantissa) * 10 ** (-int(exponent))


def read_values_file47(file47, stop_at_end=False):
    values = []
    line = file47.readline()
    while "$END" not in line.upper():
        values.extend(get_number(number) for number in line.split())
        line = file47.readline()
        if stop_at_end and "END" in line.upper():
            break
    return values


def lower_tri_to_symm(values, nbas):
    matrix = np.zeros((nbas, nbas), dtype=np.float64)
    matrix[np.tril_indices(nbas)] = values
    return matrix + np.tril(matrix, -1).T


def get_upper_tri_matrix_file47(file47, nmat, nbas):
    values = read_values_file47(file47, stop_at_end=True)
    if nmat == 2:
        tri_size = nbas * (nbas + 1) // 2
        return (
            lower_tri_to_symm(values[:tri_size], nbas),
            lower_tri_to_symm(values[tri_size:], nbas),
        )
    return lower_tri_to_symm(values, nbas)


def get_square_matrix_file47(file47, nmat, nbas):
    values = np.array(read_values_file47(file47), dtype=np.float64)
    if nmat == 2:
        return (
            values[: nbas * nbas].reshape(nbas, nbas),
            values[nbas * nbas :].reshape(nbas, nbas),
        )
    return values.reshape(nbas, nbas)


def get_dat_from_file47(filename, log):
    with open(filename) as file47:
        line = file47.readline()
        if "OPEN" not in line:
            log.write(
                "Please check your molecular system (geometry, charge, ab initio input)"
            )
            sys.exit(
                "FILE47 contains the closed shell calculation. SHI-match requires open shell calculation."
            )

        if "NBAS" not in line:
            sys.exit(
                "NBAS is not found in the first line of FILE47. Please check the FILE47."
            )
        tokens = line.split()
        for idx, token in enumerate(tokens):
            if token.startswith("NBAS="):
                nbas = int(token[len("NBAS=") :] or tokens[idx + 1])
                break
        else:
            sys.exit("Failed to parse NBAS from FILE47")

        log.write(f"\n\n\nNumber of basis functions (NBAS): {nbas}\n")
        while line:
            if "$LCAOMO" in line.upper():
                a_coef, b_coef = get_square_matrix_file47(file47, 2, nbas)
            elif "$DENSITY" in line.upper():
                a_density, b_density = get_upper_tri_matrix_file47(file47, 2, nbas)
            elif "$FOCK" in line.upper():
                a_fock, b_fock = get_upper_tri_matrix_file47(file47, 2, nbas)
            elif "$OVERLAP" in line.upper():
                ao_ovlp = get_upper_tri_matrix_file47(file47, 1, nbas)
            line = file47.readline()

        # Density matrices are only needed to count the electrons
        n_alpha_elec = int(round(np.trace(a_density @ ao_ovlp)))
        n_beta_elec = int(round(np.trace(b_density @ ao_ovlp)))
        return (
            a_coef,
            b_coef,
            ao_ovlp,
            a_fock,
            b_fock,
            nbas,
            n_alpha_elec,
            n_beta_elec,
        )
