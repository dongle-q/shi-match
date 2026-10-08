import numpy as np


def get_number(string):
    # Fortran output may omit the E in values such as 9.1-300.
    if "E" in string:
        return float(string)
    else:
        if "+" in string:
            return float(string.split("+")[0]) * 10 ** (int(string.split("+")[1]))
        elif "-" in string:
            return float(string.split("-")[0]) * 10 ** (-int(string.split("-")[1]))


def get_upper_tri_matrix_file47(file47, nmat, nbas):
    temp = []
    line = file47.readline()
    while "$END" not in line.upper():
        numbers = line.split()
        for number in numbers:
            temp.append(get_number(number))
        line = file47.readline()
        if "END" in line.upper():
            break
    if nmat == 2:
        matrix_size = int(nbas * (nbas - 1) / 2 + nbas)
        temp_a = temp[:matrix_size]
        temp_b = temp[matrix_size:]
        a_mat = np.zeros((nbas, nbas), dtype=np.float64)
        b_mat = np.zeros((nbas, nbas), dtype=np.float64)
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k += 1
                a_mat[i][j] = temp_a[k]
                a_mat[j][i] = temp_a[k]
        k = -1
        for i in range(nbas):
            for j in range(i + 1):
                k += 1
                b_mat[i][j] = temp_b[k]
                b_mat[j][i] = temp_b[k]
        return a_mat, b_mat

    matrix = np.zeros((nbas, nbas), dtype=np.float64)
    k = -1
    for i in range(nbas):
        for j in range(i + 1):
            k += 1
            matrix[i][j] = temp[k]
            matrix[j][i] = temp[k]
    return matrix


def get_square_matrix_file47(file47, nmat, nbas):
    temp = []
    line = file47.readline()
    while "$END" not in line.upper():
        numbers = line.split()
        for number in numbers:
            temp.append(get_number(number))
        line = file47.readline()
    if nmat == 2:
        a_matrix = np.array(temp[: nbas * nbas], dtype=np.float64).reshape(nbas, nbas)
        b_matrix = np.array(temp[nbas * nbas :], dtype=np.float64).reshape(nbas, nbas)
        return a_matrix, b_matrix
    return np.array(temp, dtype=np.float64).reshape(nbas, nbas)


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
        for idx, token in enumerate(line.split()):
            if token.startswith("NBAS="):
                nbas_str = token[len("NBAS=") :]
                if nbas_str:
                    nbas = int(nbas_str)
                else:
                    nbas = int(line.split()[idx + 1])
                break
        else:
            sys.exit("Failed to parse NBAS from FILE47")

        log.write(f"\n\n\nNumber of basis functions (NBAS): {nbas}\n")
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
