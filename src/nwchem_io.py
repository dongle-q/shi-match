import sys

import numpy as np


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


def write_ascii_movecs(movecs_path, nbas, a_ener, a_coef):
    nbas_ascii = a_coef.shape[0]
    with open(movecs_path, "r") as canonical_ascii:
        with open("transformed_orbitals.ascii", "w") as transformed_ascii:
            while True:
                line = canonical_ascii.readline()
                transformed_ascii.write(line)
                if "ao basis" in line:
                    break
            line = canonical_ascii.readline()
            if int(line) == 1:
                sys.exit(
                    f"Wrong ASCII movecs file. {movecs_path} cotains restricted calculation."
                )
            transformed_ascii.writelines(line)
            line = canonical_ascii.readline()
            transformed_ascii.writelines(line)
            line = canonical_ascii.readline()
            transformed_ascii.writelines(line)

            nlines = nbas // 3 if nbas % 3 == 0 else nbas // 3 + 1
            for _ in range(nlines):
                line = canonical_ascii.readline()
                transformed_ascii.writelines(line)

            nlines = write_arr_movecs(transformed_ascii, a_ener)
            for i_orb in range(nbas_ascii):
                nlines += write_arr_movecs(transformed_ascii, a_coef[i_orb])
            for _ in range(nlines):
                canonical_ascii.readline()

            line = canonical_ascii.readline()
            while line:
                transformed_ascii.writelines(line)
                line = canonical_ascii.readline()
