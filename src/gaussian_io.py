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
    for _ in range(nlines):
        fchk_input.readline()
    return fchk_input.readline()


def write_fchk(fch_path, a_coefs, a_eners):
    with open(fch_path, "r") as fchk_input:
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
                    fchk_output.write(line)
                    line = write_arr_fchk(a_eners, fchk_output, fchk_input)
                elif "alpha mo coefficients" in line.lower():
                    fchk_output.write(line)
                    line = write_arr_fchk(a_coefs.flatten(), fchk_output, fchk_input)
