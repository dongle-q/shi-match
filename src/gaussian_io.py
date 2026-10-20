def write_arr_fchk(arr, fchk_output, fchk_input):
    # Write arr (5 values per line) and skip the replaced lines of the input
    for start in range(0, len(arr), 5):
        fchk_output.write("".join(f" {value:>15.8e}" for value in arr[start : start + 5]) + "\n")
        fchk_input.readline()
    return fchk_input.readline()


def write_fchk(fch_path, a_coef, a_ener):
    with open(fch_path, "r") as fchk_input:
        with open("transformed_orbitals.fchk", "w+") as fchk_output:
            line = fchk_input.readline()
            while line:
                fchk_output.write(line)
                if "alpha orbital energies" in line.lower():
                    line = write_arr_fchk(a_ener, fchk_output, fchk_input)
                elif "alpha mo coefficients" in line.lower():
                    line = write_arr_fchk(a_coef.flatten(), fchk_output, fchk_input)
                else:
                    line = fchk_input.readline()
