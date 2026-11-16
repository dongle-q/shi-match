import sys


def write_arr_movecs(ascii_output, arr):
    # Write arr (3 values per line) and return the number of lines written
    line_starts = range(0, len(arr), 3)
    for start in line_starts:
        ascii_output.write("".join(f"   {value:>22.15e}" for value in arr[start : start + 3]) + "\n")
    return len(line_starts)


def write_ascii_movecs(movecs_path, nbas, a_ener, a_coef):
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
            transformed_ascii.write(line)
            for _ in range(2):
                transformed_ascii.write(canonical_ascii.readline())

            # Copy the occupation numbers (3 values per line)
            for _ in range(-(-nbas // 3)):
                transformed_ascii.write(canonical_ascii.readline())

            n_lines = write_arr_movecs(transformed_ascii, a_ener)
            for mo_coef in a_coef:
                n_lines += write_arr_movecs(transformed_ascii, mo_coef)
            for _ in range(n_lines):
                canonical_ascii.readline()

            line = canonical_ascii.readline()
            while line:
                transformed_ascii.write(line)
                line = canonical_ascii.readline()
