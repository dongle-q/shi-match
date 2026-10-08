def format_overlap(number, final_ovlp, b_idx, b_homo_idx):
    if final_ovlp and b_idx > b_homo_idx:
        return "       " if abs(number) < 0.01 else f"({number: >+5.2f})"
    return "       " if abs(number) < 0.01 else f"{number: >+7.2f}".replace("+", " ")


def label_mo(is_alpha, idx, a_homo_idx, b_homo_idx):
    homo_idx = a_homo_idx if is_alpha else b_homo_idx
    return "[*]" if idx <= homo_idx else "[ ]"


def print_overlap(
    smat, numb, final_ovlp, log, n_alpha_elec, ext_mo, a_homo_idx, b_homo_idx
):
    print_range = range(max(0, n_alpha_elec + ext_mo - numb), n_alpha_elec + ext_mo)
    header = "          " + "".join(f"  b_{b_idx + 1:04d}" for b_idx in print_range) + "\n"
    log.write(header)
    header = "          " + "".join(
        f"     {label_mo(False, b_idx, a_homo_idx, b_homo_idx)}"
        for b_idx in print_range
    ) + "\n"
    log.write(header)

    log.write("-" * (10 + 8 * len(print_range)) + "\n")
    for a_idx in print_range:
        row = f"a_{a_idx + 1:04d} {label_mo(True, a_idx, a_homo_idx, b_homo_idx)}"
        row += "".join(
            f" {format_overlap(smat[a_idx][b_idx], final_ovlp, b_idx, b_homo_idx)}"
            for b_idx in print_range
        )
        log.write(row + "\n")

    log.write("\n NOTE: [*] = occupied      [ ] = virtual\n")
