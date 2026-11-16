def format_overlap(value, is_final, b_idx, b_homo_idx):
    # 1% is the threshold. If ovlp < 1% --> print nothing
    if abs(value) < 0.01:
        return "         "
    return f"{value: >+9.4f}".replace("+", " ")


def label_mo(idx, homo_idx):
    return "[*]" if idx <= homo_idx else "[ ]"


def print_overlap(ovlp, n_show, is_final, log, n_alpha_elec, a_homo_idx, b_homo_idx):
    show_range = range(max(0, n_alpha_elec - n_show), n_alpha_elec)
    log.write("          " + "".join(f"    b_{b_idx + 1:04d}" for b_idx in show_range) + "\n")
    log.write(
        "          "
        + "".join(f"       {label_mo(b_idx, b_homo_idx)}" for b_idx in show_range)
        + "\n"
    )

    log.write("-" * (10 + 10 * len(show_range)) + "\n")
    for a_idx in show_range:
        row = f"a_{a_idx + 1:04d} {label_mo(a_idx, a_homo_idx)}"
        row += "".join(
            f" {format_overlap(ovlp[a_idx][b_idx], is_final, b_idx, b_homo_idx)}"
            for b_idx in show_range
        )
        log.write(row + "\n")

    log.write("\n NOTE: [*] = occupied      [ ] = virtual\n")
