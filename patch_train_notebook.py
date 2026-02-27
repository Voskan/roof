import nbformat

nb_path = "notebooks/train_deeproof.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

changed = False
for cell in nb.cells:
    if cell.cell_type == "code":
        if "NOTEBOOK_FAST_AUX_LOSSES" in cell.source and "NOTEBOOK_FAST_MODEL_PROFILE" in cell.source:
            # We want to replace True with False for these specific keys
            lines = cell.source.splitlines()
            new_lines = []
            for line in lines:
                if line.startswith("NOTEBOOK_FAST_AUX_LOSSES ="):
                    new_lines.append("NOTEBOOK_FAST_AUX_LOSSES = False")
                    changed = True
                elif line.startswith("NOTEBOOK_FAST_MODEL_PROFILE ="):
                    new_lines.append("NOTEBOOK_FAST_MODEL_PROFILE = False")
                    changed = True
                else:
                    new_lines.append(line)
            cell.source = "\n".join(new_lines)

if changed:
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook patched successfully.")
else:
    print("Notebook was not patched - no matching cell found.")
