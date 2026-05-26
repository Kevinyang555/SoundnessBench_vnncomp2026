import argparse
import csv
import os
import shutil

import torch


MODEL_CONFIGS = [
    ("model", "assets_0.pt", 30),
    ("model_residual", "assets_1.pt", 30),
]
TIMEOUT = 150


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", default=0, type=int)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_text_specs(C_mat, rhs_mat):
    nonzero_cumsum = (C_mat != 0).int().cumsum(dim=-1)
    first_var_idx = (nonzero_cumsum == 1).max(dim=-1)[1]
    second_var_idx = (nonzero_cumsum == 2).max(dim=-1)[1]
    var_var_mask = second_var_idx != 0

    rhs_mat = torch.where(var_var_mask, torch.zeros_like(rhs_mat), rhs_mat)

    first_var_sign = C_mat.gather(2, first_var_idx.unsqueeze(-1)).squeeze(-1).sign()
    smaller_eq_mask = first_var_sign == 1
    C_mat = C_mat * first_var_sign.unsqueeze(-1)
    rhs_mat = rhs_mat * first_var_sign

    second_var_pos_mask = C_mat.gather(2, second_var_idx.unsqueeze(-1)).squeeze(-1) > 0

    spec_gen_list = []
    for i in range(C_mat.shape[0]):
        spec_gen = []
        for j in range(C_mat.shape[1]):
            if var_var_mask[i, j]:
                op = "<=" if smaller_eq_mask[i, j] else ">="
                sign = "-" if second_var_pos_mask[i, j] else ""
                spec_text = f"({op} Y_{first_var_idx[i, j]} {sign}Y_{second_var_idx[i, j]})"
            else:
                op = "<=" if smaller_eq_mask[i, j] else ">="
                spec_text = f"({op} Y_{first_var_idx[i, j]} {rhs_mat[i, j]})"
            spec_gen.append(spec_text)
        spec_gen_list.append(spec_gen)

    return spec_gen_list


def generate_vnnlib_single_input(vnnlib_path, data_min, data_max, C_mat, rhs_mat):
    num_distinct_x = data_min.shape[0]
    assert num_distinct_x == 1, "Assume only one input range"
    data_min = data_min.view(-1)
    data_max = data_max.view(-1)
    input_dim = data_min.shape[0]
    output_dim = C_mat.shape[-1]

    spec_list = generate_text_specs(C_mat, rhs_mat)[0]

    with open(vnnlib_path, "w", encoding="utf-8") as f:
        for i in range(input_dim):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        for i in range(output_dim):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        f.write("; Input constraints:\n")
        for i in range(input_dim):
            f.write(f"(assert (<= X_{i} {data_max[i]}))\n")
            f.write(f"(assert (>= X_{i} {data_min[i]}))\n")
            f.write("\n")
        f.write("\n")

        f.write("; Output constraints:\n")
        f.write("(assert (or\n")
        f.write("    (and")
        for spec_text in spec_list:
            f.write(f" {spec_text}")
        f.write(")\n")
        f.write("))\n")


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    args = get_args()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    set_seed(args.seed)

    bench_dir = "."
    onnx_dir_name = "onnx"
    metadata_dir_name = "metadata"
    vnnlib_dir_name = "vnnlib"
    vnnlib_dir = os.path.join(bench_dir, vnnlib_dir_name)
    property_dir = os.path.join(vnnlib_dir, "properties")
    reset_dir(property_dir)

    selected = []
    for model_name, metadata_name, count in MODEL_CONFIGS:
        local_onnx_path = os.path.join(onnx_dir_name, f"{model_name}.onnx")
        csv_onnx_path = f"{onnx_dir_name}/{model_name}.onnx"
        metadata_path = os.path.join(metadata_dir_name, metadata_name)
        if not os.path.exists(local_onnx_path):
            raise FileNotFoundError(local_onnx_path)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(metadata_path)

        metadata = torch.load(metadata_path, map_location=device, weights_only=False)
        data_min = metadata["data_min"].to(device)
        data_max = metadata["data_max"].to(device)
        spec_mat = metadata["spec_mat"].to(device)
        rhs_mat = metadata["rhs_mat"].to(device)
        if count > rhs_mat.shape[0]:
            raise ValueError(f"not enough properties for {model_name}")

        spec_indices = torch.randperm(rhs_mat.shape[0], device=device)[:count].tolist()
        for idx in spec_indices:
            selected.append(
                (
                    csv_onnx_path,
                    data_min[idx : idx + 1],
                    data_max[idx : idx + 1],
                    spec_mat[idx : idx + 1],
                    rhs_mat[idx : idx + 1],
                )
            )

    order = torch.randperm(len(selected), device=device).tolist()
    csv_items = []
    for public_idx, selected_idx in enumerate(order):
        onnx_path, data_min, data_max, spec_mat, rhs_mat = selected[selected_idx]
        vnnlib_name = f"property_{public_idx:03d}.vnnlib"
        csv_vnnlib_path = f"{vnnlib_dir_name}/properties/{vnnlib_name}"
        vnnlib_path = os.path.join(property_dir, vnnlib_name)
        generate_vnnlib_single_input(vnnlib_path, data_min, data_max, spec_mat, rhs_mat)
        csv_items.append((onnx_path, csv_vnnlib_path, TIMEOUT))

    with open(os.path.join(bench_dir, "instances.csv"), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_items)

    print(f"Generated {len(csv_items)} instances for seed {args.seed}")


if __name__ == "__main__":
    main()
