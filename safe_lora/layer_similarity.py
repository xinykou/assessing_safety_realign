import argparse
import torch
import os

def main():
    parser = argparse.ArgumentParser("layer_similarity for region identification methods: wanda, wandg and low_rank")
    parser.add_argument("--first_lora_layer_distributions", default="")
    parser.add_argument("--second_lora_layer_distributions", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--tau", default=0.5, type=float)
    parser.add_argument("--sparsity_ratio", default=0.1, type=float)
    args = parser.parse_args()

    first_record = []
    second_record = []
    if args.first_lora_layer_distributions == "" or args.second_lora_layer_distributions == "":
        raise ValueError("Please provide two lora_layer_distributions to compare")
    else:
        with open(args.first_lora_layer_distributions, "r") as f:
            for lin in f:
                parts = lin.strip().split(",")
                if len(parts) == 1:
                    continue
                else:
                    first_record.append(parts[0])

        with open(args.second_lora_layer_distributions, "r") as f:
            for lin in f:
                parts = lin.strip().split(",")
                if len(parts) == 1:
                    continue
                else:
                    second_record.append(parts[0])

    layer_similarity = []
    total_layers_num = len(first_record) if len(first_record) > len(second_record) else len(second_record)
    save_name = f"sparsity_ratio_{args.sparsity_ratio}-layer_similarity.txt"
    save_path = os.path.join(args.output_path, save_name)

    for name in first_record:
        if name in second_record:
            layer_similarity.append(name)

    similarity_ratio = len(layer_similarity) / total_layers_num
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write(f"Similarity ratio (tau{args.tau}): {similarity_ratio}\n")
    else:
        with open(save_path, "a") as f:
            f.write(f"Similarity ratio (tau{args.tau}): {similarity_ratio} \n")

    print(f"Similarity ratio (tau{args.tau}): {similarity_ratio}")


if __name__ == "__main__":
    main()