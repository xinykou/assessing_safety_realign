import random
import argparse

from utils import *


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='')
    parser.add_argument("--total_num", type=int, default=1000,
                        help="Total number of examples in the merged dataset")
    parser.add_argument("--n", type=int, default=0,
                        help="Number of examples to sample from unsafety dataset")
    parser.add_argument("--p", type=float, default=0,
                        help="Proportion of unsafe examples to include in the merged dataset")
    parser.add_argument("--unsafe_path", type=str, default="unsafety/BeaverTails_unsafe.json",
                        help="Path to the unsafety dataset")

    args = parser.parse_args()

    n = args.n
    p = args.p
    total_num = args.total_num
    dataset_name = args.dataset_name
    unsafe_path = args.unsafe_path

    assert n <= total_num, "n should be less than total_num"
    assert 0 <= p <= 1, "p should be in [0, 1]"
    unsafe_data = load_json_file(unsafe_path)
    ft_data = load_json_file(f"{dataset_name}/train.json")

    if p == 0 and n == 0:
        unsafe_data = []
    else:
        if n == 0:
            unsafe_data = unsafe_data[:int(total_num * p)]
        elif p == 0:
            unsafe_data = unsafe_data[:n]

        for example in unsafe_data:
            example["flag"] = "unsafe"

    ft_data = ft_data[:total_num - len(unsafe_data)]
    for example in ft_data:
        example["flag"] = "finetune"

    print(f"Number of finetune examples: {len(ft_data)}")
    print(f"Number of unsafe examples: {len(unsafe_data)}")

    data = ft_data + unsafe_data
    random.seed(42)
    random.shuffle(data)
    new_data = []
    for example in data:
        if example.get("input", "") != "":
            prompt = PROMPT_DICT["prompt_input"]
        else:
            prompt = PROMPT_DICT["prompt_no_input"]
        new_data.append({
            "instruction": prompt.format_map(example),
            "input": "",
            "output": example["output"],
            "flag": example["flag"]
        })

    write_json_file(f"{dataset_name}/train_merged_total{total_num}_n{n}_p{p}.json", new_data)


if __name__ == '__main__':
    main()
