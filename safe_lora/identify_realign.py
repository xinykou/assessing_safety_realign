import argparse
import os
from inner import Lora_inner_Wrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import torch.nn as nn


def lora_operation():
    print("Searching for LoRa unsafe...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--unaligned_path", type=str,
                        help="Unaligned path")
    parser.add_argument("--aligned_path", type=str,
                        help="Aligned path")

    parser.add_argument("--model_path", type=str,
                        help="Base model path")
    parser.add_argument("--lora_path", type=str,
                        default="lora",
                        help="LoRa path")
    parser.add_argument("--tau", type=float,
                        default=0.1,
                        help="tau")
    parser.add_argument("--output_path", type=str,
                        default="lora",
                        help="output path")
    # ------------------------masking safety region------------------------
    parser.add_argument("--realign_type", type=str,
                        default="replace",
                        help="realign types: scale, mask_scale, mask_replace")
    parser.add_argument("--mask_path", type=str,
                        default=None,
                        help="mask path for safety region")
    parser.add_argument("--sparsity_ratio", type=float,
                        default=0.01,
                        help="sparsity ratio")
    parser.add_argument("--tau_change_enable", action="store_true",
                        help="if sparsity_ratio is fixed, then tau is ablationed")

    args = parser.parse_args()

    print(args)

    lora_Operation = Lora_inner_Wrapper(args)
    if args.realign_type == "scale":
        lora_Operation.identify_unsafe_lora()
    elif args.realign_type == "mask_scale":
        lora_Operation.identify_unsafe_lora()
    elif args.realign_type == "mask_replace":
        lora_Operation.identify_unsafe_region()
    else:
        raise ValueError("Invalid realign types")


if __name__ == "__main__":
    lora_operation()
