import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, load_peft_weights
import torch.nn.functional as F


class Lora_inner_Wrapper(nn.Module):
    def __init__(self, args):
        super(Lora_inner_Wrapper, self).__init__()  # 调用父类的构造函数
        self.model_path = args.model_path
        self.lora_path = args.lora_path
        self.output_path = args.output_path
        self.realign_types = args.realign_type
        self.tau = args.tau
        self.sparsity_ratio = args.sparsity_ratio
        self.total_layers = 0
        self.tau_change_enable = args.tau_change_enable

        self.modified_layers = []  # 记录修改的层名字和cos_similarity值

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
        if args.lora_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                args.lora_path,
                torch_dtype=torch.bfloat16
            )
            print("LoRa model loaded")

        if args.mask_path is not None:
            self.mask = torch.load(args.mask_path)
            print("Mask loaded")
        else:
            self.mask = None

        self.delta_model = load_peft_weights(args.aligned_path)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        self.target_layer = {}

    def Scale_Opertaion(self, W_A=None, W_B=None, Vec=None):
        """
        This function wraps the LoRa layer for specific operations.
        """
        delta_W = W_A @ W_B * 2  # todo: alpha/rank = 2
        # 计算投影矩阵 C
        Vec = Vec
        Vec_T = Vec.t()
        Vec_Vec_T = Vec @ Vec_T
        norm_Vec = torch.norm(Vec, p='fro')  # Frobenius 范数
        C = Vec_Vec_T / norm_Vec
        # 计算 cos(CΔW, ΔW)
        try:
            C_delta_W = C @ delta_W
        except RuntimeError:
            print(f"Error in matrix multiplication: {C.shape}, {delta_W.shape}")
            raise RuntimeError
        cos_similarity = F.cosine_similarity(C_delta_W.flatten(), delta_W.flatten(), dim=0)

        # 如果 cos(CΔW, ΔW) < τ，则更新 ΔW
        if cos_similarity < self.tau:
            cos_similarity_value = cos_similarity.cpu().item()
            self.modified_layers.append((self.target_layer, cos_similarity_value))
            print(f"Alignment performed on {self.target_layer}")
            return C @ W_A
        else:
            return W_A

    def Fusion_Opertions(self, f_m, a_m=None):
        """
        This function wraps the LoRa layer for specific operations.
        """
        cos_similarity = F.cosine_similarity(f_m.flatten(), a_m.flatten(), dim=0)
        if cos_similarity < self.tau:
            cos_similarity_value = cos_similarity.cpu().item()
            self.modified_layers.append((self.target_layer, cos_similarity_value))
            print(f"Alignment performed on {self.target_layer}")
            if self.realign_types == "mask_replace":
                return True
            else:
                raise ValueError("Invalid realign types")
        else:
            return False

    def identify_unsafe_region(self):
        print("Searching for LoRa unsafe...")
        # 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                delta_name = name.replace('default.weight', 'weight')
                alignment_W_A = self.delta_model[delta_name]
                alignment_W_A = alignment_W_A.data.to(param.device)
                W_A = param.data.clone()
                # base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
                W_B_name = name.replace('lora_A', 'lora_B')
                W_B = self.model.state_dict()[W_B_name].data.clone()
                delta_B_name = W_B_name.replace('default.weight', 'weight')
                alignment_W_B = self.delta_model[delta_B_name]
                alignment_W_B = alignment_W_B.data.to(param.device)

                mask_name = name.split('.weight')[0]
                mask_W_B_name = W_B_name.split('.weight')[0]
                assert mask_name in self.mask, f"Layer {mask_name} not found in mask"
                lora_A_mask = self.mask.get(mask_name, None)
                lora_B_mask = self.mask.get(mask_W_B_name, None)
                lora_A_mask = lora_A_mask.data.to(W_A.device)
                lora_B_mask = lora_B_mask.data.to(W_B.device)

                alignment_W_A = alignment_W_A * lora_A_mask
                alignment_W_B = alignment_W_B * lora_B_mask
                W_A = W_A * lora_A_mask
                W_B = W_B * lora_B_mask

                self.target_layer = delta_name
                W_A = W_A.t()
                W_B = W_B.t()
                f_m = W_A @ W_B
                alignment_W_A = alignment_W_A.t()
                alignment_W_B = alignment_W_B.t()
                alignment_matrix = alignment_W_A @ alignment_W_B
                modifiable = self.Fusion_Opertions(f_m=f_m, a_m=alignment_matrix)

                # update lora_A
                if modifiable:
                    alignment_W_A = alignment_W_A.t()
                    n_lora_A_mask = ~lora_A_mask
                    param.data = param.data * n_lora_A_mask + alignment_W_A
                    # base_model.model.model.layers.0.self_attn.q_proj.lora_B.default
                    module_lora_B_name = W_B_name.split('.weight')[0]
                    specified_module = self.model.get_submodule(module_lora_B_name)

                    alignment_W_B = alignment_W_B.t()
                    n_lora_B_mask = ~lora_B_mask
                    specified_module.weight.data = specified_module.weight.data * n_lora_B_mask + alignment_W_B

                self.total_layers += 1

        total_layers = self.total_layers
        modified_layers = len(self.modified_layers)
        ratio = modified_layers / total_layers
        print(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        # Ensure all parameters are contiguous before saving
        for name, param in self.model.named_parameters():
            param.data = param.data.contiguous()

        # save the model and log the modified layers
        tau = f'tau_{str(self.tau)}'

        if self.mask is not None:
            tau = f'sparsity_ratio_{str(self.sparsity_ratio)}-{tau}'

        # self.model = self.model.merge_and_unload()
        if not self.tau_change_enable:
            save_path = os.path.join(self.output_path, tau)
            self.model.save_pretrained(save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(save_path)
        else:
            if os.path.exists(self.output_path):
                pass
            else:
                os.makedirs(self.output_path, exist_ok=True)
            save_path = self.output_path
        modified_layers_path = os.path.join(save_path, f"modified_layers_{tau}.txt")
        with open(modified_layers_path, 'w') as f:
            for layer in self.modified_layers:
                f.write(f"{layer}\n")

            f.write(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        print(f"Model saved at {save_path}")

    def identify_unsafe_lora(self):
        print("Searching for LoRa unsafe...")
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                # delta_name = name.split('base_model.model.')[1]
                # delta_name = delta_name.replace('lora_A.default.weight', 'weight')
                delta_name = name.replace('default.weight', 'weight')
                alignment_W_A = self.delta_model[delta_name]
                alignment_W_A = alignment_W_A.data.to(param.device)
                W_A = param.data
                W_B_name = name.replace('lora_A', 'lora_B')
                W_B = self.model.state_dict()[W_B_name].data
                delta_B_name = W_B_name.replace('default.weight', 'weight')
                alignment_W_B = self.delta_model[delta_B_name]
                alignment_W_B = alignment_W_B.data.to(param.device)

                # get the mask for the layer if it exists
                if self.mask is not None:
                    mask_name = name.split('.weight')[0]
                    mask_W_B_name = W_B_name.split('.weight')[0]
                    assert mask_name in self.mask, f"Layer {mask_name} not found in mask"
                    lora_A_mask = self.mask.get(mask_name, None)
                    lora_B_mask = self.mask.get(mask_W_B_name, None)
                    lora_A_mask = lora_A_mask.data.to(W_A.device)
                    lora_B_mask = lora_B_mask.data.to(W_B.device)
                    alignment_W_A = alignment_W_A * lora_A_mask
                    alignment_W_B = alignment_W_B * lora_B_mask

                self.target_layer = delta_name
                W_A = W_A.t()
                W_B = W_B.t()
                alignment_W_A = alignment_W_A.t()
                alignment_W_B = alignment_W_B.t()
                alignment_matrix = alignment_W_A @ alignment_W_B
                W_A = self.Scale_Opertaion(W_A=W_A, W_B=W_B, Vec=alignment_matrix)

                # Ensure the modified weights are contiguous before assignment
                W_A = W_A.contiguous()

                # update lora_A
                param.data = W_A.t()
                self.total_layers += 1

        total_layers = self.total_layers
        modified_layers = len(self.modified_layers)
        ratio = modified_layers / total_layers
        print(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        # Ensure all parameters are contiguous before saving
        for name, param in self.model.named_parameters():
            param.data = param.data.contiguous()

        # save the model and log the modified layers
        tau = f'tau_{str(self.tau)}'

        if self.mask is not None:
            tau = f'sparsity_ratio_{str(self.sparsity_ratio)}-{tau}'

        # self.model = self.model.merge_and_unload()
        if not self.tau_change_enable:
            save_path = os.path.join(self.output_path, tau)
            self.model.save_pretrained(save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(save_path)
        else:
            if os.path.exists(self.output_path):
                pass
            else:
                os.makedirs(self.output_path, exist_ok=True)
            save_path = self.output_path
        modified_layers_path = os.path.join(save_path, f"modified_layers_{tau}.txt")
        with open(modified_layers_path, 'w') as f:
            for layer in self.modified_layers:
                f.write(f"{layer}\n")

            f.write(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        print(f"Model saved at {save_path}")
