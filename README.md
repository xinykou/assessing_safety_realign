# NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning
![image](overview.png)

# 1. Fine-tuning Attack
### Safety Alignment Approaches
- [x] [SFT]
- [x] [DPO] 
- [x] [ORPO]
- [x] [KTO]
- [x] [SimPO] 


````
bash ./scripts/alignment/SFT/SFT.sh
bash ./scripts/alignment/ORPO/ORPO.sh
...
````


**Note**: Training config includes lora_rank=128, lora_alpha=256 

[//]: # (| Methods | Train dataset | Harmful Score ⬇ |)

[//]: # (|---------|---------------|-----------------|)

[//]: # (| SFT     | 2000          | 44.6            |)

[//]: # (| DPO     | 2000          | 26.2            |)

[//]: # (| ORPO    | 2000          | 34.5            |)

[//]: # (| KTO     | 2000          | 16.1            |)

[//]: # (| SimPO   | 2000          | 26.8            |)

### Harmful Fine-tuning Attack
- [x] harmful data ratio: clearn, 0.01, 0.05, 0.1, 0.2, 0.3

``
bash ./scripts/baseline/poison_rati/aligned_finetune.sh
``

### Datasets
- Alignment dataset: [PKU-Alignment](https://huggingface.co/PKU-Alignment)
- Downstream tasks

- [x] [SST2]
- [x] [AG_NEWS]
- [x] [GSM8K]


### Safety Evaluation
[beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)

[//]: # (### Fine-tuning Setting)

[//]: # (- [ ] harmful data ratio: clearn, 0.01, 0.05, 0.1, 0.2)

[//]: # (- [ ] harmful data size: 100, 500, 1000, 2000, 2500)


# 2. Safety Reference Model
For example, if aligned model is train based on DPO, then the safety reference model is DPO+EXPO.

Aligned models include:
- [x] [SFT]
- [x] [DPO]
- [x] [ORPO]
- [x] [KTO]
- [x] [SimPO]


``
bash ./scripts/strong_alignment/expo-sft_to_dpo-lora.sh
``

Note: It is based on [EXPO](https://arxiv.org/abs/2404.16792), which is a method to extrapolation of a alinged model to super-aligned model.

[//]: # (### Source Model)

[//]: # (- [X] [SFT])

[//]: # (- [X] [DPO])

[//]: # (- [X] [ORPO])

[//]: # (- [X] [KTO])

[//]: # (- [X] [SimPO])

[//]: # ()
[//]: # ()
[//]: # (| Expo Methods | alpha | Harmful Score |)

[//]: # (|:------------:|:-----:|:-------------:|)

[//]: # (| DPO          | -     |     26.2      |)

[//]: # (| DPO+EXPO     | 0.9   |      0.6      |)

[//]: # (| ORPO         | -     |     34.5      |)

[//]: # (| ORPO+EXPO    | 0.9   |     24.0      |)

[//]: # (| KTO          | -     |     16.1      |)

[//]: # (| KTO+EXPO     | 0.9   |      0.9      |)

[//]: # (| SimPO        | -     |     26.8      |)

[//]: # (| SimPO+EXPO   | 0.9   |      0.4      |)


# 3. Recognition of Safety-Critical Neurons
## [Neuron Level](https://arxiv.org/abs/2402.05162)
- [x] [SNIP score]
- [x] [Wanda score]
- [x] [Ours]

````
bash ./scripts/regions/build_preference_dataset.sh
bash ./scripts/regions/build_dataset.sh
bash ./scripts/regions/low_rank_sparity.sh
bash ./scripts/regions/low_rank_prune.sh
````

# 4. Restoration for a Safety-Broken Neurons

- [x] [Probability-based Layer Pruning]
- [x] [Neuron-Level Correction]
- [x] [Layer-Level Correction]

For example, if initially aligned model is train based on DPO, then the safety reference model is DPO+EXPO. 
Safety-broken neurons are recognized and restored:
````
bash ./scripts/ours/poison_ratio/expo-adaptive_mask_replace-eval_safety.sh
## eval downstream tasks
bash ./scripts/ours/poison_ratio/expo-adaptive_mask_replace-eval_downstream.sh
## eval safety
bash ./scripts/ours/poison_ratio/expo-adaptive_mask_replace-eval_safety.sh

````

# Acknowledgements

Our repository is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [alignment-attribution](https://github.com/boyiwei/alignment-attribution-code) and [Vaccine](https://github.com/git-disl/Vaccine). We thank the authors for their contributions.




