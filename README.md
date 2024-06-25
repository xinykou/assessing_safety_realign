# Safety Realignment of Large Language Models via Restoration and Augmentation of Safety Neruons
![image](overview.png)

# 1. Fine-tuning Attack
### Safety Alignment Approach
- [x] [SFT] (path: /home/yx/project_v2/saves/lora/sft)
- [x] [DPO]  (path: /home/yx/project_v2/saves/lora/dpo)
- [x] [ORPO]
- [x] [KTO]
- [x] [SimPO] 

**Note**: Training config includes lora_rank=128, lora_alpha=256 

| Methods | Train dataset | Harmful Score ⬇ |
|---------|---------------|-----------------|
| SFT     | 2000          | 44.6            |
| DPO     | 2000          | 26.2            |
| ORPO    | 2000          | 34.5            |
| KTO     | 2000          | 16.1            |
| SimPO   | 2000          | 26.8            |


### Datasets
- Alignment dataset: [PKU-Alignment](https://huggingface.co/PKU-Alignment)
- Downstream task

- [ ] [SST2]
- [ ] [AG_NEWS]
- [ ] [GSM8K]
- [ ] [Alpaca]

### Safety Evaluation
[beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)

### Fine-tuning Setting
- [ ] harmful data ratio: clearn, 0.01, 0.05, 0.1, 0.2
- [ ] harmful data size: 100, 500, 1000, 2000, 2500

### Fine-tuning Attack

# 2. Amplification Safety ([EXPO](https://arxiv.org/abs/2404.16792))

### Source Model
- [X] [SFT]
- [X] [DPO]
- [X] [ORPO]
- [X] [KTO]
- [X] [SimPO]


| Expo Methods | alpha | Harmful Score |
|:------------:|:-----:|:-------------:|
| DPO          | -     |     26.2      |
| DPO+EXPO     | 0.9   |      0.6      |
| ORPO         | -     |     34.5      |
| ORPO+EXPO    | 0.9   |     24.0      |
| KTO          | -     |     16.1      |
| KTO+EXPO     | 0.9   |      0.9      |
| SimPO        | -     |     26.8      |
| SimPO+EXPO   | 0.9   |      0.4      |


# 3. Identifying safety-critical neroens
## [Neuron Level](https://arxiv.org/abs/2402.05162)
- [ ] [SNIP score]
- [ ] [Wanda score]
## [Rank Level](https://arxiv.org/abs/2402.05162)


# 4. Model Fusion

[//]: # (### Base Methods)

[//]: # (- [ ] [Weight averaging])

[//]: # (- [ ] [Task Arithmetic])

[//]: # (- [ ] [TIES]&#40;http://arxiv.org/abs/2306.01708v2&#41;)

[//]: # (- [ ] [Consensus TA]&#40;https://arxiv.org/abs/2405.07813&#41;)

[//]: # (- [ ] [Consensus TIES]&#40;https://arxiv.org/abs/2405.07813&#41;)


### Fusion in the Safety Regions

[//]: # (- [ ] [Attention Level]&#40;https://arxiv.org/abs/2406.01563&#41;)