# Assessing the safety re-alignment via model fusion 
![image](overview.png)

# 1. Safety Alignment
### Training Approach
- [x] [SFT] (path: /home/yx/project_v2/saves/lora/sft)
- [x] [DPO]  (path: /home/yx/project_v2/saves/lora/dpo)
- [x] [ORPO]
- [x] [KTO]
- [x] [SimPO] 

**Note**: Training config includes lora_rank=128, lora_alpha=256 

| Methods | Train dataset | Harmful Score ⬇ |
|---------|---------------|-----------------|
| SFT     | 2000          | 44.4            |
| DPO     | 2000          | 26.0            |
| ORPO    | 2000          | 34.6            |
| KTO     | 2000          | 15.1            |
| SimPO   | 2000          | 26.8            |


### Datasets
[PKU-Alignment](https://huggingface.co/PKU-Alignment)

### Safety Evaluation
[beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)


# 2. Downstream Task Fine-tuning

### Datasets
- [ ] [SST2]
- [ ] [AG_NEWS]
- [ ] [GSM8K]
- [ ] [Alpaca]

### Fine-tuning setting
- [ ] harmful data ratio: clearn, 0.01, 0.05, 0.1, 0.2
- [ ] harmful data size: 100, 500, 1000, 2000, 2500

### Safety Evaluation


# 3. Stronger Alignment ([EXPO](https://arxiv.org/abs/2404.16792))

### Source Model
- [X] [SFT]
- [X] [DPO]
- [X] [ORPO]
- [X] [KTO]
- [X] [SimPO]


| Expo Methods | alpha | Harmful Score |
|:------------:|:-----:|:-------------:|
| DPO          | -     | 26.0          |
| DPO+EXPO     | 0.9   | 10.5          |
| ORPO         | -     | 34.6          |
| ORPO+EXPO    | 0.9   | 27.8          |
| KTO          | -     | 15.1          |
| KTO+EXPO     | 0.9   | 11.6          |
| SimPO        | -     | 26.8          |
| SimPO+EXPO   | 0.9   | 9.0           |

### Removing the least safety-critical ranks
- [ ] [SNIP score]
- [ ] [Wanda score]


# 4. Model Fusion
### Base Methods
- [ ] [Weight averaging]
- [ ] [Task Arithmetic]
- [ ] [TIES](http://arxiv.org/abs/2306.01708v2)
- [ ] [Consensus TA](https://arxiv.org/abs/2405.07813)
- [ ] [Consensus TIES](https://arxiv.org/abs/2405.07813)


### Fusion in the Safety Regions
- [ ] [Neuron Level](https://arxiv.org/abs/2402.05162)
- [ ] [Rank Level](https://arxiv.org/abs/2402.05162)
- [ ] [Attention Level](https://arxiv.org/abs/2406.01563)