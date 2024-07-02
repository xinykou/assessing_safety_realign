from datasets import load_dataset
import random
import torch


# Load and process aligned dataset
def get_align(nsamples, seed, seqlen, tokenizer, disentangle=False, data_path=""):
    # Load train and test datasets
    data_files = {"train": data_path}
    traindata = load_dataset("json", data_files=data_files, split="train")
    trainloader = []
    random.seed(seed)
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    if disentangle:
        if nsamples < len(traindata):
            traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        else:
            traindata_sampled = traindata
        for i in range(nsamples):
            prompt = template.format(instruction=traindata_sampled["input"][i])
            trainenc_prompt = tokenizer(prompt, return_tensors="pt")
            trainenc_response = tokenizer(traindata_sampled["output"][i], return_tensors="pt")
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")

        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None