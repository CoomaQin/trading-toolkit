import os, sys

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    HfArgumentParser,
)
from arguments import ModelArguments


def main():
    parser = HfArgumentParser((ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    if model_args.ptuning_checkpoint is not None:
        print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half().cuda()
        model.transformer.prefix_encoder.float().cuda()
    
    model = model.eval()
    cnts = [
        "Given a news: The maker of construction and mining equipment said Tuesday that its existing office in Irving, Texas, a suburb of Dallas, would serve as its new global headquarters. Caterpillar said that the move from its current base in suburban Chicago would help it grow and that the company wasn't getting any economic or tax incentives related to the headquarters move. Rate how it will influence company Tesla from very positive, positive, neutral, negative and very negative.",
        "Given a news: Electric-vehicle maker Tesla Inc. reported the most vehicle crashes suspected of involving advanced driver-assistance technology in the U.S. government's first-ever survey of such incidents. The auto industry's top safety regulator said Wednesday that it had received reports of nearly 400 recent crashes in which advanced driver-assistance features were engaged during or immediately before the incident. More than two-thirds of those crashes happened in a Tesla vehicle, it said. Caterpillar Inc. is decamping to Texas from its longtime Illinois base, joining other major U.S. companies weighing hiring and costs as they work to move past the continuing Covid-19 pandemic. Rate how it will influence company Tesla from very positive, positive, neutral, negative and very negative."
    ]
    tokens = tokenizer(cnts, return_tensors='pt', padding=True, max_length=128).to('cuda')   
    res = model.generate(**tokens, max_length=128)
    res_sentences = [tokenizer.decode(i) for i in res]
    print(res_sentences)


if __name__ == "__main__":
    main()