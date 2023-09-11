from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import os


model = "THUDM/chatglm2-6b"
# local_model_weight = "./output/"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
config = AutoConfig.from_pretrained(model, trust_remote_code=True) 
config.pre_seq_len = 2000 
model = AutoModel.from_pretrained(model, load_in_8bit=True, trust_remote_code=True, device_map="auto", config=config)

prefix_state_dict = torch.load(os.path.join("./teslagpt", "adapter_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict, strict=False)

model = model.cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

# Make prompts
prompt = [
"Given a news: Electric-vehicle maker Tesla Inc. reported the most vehicle crashes suspected of involving advanced driver-assistance technology in the U.S. government's first-ever survey of such incidents. The auto industry's top safety regulator said Wednesday that it had received reports of nearly 400 recent crashes in which advanced driver-assistance features were engaged during or immediately before the incident. More than two-thirds of those crashes happened in a Tesla vehicle, it said. Caterpillar Inc. is decamping to Texas from its longtime Illinois base, joining other major U.S. companies weighing hiring and costs as they work to move past the continuing Covid-19 pandemic. The maker of construction and mining equipment said Tuesday that its existing office in Irving, Texas, a suburb of Dallas, would serve as its new global headquarters. Caterpillar said that the move from its current base in suburban Chicago would help it grow and that the company wasn't getting any economic or tax incentives related to the headquarters move. Rate how it will influence company Tesla from very positive, positive, neutral, negative and very negative."
]

# Generate results
tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=500).to('cuda')
res = model.generate(**tokens, max_length=500)
res_sentences = [tokenizer.decode(i) for i in res]
print(res_sentences)