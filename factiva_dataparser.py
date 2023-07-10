import PyPDF2
import re
import json
from tqdm import tqdm
import datetime
import os

import datasets
import transformers


class FactivaDataloader:
    def __init__(self, max_seq_length=2000, skip_overlength=False, model_name="THUDM/chatglm-6b"):
        self.max_seq_length = max_seq_length
        self.skip_overlength = skip_overlength
        self.keys = {"BY", "WC", "PD", "ET", "SN", "SC", "LA", "LP", "CO", "IN", "NS", "RE", "IPD", "IPC", "PUB", "AN"}

        self.config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        

    # parse factiva export pdf into dict
    def extract_text_from_pdf(self, pdf_path, sp_path):
        stock_price = []
        period_days = 30
        with open(sp_path) as f:
            stock_price = json.load(f)

        date_target_mapping = {}
        for i in range(period_days, len(stock_price)):
            start_item = stock_price[i]
            end_item = stock_price[i-period_days]
            target = int((end_item["open"] - start_item["close"]) / start_item["close"] * 100)
            date_target_mapping[start_item] = target

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        time_re = re.compile(r'^(([01]\d|2[0-3]):([0-5]\d)|24:00)$')
        wc_re = re.compile(r'\d+(?:,\d+)? words')
        cnt_list = text.split("\nHD")
        data = []
        for cnt in cnt_list[1:]:
            tmp = {}
            rows = cnt.split("\n")
            key_to_add = "HD"
            val_to_add = ""
            for row in rows[1:]:
                key = row.split(" ")[0]
                if key in self.keys:
                    tmp[key_to_add] = val_to_add
                    key_to_add = key
                    val_to_add = row[(len(key) + 1):] # skip the space 
                else:
                    val_to_add = val_to_add + " " + row
            try:
                target_date = datetime.datetime.strptime(tmp["PD"], "%d %B %Y").date() + datetime.timedelta(days=period_days)
            except ValueError:
                print(tmp["PD"])
                continue
            if target_date.weekday == 5:
                target_date += datetime.timedelta(days=2)
            elif target_date.weekday == 6:
                target_date += datetime.timedelta(days=1)
            key = target_date.strftime("%Y-%m-%d")
            if key not in date_target_mapping:
                tmp["target"] = "pending"
            else:
                tmp["target"] = date_target_mapping[key]
            data.append(tmp)
        return data

    
    def preprocess(self, example):
        prompt = example["LP"]
        target = example["target"]
        prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_seq_length, truncation=True)
        target_ids = self.tokenizer.encode(target, max_length=self.max_seq_length, truncation=True, add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [self.config.eos_token_id]
        return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

    
    # construct dataset generator
    def read_jsonl(self, data):
        for item in data:
            if "LP" not in item:
                print(item)
                continue
            feature = self.preprocess(item)
            if self.skip_overlength and len(feature["input_ids"]) > self.max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:self.max_seq_length]
            yield feature

    
    def convert_to_dataset(self, pdf_folder_path, stock_path, save_path):
        pdf_paths = []
        for (_, _, filenames) in os.walk(pdf_folder_path):
            for f in filenames:
                if f.split(".")[-1] == "pdf":
                   pdf_paths.append(os.path.join(pdf_folder_path, f))
            break
        data = []
        for p in tqdm(pdf_paths):
            res = self.extract_text_from_pdf(p, stock_path)
            data.extend(res)
        print(f"{len(data)} news collected")
        dataset = datasets.Dataset.from_generator(lambda: self.read_jsonl(data))
        dataset.save_to_disk(save_path)
    

if __name__ == "__main__":
    fd = FactivaDataloader(max_seq_length=2000, skip_overlength=False, model_name="THUDM/chatglm-6b")
    fd.convert_to_dataset("./pdf/", "tesla.json", "./tesla_2022_2023")
