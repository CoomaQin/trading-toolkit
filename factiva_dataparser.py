import PyPDF2
import re
import json
from tqdm import tqdm
import datetime
import os
from collections import defaultdict

import datasets
import transformers


class FactivaDataloader:
    def __init__(self, max_seq_length=2000, skip_overlength=False, model_name="THUDM/chatglm-6b"):
        self.max_seq_length = max_seq_length
        self.skip_overlength = skip_overlength
        self.keys = {"BY", "HD", "WC", "PD", "ET", "SN", "SC", "LA", "LP", "CO", "IN", "NS", "RE", "IPD", "IPC", "PUB", "AN"}

        self.config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    # define the target based on stock prices and dates
    def extract_date_target_mapping(self, sp_path, period_days):
        stock_price = []
        with open(sp_path) as f:
            stock_price = json.load(f)['historical']
        start_date, end_date = datetime.datetime.strptime(stock_price[-1]['date'], "%Y-%m-%d").date(), datetime.datetime.strptime(stock_price[0]['date'], "%Y-%m-%d").date()
        days_range = (end_date - start_date).days
        date_sp_mapping = {}
        for item in stock_price:
            date_sp_mapping[item['date']] = item["open"] # use open price because the sentiment has already make effect on the price during that day 
        # fill up weekend and holiday price
        start_date_sp_mapping, end_date_sp_mapping = {}, {}
        for i in range(1, days_range):
            key = (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            curr_sdate, curr_edate = start_date + datetime.timedelta(days=i), start_date + datetime.timedelta(days=i)
            while curr_sdate.strftime("%Y-%m-%d") not in date_sp_mapping:
                curr_sdate -= datetime.timedelta(days=1)
            while curr_edate.strftime("%Y-%m-%d") not in date_sp_mapping:
                curr_edate += datetime.timedelta(days=1)
            start_date_sp_mapping[key] = date_sp_mapping[curr_sdate.strftime("%Y-%m-%d")]
            end_date_sp_mapping[key] = date_sp_mapping[curr_edate.strftime("%Y-%m-%d")]
        # define the target as the level of diff between sentiment_date and sentiment_date + period_days
        date_target_mapping = {}
        for i in range(1, days_range - period_days):
            # weekend start_time round to last friday
            curr_start_date = (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            curr_end_date = (start_date + datetime.timedelta(days=(i+period_days))).strftime("%Y-%m-%d")
            target = int((end_date_sp_mapping[curr_end_date] - start_date_sp_mapping[curr_start_date]) / start_date_sp_mapping[curr_start_date] * 100)
            tag = "unknown"
            if target > 5: 
                tag = "very positive"
            elif target > 2: 
                tag = "positive"
            elif target > -2:
                tag = "neutral"
            elif target > -5:
                tag = "negative"
            else:
                tag = "very negative"
            date_target_mapping[curr_start_date] = tag
        return date_target_mapping


    # parse factiva export pdf into dict
    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
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
            data.append(tmp)
        return data

    
    def preprocess(self, example, date, company, stock_path, period_days=30):
        # get target
        date_target_mapping = self.extract_date_target_mapping(stock_path, period_days)
        target_date = datetime.datetime.strptime(date, "%d %B %Y").date() + datetime.timedelta(days=period_days)
        if target_date.weekday() == 5:
            target_date += datetime.timedelta(days=2)
        elif target_date.weekday() == 6:
            target_date += datetime.timedelta(days=1)
        key = target_date.strftime("%Y-%m-%d")
        target = date_target_mapping[key]

        # get prompt
        prompt = f"Given a news: {example}. Rate how it will influence company {company} from very positive, positive, neutral, negative and very negative."
        # prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_seq_length, truncation=True)
        # target_ids = self.tokenizer.encode(target, max_length=self.max_seq_length, truncation=True, add_special_tokens=False)
        # input_ids = prompt_ids + target_ids + [self.config.eos_token_id]
        # res = {"input_ids": input_ids}
        res = {"content": prompt, "summary": target}
        return res

    
    # construct dataset generator
    def read_jsonl(self, data, company, sp_path):
        for item in tqdm(data):
            feature = self.preprocess(data[item], item, company, sp_path)
            # if self.skip_overlength and len(feature["input_ids"]) > self.max_seq_length:
            #     continue
            # feature["input_ids"] = feature["input_ids"][:self.max_seq_length]
            yield feature

    
    def convert_to_dataset(self, pdf_folder_path, stock_path, save_path, company):
        pdf_paths = []
        for (_, _, filenames) in os.walk(pdf_folder_path):
            for f in filenames:
                if f.split(".")[-1] == "pdf":
                   pdf_paths.append(os.path.join(pdf_folder_path, f))
        news_data = []
        for p in tqdm(pdf_paths):
            res = self.extract_text_from_pdf(p)
            news_data.extend(res)
        # reduce news to daily sentiment
        sentiment_data = defaultdict(str)
        for news in news_data:
            date = news["PD"]
            if "LP" not in news:
                continue
            leading_parah = news["LP"]
            if len(sentiment_data[date]) + len(leading_parah) < self.max_seq_length: # limit max_seq_length
                sentiment_data[date] += leading_parah
        print(f"{len(sentiment_data.keys())} news collected")
        # print(self.extract_date_target_mapping(stock_path, 30))
        dataset = datasets.Dataset.from_generator(lambda: self.read_jsonl(sentiment_data, company, stock_path))
        dataset.save_to_disk(save_path)

    def save_as_json(self, dataset_path, train_save_path, val_save_path):
        dataset = datasets.load_from_disk(dataset_path)
        train_val = dataset.train_test_split( test_size = 0.1, shuffle=True, seed=42 )
        train_data = train_val["train"].shuffle()
        val_data = train_val["test"].shuffle()
        train_data.to_json(train_save_path) 
        val_data.to_json(val_save_path) 
    

if __name__ == "__main__":
    max_seq_length = 500
    fd = FactivaDataloader(max_seq_length=max_seq_length, skip_overlength=False, model_name="THUDM/chatglm-6b")
    fd.convert_to_dataset("./data/pdf/", "./data/tesla.json", f"./data/tesla_22to23_{max_seq_length}", "Tesla")
    fd.save_as_json("./data/tesla_22to23_500", "./data/tesla_22to23_500_train.json", "./data/tesla_22to23_500_val.json",)
