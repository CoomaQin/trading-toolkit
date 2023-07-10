import json
import requests
import pandas as pd


class FinancialModelingDataloader:
    def __init__(self):
        self.api_key = "c0126a6544ceb401fec5ef4b97dcc5d7"


    def get_stock_price(self, stock, save_path=""):
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?apikey={self.api_key}"
        res = requests.get(url)
        res = json.loads(res.text)
        if save_path:
            with open(save_path, "w") as outfile:
                json.dump(res, outfile, indent=4)


if __name__ == "__main__":
    fd = FinancialModelingDataloader()
    fd.get_stock_price("TSLA", save_path="tesla.json")