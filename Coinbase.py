# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 08:50:09 2021

@author: ytrop
"""

import requests
import pandas as pd
import json
from pandas import json_normalize
import seaborn as sns
from matplotlib import pyplot as plt

url = "https://rest.coinapi.io/v1/ohlcv/BTC/USD/history?period_id=5MIN&time_start=2021-12-04T00:00:00&time_end=2021-12-07T00:00:00&limit=100000"
headers = {'X-CoinAPI-Key' : '3623AB15-E80B-4FDA-A3A4-704E7E5A8AD3'}


def get_report(url):
    print("Obteniendo datos...")
    response = requests.get(url, headers=headers)
    if response.ok:
        print("Datos recibidos...")
        print("HTTP %i - %s" % (response.status_code, response.reason))
        return response.text
    else:
        print("HTTP %i - %s" % (response.status_code, response.reason))

def export_to_json(url):
    print("Exportando a Json...")
    response = get_report(url)
    text_file = open("datosCriptoOHLCV", "w", encoding="utf8")
    text_file.write(response)
    text_file.close()

export_to_json(url)    


