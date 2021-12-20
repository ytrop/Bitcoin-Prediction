# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:24:25 2021

@author: ytrop
"""

from websocket import create_connection
import json


test_key = '3623AB15-E80B-4FDA-A3A4-704E7E5A8AD3'


class CoinAPIv1_subscribe(object):
  def __init__(self, apikey):
    self.type = "hello"
    self.apikey = test_key
    self.heartbeat = True
    self.subscribe_data_type = ["trade"]

ws = create_connection("ws://ws.coinapi.io/v1/")
sub = CoinAPIv1_subscribe(test_key)
ws.send(json.dumps(sub.__dict__))
while True:
  msg =  ws.recv()
  print(msg)
  
ws.close()