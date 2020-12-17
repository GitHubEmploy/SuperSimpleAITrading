
import alpaca_trade_api as tradeapi
import coloredlogs, logging
global logger
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
# get s&p500
import pandas as pd
table=pd.read_html('https://www.barchart.com/stocks/sectors/penny-stocks?viewName=main')
df = table[0]
sp = df['Symbol']

logger.info(sp)

import os, os.path

from PYkeys import Keys
keys = Keys()
api = tradeapi.REST(key_id = keys.get_key_id(),
                    secret_key = keys.get_secret_key(),
                    base_url = keys.get_base_url())

barset = []
for symbol in range(503, 504):
  barset.append(api.get_barset(sp[symbol],'5Min',limit=1000))
  logger.info(sp[symbol])

logger.info(barset[0])

Dataset = open(r"newDataset.csv",'w')

Dataset.write('Open, Close, High, Low, Volume\n')

for symbolNum in range(503, 504):
  symbol = sp[symbolNum]
  symbol_bars = barset[0][symbol] # 0 = symbolNum
  for barNum in symbol_bars:
  #test = symbol
    #logger.info(barNum)
    Dataset.write(str(barNum.t) + ',')
    Dataset.write(str(barNum.o) + ',')
    Dataset.write(str(barNum.c) + ',')
    Dataset.write(str(barNum.h) + ',')
    Dataset.write(str(barNum.l) + ',')
    Dataset.write(str(barNum.v) + '\n')

Dataset.close()

