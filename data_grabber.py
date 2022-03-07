# just a file to get historical data and save in CSV
# will create a data folder and save it in there

from datetime import datetime
import json
import csv
from api_wrapper import Oanda

instrument = 'USD_JPY'
count = 10
granularity = 'S5'
start_date = '2010-01-05T00:00:00'
end_date = ''

# dt = datetime.fromisoformat(start_date)
# print(dt)

with open('config.json', 'r') as f:
    configs = json.load(f)
    token = configs['token']
    account = configs['account']
    user = configs['user']
    
oanda = Oanda(token=token, account=account, user=user, time_format='RFC3339')

candles = oanda.get_candle(instrument=instrument,
                           count=count,
                           granularity=granularity,
                           start=start_date)

for candle in candles['candles']:
    print(candle)