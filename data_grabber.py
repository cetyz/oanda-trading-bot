# just a file to get historical data and save in CSV
# will create a data folder and save it in there

from datetime import datetime
import json
import csv
from time import sleep
from api_wrapper import Oanda

# select instrument (you can go onto the oanda app and check for the full list (replace "/" with "_"))
# you can leave count at 5000 so it's the fastest
# feel free to change granularity of candles (refer to api_wrapper.py script for possible granularities)
# select data start date
# select data end date

# just run and it will pull the data for you

instrument = 'ETH_USD'
count = 5000
granularity = 'S5'
# start_date = '2010-01-05T00:00:00' # you can go into seconds as well
start_date = '2019-01-01'
end_date = '2021-12-31'

csv_filename = f'{instrument}_data_{start_date}_to_{end_date}.csv'

with open('config.json', 'r') as f:
    configs = json.load(f)
    token = configs['token']
    account = configs['account']
    user = configs['user']
    
oanda = Oanda(token=token, account=account, user=user, time_format='RFC3339')

latest_date = datetime.fromisoformat(start_date)

end_date_f = datetime.fromisoformat(end_date)



with open(csv_filename, 'w') as f:

    writer = csv.writer(f)

    while latest_date < end_date_f:

        if not latest_date:

            candles = oanda.get_candle(instrument=instrument,
                                    count=count,
                                    granularity=granularity,
                                    start=start_date)
        
        else:

            candles = oanda.get_candle(instrument=instrument,
                                    count=count,
                                    granularity=granularity,
                                    start=latest_date.isoformat())
                            
        for candle in candles['candles']:
            trunc_time = candle['time'][:-11]
            time = datetime.fromisoformat(trunc_time).isoformat()
            volume = candle['volume']
            o = float(candle['mid']['o'])
            h = float(candle['mid']['h'])
            l = float(candle['mid']['l'])
            c = float(candle['mid']['c'])
            values = [time, volume, o, h, l, c]
            writer.writerow(values)
            latest_date = datetime.fromisoformat(time)

        print(f'Data pulled up to {latest_date}')
        sleep(1)
        