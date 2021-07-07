import pandas as pd
import os
import datetime
from sklearn.model_selection import train_test_split


#read data
json_path = os.path.abspath(os.path.join(__file__, '..', 'json'))
info_df = pd.read_json(os.path.join(json_path, 'info.json'))
info_df = info_df.reset_index()

def hk_time(info_df):
    #solve the time problem within the dataset
    info_df['Releasing time'] = info_df['Releasing time'].apply(lambda d: (datetime.datetime.fromtimestamp(int(d)/1000)-datetime.timedelta(hours=8, minutes=0)))
    
    return info_df

hk_time(info_df)

#filter the past 14 days data
previous_14_days_df = info_df[(info_df['Releasing time']>pd.Timestamp(2020,10,11))]

#split the data into train and test data
train_df ,test_df = train_test_split(info_df, train_size=0.8, random_state=4011)

#save the file into json
train_df.to_json(os.path.join(json_path, 'train.json'))

test_df.to_json(os.path.join(json_path, 'test.json'))

previous_14_days_df.to_json(os.path.join(json_path, 'predict.json'))