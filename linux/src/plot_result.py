import pandas as pd 
import os
import datetime
import matplotlib.pyplot as plt

json_path = os.path.abspath(os.path.join(__file__, '..', 'json'))
pic_path = os.path.abspath(os.path.join(__file__, '..', 'pic'))

positive = pd.read_json(os.path.join(json_path, 'positive.json'))
negative = pd.read_json(os.path.join(json_path, 'negative.json'))
neutral = pd.read_json(os.path.join(json_path, 'neutral.json'))
date = pd.read_json(os.path.join(json_path, 'date.json'))
date['date_x'] = date['date_x'].apply(lambda d: (datetime.datetime.fromtimestamp(int(d)/1000)-datetime.timedelta(hours=8, minutes=0)))
result_df = pd.concat([date, positive, negative, neutral],axis=1)
result_df.columns = ['date', 'positive', 'negative', 'neutral']
result_df = result_df.sort_values(by=['date'])

#plot 
plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.style.use('ggplot')
plt.plot(result_df['date'], result_df['positive'], linewidth=0.5)
plt.plot(result_df['date'], result_df['negative'], linewidth=0.5)
plt.plot(result_df['date'], result_df['neutral'], linewidth=0.5)

plt.legend(['positive', 'negative', 'neutral'], loc='upper left')
plt.savefig(os.path.join(pic_path, 'overall.png'), dpi = 500)