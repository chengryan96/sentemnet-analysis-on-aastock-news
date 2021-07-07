import time
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import itertools
import re
from natsort import natsorted, ns
import pandas as pd

num_range = [x for x in range(1040000, 1051081)]
news_no = ['Now.'+ str(num) for num in num_range]
url_list = [r'http://www.aastocks.com/en/stocks/news/aafn-con/' + news_id + r'/company-news' for news_id in news_no]
url_df = pd.DataFrame(url_list)
url_df.to_json(r'C:\Users\cheng\OneDrive\Desktop\ey_test_code\json\all_url.json')