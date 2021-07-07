from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

url_df = pd.read_json(r'C:\Users\cheng\OneDrive\Desktop\ey_test_code\json\all_url.json')[0]
options = webdriver.ChromeOptions()
#options.add_argument("--headless")
driver = webdriver.Chrome(executable_path=r"C:\Users\cheng\OneDrive\Desktop\ey_test_code\drivers\chromedriver.exe", options = options)


def scrap_detail(url):
    driver.implicitly_wait(5)
    response = driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    if soup.find_all("p") == []:
        pass

    time = str(soup.find_all('div', class_='float_l newstime5'))
    time = re.findall(r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}', time)[0]
    time = datetime.strptime(time, '%Y/%m/%d %H:%M')

    header = soup.find_all('div', class_='newshead5')[0].get_text()
    header = header.replace('\n', '')
    header = header.strip()

    abstract_list = soup.find_all("p")
    abstract = []
    for a in abstract_list:
        abstract.append(a.text)
    abstract = abstract[0]
    abstract = abstract.replace('\xa0', '')

    company_name_code = re.findall(r'([A-Z\s]+)\((\d{5}.HK)\)', abstract)
    company_name_code = list(set(company_name_code))

    company_name = [name_code[0] for name_code in company_name_code]
    company_name = [name.strip() for name in company_name]
    stock_code = [name_code[1] for name_code in company_name_code]

    if company_name == [] or stock_code == []:
        company_name_code = re.findall(r'([A-Z\s]+)\((\d{5}.HK)\)', header)
        company_name = [name_code[0] for name_code in company_name_code]
        company_name = [name.strip() for name in company_name]
        stock_code = [name_code[1] for name_code in company_name_code]

    recommand = soup.find_all('div', class_='divRecommend')[0].get_text()
    recommand = int(re.findall(r'\d+', recommand)[0])
    bullish = soup.find_all('div', class_='divBullish')[0].get_text()
    bullish = int(re.findall(r'\d+', bullish)[0])
    bearish = soup.find_all('div', class_='divBearish last')[0].get_text()
    bearish = int(re.findall(r'\d+', bearish)[0])


    info_dict_list = []
    #save the result as dictionary
    if len(company_name) > 1:
        for i in range(0, len(company_name)):
            info_dict = {'headline':header, 'Releasing time':time, 'Company name':company_name[i], 'Stock Code':stock_code[i], 'Abstract':abstract, 'positive':bullish, 'neutral':recommand, 'negative':bearish}
            info_dict_list.append(info_dict)
    else:
        info_dict = {'headline':header, 'Releasing time':time, 'Company name':company_name[0], 'Stock Code':stock_code[0], 'Abstract':abstract, 'positive':bullish, 'neutral':recommand, 'negative':bearish}
    info_dict_list.append(info_dict)

    info_df = pd.DataFrame(info_dict_list)

    return info_df

info_df = pd.DataFrame(columns = ['headline', 'Releasing time', 'Company name', 'Stock Code', 'Abstract', 'positive', 'neutral', 'negative'])


for i in range(0,len(url_df)):
    try:
        info = scrap_detail(url_df[i])
        print(i)
        info_df = info_df.append(info, ignore_index=True)
    except:
        pass


info_df = info_df.drop_duplicates()

info_df.to_json(r'C:\Users\cheng\OneDrive\Desktop\ey_test_code\json\info.json')

