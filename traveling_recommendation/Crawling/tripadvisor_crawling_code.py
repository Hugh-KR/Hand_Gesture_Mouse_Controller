from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
result=[]
driver = webdriver.Chrome()
url="https://www.tripadvisor.co.kr/Attraction_Review-g294197-d324888-Reviews-Gyeongbokgung_Palace-Seoul.html"
driver.get(url)
for i in range(1):
    print("{}번째 page".format(i+1))
    for span in driver.find_elements_by_class_name("_2tsgCuqy"):
        print(span.text)
        result.append(span.text)
    time.sleep(1)
    #driver.find_element_by_xpath('//*[@id="tab-data-qa-reviews-0"]/div/div[5]/div[11]/div[1]/div/div[1]/div[2]/div/a').click()#다음페이지 경로설정
    num=((i+1)*10)
    url="https://www.tripadvisor.co.kr/Attraction_Review-g294197-d324888-Reviews-or"+str(num)+"-Gyeongbokgung_Palace-Seoul.html"
    driver.get(url)
    time.sleep(6)#6초 휴식
    print("-----------------------------------------------------------------------")

result_df = pd.DataFrame(result)


result_df.to_csv("practice2.csv",index=False,encoding = 'utf-8-sig')
#df1=pd.read_csv("result_data.csv")
#df2=pd.read_csv("practice2.csv")
#result_data=pd.concat([df1,df2],axis=1)
#result_data.to_csv("result_data.csv",index=False,encoding = 'utf-8-sig')