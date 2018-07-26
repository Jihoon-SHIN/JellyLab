# -*- coding: utf-8 -*-
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import re
from time import sleep

def get_html(url):
	_html = ""
	resp = requests.get(url)
	if (resp.status_code == 200):
		_html = resp.text
	return _html


# Increase the Date
def increaseData(date):
	date_list = list(date)
	month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
	_year = date_list[0:4]
	_month = date_list[4:6]
	_day = date_list[6:8]
	string_year = ''.join(_year)
	string_month = ''.join(_month)
	string_day = ''.join(_day)
	int_year = int(string_year)
	int_month = int(string_month)
	int_day = int(string_day)
	if(_month[0]=='1'):
		if(_month[1]=='1'):
			if(int_day==30):
				string_month='12'
				string_day='01'
			else:
				int_day = int_day+1
				if(int_day<10):
					string_day = "0" + str(int_day)
				else:
					string_day = str(int_day)
		elif(_month[1]=='0'):
			if(int_day==30):
				string_month ="11"
				string_day ="01"
			else:
				int_day = int_day+1
				if(int_day<10):
					string_day = "0" + str(int_day)
				else:
					string_day = str(int_day)
		else:
			if(int_day==31):
				int_year = int_year +1
				string_year = str(int_year)
				string_month ="01"
				string_day ="01"
			else:
				int_day = int_day+1
				if(int_day<10):
					string_day = "0" + str(int_day)
				else:
					string_day = str(int_day)
	else:
		thirty_one = ((_month[1]=='1') or (_month[1]=='3') or (_month[1]=='5') or (_month[1]=='7') or (_month[1]=='8') or (_month[1]=='10') or (_month[1]=='12'))
		thirty = ((_month[1]=='4') or (_month[1]=='6') or (_month[1]=='9') or (_month[1]=='11'))
		if(thirty_one):
			if(_month[1]=='7'):
				if(int_year==2018 and int_day==25):
					return 'end'
			if(int_day==31):
				int_month = int_month +1
				if(int_month<10):
					string_month = "0" + str(int_month)
				else:
					string_month = str(int_month)
				string_day ="01"
			else:
				int_day = int_day+1
				if(int_day<10):
					string_day = "0" + str(int_day)
				else:
					string_day = str(int_day)
		elif(thirty):
			if(int_day==30):
				int_month = int_month +1
				if(int_month<10):
					string_month = "0" + str(int_month)
				else:
					string_month = str(int_month)
				string_day ="01"
			else:
				int_day = int_day+1
				if(int_day<10):
					string_day = "0" + str(int_day)
				else:
					string_day = str(int_day)
		else:
			if(int_day==28):
				int_month = int_month +1
				if(int_month<10):
					string_month = "0" + str(int_month)
				else:
					string_month = str(int_month)
				string_day ="01"
			else:
				int_day = int_day+1
				if(int_day<10):
					string_day = "0" + str(int_day)
				else:
					string_day = str(int_day)

	return string_year+string_month+string_day


	


base_url_news = ["37?"]
base_url_page = "page=" #여기에다가 pagenumber& 붙여야함
base_url_date = "&regDate=" #ex 20180718
class daumCrawl(object):
	def __init__(self, baseurl, result):
		# Set the url
		self.baseurl = baseurl
		self.result = result

	# def test(self):
	# 	html = get_html(self.url)
	# 	soup = BeautifulSoup(html, 'html.parser')
	# 	page_number = soup.findAll("a", {"class":"num_page"})[-1].text #Max page number in 1,11,21
	# 	link = soup.findAll("a",{"class":"link_thumb"})
	# 	link = link[0:len(link)-1] # link to article

	# 	# print(link)
	# 	# print(len(html))

	def crawl_data(self):
		result = open(self.result, 'a')
		for i in range(len(base_url_news)):
			# if i ==0 :
			# 	date = "20170302"
			# else:
			date = "20111105"
			print(i, base_url_news[i])
			while(True):
				base_num = 1
				flag = 0
				url = self.baseurl+base_url_news[i]+base_url_page+str(base_num)+base_url_date+date
				print(url)
				html = get_html(url)
				soup = BeautifulSoup(html, 'html.parser')
				current_page = soup.findAll("em", {"class":"num_page"})#Max page number in 1,11,21
				index = str(current_page).find("</span>")

				if index == -1:
					flag = 1

				if flag==0:
					current_page = str(current_page)[index+8]
					if(int(current_page)==1):
						num_list = soup.findAll("a", {"class":"num_page"})
						if(len(num_list)==0):
							page_number = 1
						else:
							page_number = num_list[-1].text #Max page number in 1,11,21
					link = soup.findAll("a",{"class":"link_thumb"})
					link = link[0:len(link)-1] # link to article
					for j in range(1,int(page_number)+1):
						url = self.baseurl+base_url_news[i]+base_url_page+str(j)+base_url_date+date
						print(url)
						html = get_html(url)
						soup = BeautifulSoup(html, 'html.parser')
						link = soup.findAll("a",{"class":"link_thumb"})
						link = link[0:len(link)-2] # link to article
						for _link in link:
							index_start = str(_link).find("href=")+6
							index_end = str(_link).find("<img")-3
							_link = str(_link)[index_start:index_end]
							# print(_link)
							detail_html = get_html(_link)
							detail_soup = BeautifulSoup(detail_html, 'html.parser')
							soup_list = detail_soup.select('div#mArticle  div#harmonyContainer')
							if len(soup_list) == 0:
								continue
							else:
								title = detail_soup.select('div#mArticle  div#harmonyContainer')[0]	
							result_string = title.get_text().strip() 
							result.write(result_string)
				sleep(2)
				date = increaseData(date)
				if(date=='end'):
					break




result_file = "2010_2018.txt"
base_url = "http://media.daum.net/cp/"

crawl_test = daumCrawl(base_url, result_file)
crawl_test.crawl_data()



# crawl_test1 = daumCrawl(url, result_file)
# crawl_test1.test()