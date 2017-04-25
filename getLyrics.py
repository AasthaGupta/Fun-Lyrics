# -*- coding: utf-8 -*-
# @Author: Aastha Gupta
# @Date:   2017-04-18 01:43:41
# @Last Modified by:   Aastha Gupta
# @Last Modified time: 2017-04-21 13:05:44

from urllib.request import urlopen
import re
from bs4 import BeautifulSoup
import config

f = open(config.LYRICS_FILE,'w')

def saveLyrics(name, link):
	try:
		response = urlopen(link)
		html = response.read()
		response.close()
		soup = BeautifulSoup(html,"html.parser")
		data = soup.find("div", attrs={"id":"lyrics-body-text"})
		verses = data.find_all("p")
		lyrics = ""
		for verse in verses:
			lyrics = lyrics + verse.getText() + "\n"
		f.write(lyrics)
	except Exception as e:
		print("Couldn't get song",name)
		print(str(e))

def getData():

	url_template = config.URL[:-11] + "alpage-{}.html"
	page = 1
	songs_count = 0

	print ("Fetching lyrics...")
	while True:
		url = url_template.format(page)
		try:
			response = urlopen(url)
			html = response.read()
			response.close()
			soup = BeautifulSoup(html,"html.parser")

			songs_data = {}
			a_tags = soup.findAll("a", attrs={"class":"title hasvidtable"})
			for data in a_tags:
				link = data["href"]
				name = data.contents[0][1:-1].replace("Lyrics","")
				name = re.sub(r'[\\\/*?:"<>|]', "", name)
				songs_data[name]=link
			print("Fetched songs list from page {}".format(page))
			for key, value in songs_data.items():
				saveLyrics(key, value)
				songs_count += 1

			# to check if pages are left or not
			end = soup.findAll("a", attrs={"class":"button next disabled"})
			if end:
				raise ValueError("Fetched lyrics")

			page += 1
		except Exception as e:
			print("Total songs fetched:", songs_count)
			print( str(e) )
			f.close()
			break



if __name__ == "__main__":
    getData()