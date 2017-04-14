from bs4 import BeautifulSoup
import requests

url = "www.goodreads.com/work/quotes/3078186-the-hitch-hiker-s-guide-to-the-galaxy"

r = requests.get("https://" + url)

data = r.text
soup = BeautifulSoup(data)

#get the quote elements
quote_divs = soup.findAll("div", {"class" : "quoteText"})

#remove the scripts and links
junk = [s.extract() for s in soup(['script','br', 'a', 'span'])]

#only the quote content remains
#store it in file
quote_file = open('./text/hitch_hiker_quotes.txt', 'w')

for quote in quote_divs:
	print(quote.text)
	quote_file.write(quote.text)