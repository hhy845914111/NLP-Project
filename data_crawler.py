import requests as rq
import bs4
import re
from tqdm import tqdm
from time import sleep
from random import random

PREFIX = "https://www.federalreserve.gov"

YEAR_LST2 = [f"{PREFIX}/newsevents/speech/{i}-speeches.htm" for i in range(2011, 2020)]
YEAR_LST1 = [f"{PREFIX}/newsevents/speech/{i}speech.htm" for i in range(2006, 2011)]


def get_all(verbose=True):
    rst_lst = []

    itr = tqdm(YEAR_LST1) if verbose else YEAR_LST1
    for this_year in YEAR_LST1 + YEAR_LST2:
        response = rq.get(this_year)
        sp_obj = bs4.BeautifulSoup(response.text, "html.parser")
        all_article_obj_lst = sp_obj.find_all("a", {"href": re.compile("^/newsevents/speech/")})

        for this_article in all_article_obj_lst:
            sleep(random())

            this_href = this_article["href"]
            this_title = this_article.text
            this_date = re.findall("\d+", this_href)[0]
            this_content = rq.get(f"{PREFIX}{this_href}")

            print(this_date, this_title)

            rst_lst.append((this_title, this_date, parse_one_article(this_content)))

    return rst_lst


def parse_one_article(response_obj):
    #TODO: implement
    pass


if __name__ == "__main__":
    rst = get_all()
    print(1)
