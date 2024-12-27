import json
from bs4 import BeautifulSoup
from naotool.httpn import AutoCloseAsyncClient
import asyncio
import cloudscraper

sc = cloudscraper.create_scraper()


# 最终目标，提取全文本

with open("tmp.json", "r") as f:
    pages = json.load(fp=f)

async def get_a_article(url: str):
    # 访问网站
    await asyncio.sleep(3)
    html = sc.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    text = "\n".join((s.strip() for s in text.split("\n") if s.strip()))
    print(text)
    pages.append({"text":text, "source": url})
    with open("tmp.json", "w") as f:
        json.dump(pages, f)

async def main():
    next_url = "https://support.strikingly.com/api/v2/help_center/en-us/articles.json?page=1&per_page=30"
    while next_url:
        async with AutoCloseAsyncClient() as client:
            res = await client.get(next_url)
            json: dict = res.json()
            tasks = [
                get_a_article(article.get("html_url"))
                for article in json.get("articles")
            ]
             # 等待所有任务完成
            await asyncio.gather(*tasks)
            next_url = json.get("next_page")

asyncio.run(main())
