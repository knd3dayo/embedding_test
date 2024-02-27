import os, json
from openai import OpenAI, AzureOpenAI

from test01 import load_text, do_test, SENTENCE, DATA_URL

DOCOMENT_OVERVIEW = """
丸美屋食品工業が1960年から販売しているふりかけ「のりたま」は、
玉子味の顆粒ときざみ海苔の組み合わせが特徴であり、画期的な商品とされている。
登録商標でもあり、主原料に高級食材の玉子と海苔を使用している。
"""

def create_test_data(url, chunk_size=500):
    text_list = load_text(url, chunk_size)
    
    return [ DOCOMENT_OVERVIEW + '\n' + text for text in text_list]

if __name__ == "__main__":

    test_data_list = create_test_data(DATA_URL)
    do_test(SENTENCE, test_data_list, "text-embedding-3-small")
