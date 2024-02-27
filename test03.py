import os, json
from openai import OpenAI, AzureOpenAI

from test01 import do_test, SENTENCE, DATA_URL
from test02 import create_test_data

def create_summary_with_synopsis(client:OpenAI|AzureOpenAI, chat_model_name, document, synopsis) -> dict:
    json_format = "{summary:文章の要約}"
    json_string = json.dumps(json_format)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
         - 以下の文章はあるドキュメントの一部です。この文章の要約を作成してください.
         - [ここまでの要約]はここまでの文章の要約の要約です。
         - [ここまでの要約]がある場合は、[ここまでの要約]を参考にして要約を作成してください。
         - 出力はJSON形式で、以下のようにしてください。
         {json_string}
        ---
        [ここまでの要約]:{synopsis}
        ---
        [文章]:{document} 
        ---
        """},
    ]
    # 要約の作成
    response = client.chat.completions.create(
        model = chat_model_name,
        messages=messages,
        response_format= { "type":"json_object" },
    )

    response_message = response.choices[0].message.content
    # print(response_message)
    return json.loads(response_message)

def create_summary_with_synopsis_list(client:OpenAI|AzureOpenAI, chat_model_name, text_list):
    title_and_summary_list = []
    synopsis = "なし"
    for i in range(len(text_list)):
        text = text_list[i]
        # 要約の作成.あらすじとして、前の文章の要約を使う
        title_and_summary = create_summary_with_synopsis(client, chat_model_name, text, synopsis)
        title_and_summary_list.append(title_and_summary)
        synopsis = title_and_summary.get("summary", "")
         
    return [item["summary"] for item in title_and_summary_list]

if __name__ == "__main__":

    test_data_list = create_test_data(
        DATA_URL,
        create_list_func=create_summary_with_synopsis_list
        )
    do_test(SENTENCE, test_data_list, "text-embedding-3-small")
