import os, json
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

from test01 import load_text, do_test, SENTENCE, DATA_URL


def create_summary(client:OpenAI|AzureOpenAI, chat_model_name, document):

    json_format = "{summary:文章の要約}"
    json_string = json.dumps(json_format)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
         - 以下の文章の要約を作成してください.
         - 出力はJSON形式で、以下のようにしてください。
         {json_string}
        ---
        {document}             
        """},
    ]
    # 要約の作成
    response = client.chat.completions.create(
        model = chat_model_name,
        messages=messages,
        response_format= { "type":"json_object" },
    )

    response_message = response.choices[0].message.content
    return json.loads(response_message)

def create_summary_list(client:OpenAI|AzureOpenAI, chat_model_name, text_list):
    result = []
    for i in range(len(text_list)):

        text = text_list[i]
        # タイトルと要約
        title_and_summary = create_summary(client, chat_model_name, text)
        result.append(title_and_summary)
    
    return [item["summary"] for item in result]

def create_test_data(url, chunk_size=500, create_list_func=create_summary_list):
    text_list = load_text(url, chunk_size)

    # .envファイルの読み込み
    load_dotenv()
    # 各テキストの要約を取得
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    chat_model_name = "gpt-3.5-turbo"

    return create_list_func(client, chat_model_name, text_list)

if __name__ == "__main__":

    test_data_list = create_test_data(
        DATA_URL, 
        create_list_func=create_summary_list
        )
    do_test(SENTENCE, test_data_list, "text-embedding-3-small")
