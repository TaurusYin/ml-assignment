'''
%%Assignment for CTW.%%
This script perform inference to translate a given text.
'''
import json
import requests
from flask import Flask
from flask import request
from flask_restful import Api , Resource 
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer



class Translation(Resource):
    def __init__(self):
        self.max_memory_mapping = {0: "2GB", 1: "4GB"}
        self.model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M',device_map="auto", load_in_8bit=True, max_memory=self.max_memory_mapping)
        self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M', src_lang="en", tgt_lang="ja",device_map="auto", load_in_8bit=True, max_memory=self.max_memory_mapping)

    def post(self):
        payload = request.get_json()
        data = payload['payload']
        fromLang=data["fromLang"]
        records=data["records"]
        toLang=data["toLang"]
        src_text = records[0]["text"]
        self.tokenizer.src_lang = fromLang
        model_inputs = self.tokenizer(src_text, return_tensors="pt")
        generated_tokens =self. model.generate(**model_inputs, forced_bos_token_id=self.tokenizer.get_lang_id(toLang))
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        outputJson={
        "result":[
            {
                "id":records[0]["id"],
                "text":result[0]
            }
        ]
        }

        return outputJson

app = Flask(__name__)
api = Api(app)

api.add_resource(Translation, '/translation')

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=9527)