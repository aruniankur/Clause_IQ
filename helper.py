from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import systemprompt
import numpy as np
import json

model_name = "AruniAnkur/aiquest_custom_qwen_1.7B"
embeddingmodel = SentenceTransformer("nomic-ai/nomic-embed-text-v1",trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_qwen_response(attribute,template_contract,contract_clause):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",device_map="auto")
    messages = [
    {"role": "system", "content": systemprompt},
    {"role": "user", "content": "Template Attribute:"},
    {"role": "user", "content": attribute},
    {"role": "user", "content": "Template Contract:"},
    {"role": "user", "content": template_contract},
    {"role": "user", "content": "Contract Clause:"},
    {"role": "user", "content": contract_clause}
    ]
    print("-------------------------------fff123ff-")
    text = tokenizer.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=True,enable_thinking=False
    )
    print("-------------------------------fff123ff-")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print("-------------------------------fff123ff-")
    generated_ids = model.generate(**model_inputs,max_new_tokens=100)
    print("-------------------------------fff123ff-")
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    print("-------------------------------fff123ff-")
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    print("-------------------------------fff123ff-")
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    content = json.loads(content)
    return content

def getresult(template_clause,template_embedding,contract_clause,contract_embedding,attribute,tre):
    #print(attribute)
    print("---------------------ffff-----------")
    attribute_vec = embeddingmodel.encode(attribute, convert_to_numpy=True)
    print(attribute_vec.shape)
    template_vec = template_embedding
    print(template_vec.shape)
    contract_vec = contract_embedding
    print(contract_vec.shape)
    a_tem_cosine_scores = np.dot(template_vec, attribute_vec) / (
        np.linalg.norm(template_vec) * np.linalg.norm(attribute_vec)
    )
    a_con_cosine_scores = np.dot(contract_vec, attribute_vec) / (
        np.linalg.norm(contract_vec) * np.linalg.norm(attribute_vec)
    )
    print(a_tem_cosine_scores,a_con_cosine_scores)
    print(a_tem_cosine_scores.argsort()[-1:][::-1],a_con_cosine_scores.argsort()[-1:][::-1])
    tem_doc = template_clause[a_tem_cosine_scores.argsort()[-1:][::-1][0]]
    con_doc = contract_clause[a_con_cosine_scores.argsort()[-1:][::-1][0]]

    print(tem_doc,con_doc)
    print("-------------------------------fffff-")
    content = get_qwen_response(attribute,tem_doc,con_doc)
    print(content)
    return {
        "attribute": tre,
        "template_clause": tem_doc,
        "your_clause": con_doc,
        "match_status": content['classification'],
        "confidence": content['confidence'],
        'reasoning': "resoning",
        "similarity_score" : 1,
    }