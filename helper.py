from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import systemprompt
import numpy as np
import json

model_name = "AruniAnkur/aiquest_custom_qwen_1.7B"
embeddingmodel = SentenceTransformer("nomic-ai/nomic-embed-text-v1",trust_remote_code=True)
qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto",device_map="auto")


def get_embedding(docs,attribute):
    texts = [doc.page_content for doc in docs]
    embeddings = embeddingmodel.encode(texts, convert_to_numpy=True)
    print("Number of chunks:", len(texts))
    print("Embedding shape:", embeddings.shape)
    query = attribute
    query_vec = embeddingmodel.encode(query, convert_to_numpy=True)
    cosine_scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )
    doc = docs[cosine_scores.argsort()[-1:][::-1]]
    print(doc)
    return doc,cosine_scores.argsort()[-1:][::-1],embeddings[cosine_scores.argsort()[-1:][::-1]]

def get_qwen_response(template_attribute,template_contract,contract_clause):
    messages = [
    {"role": "system", "content": systemprompt},
    {"role": "user", "content": "Template Attribute:"},
    {"role": "user", "content": template_attribute},
    {"role": "user", "content": "Template Contract:"},
    {"role": "user", "content": template_contract},
    {"role": "user", "content": "Contract Clause:"},
    {"role": "user", "content": contract_clause}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=True,enable_thinking=False
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    generated_ids = qwen_model.generate(**model_inputs,max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = qwen_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = qwen_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    content = json.loads(content)
    print(content)
    return content

def get_qwen_tokenizer():
    return qwen_tokenizer

def get_qwen_model():
    return qwen_model