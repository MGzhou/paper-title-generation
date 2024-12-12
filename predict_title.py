#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/12/12 09:24:41
@Desc :None
'''
import torch
# 推荐 transformers >= 4.33.2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def preprocess(text):
    return text.replace("\n", "_")

def postprocess(text):
    return text.replace("_", "\n")

def predict(model, tokenizer, text, title_num=1, sample=False, top_p=0.6, input_length=512, output_length=128, device='cpu'):
    """ 
    预测函数
    Args:
        model: 模型
        tokenizer: 分词器
        text: 输入文本
        title_num: 生成标题数量, 推荐1-4,。注意多个标题是可能重复的。
        sample: 是否采样
        top_p: 采样概率, 只有sample为True时生效
        input_length: 输入长度限制，超出会自动截断
        output_length: 输出长度限制，超出会自动截断
    """
    prompt = f"""为以下的论文摘要生成标题：\n{text}\n标题："""

    if title_num <= 4:
        num_beams = 4
        num_return_sequences = title_num
    else:
        num_beams = title_num
        num_return_sequences = title_num
    prompt = preprocess(prompt)
    encodes = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=input_length).to(device)
    outputs = model.generate(encodes.input_ids, 
        max_length=output_length,
        do_sample=sample,
        repetition_penalty=1.2,
        length_penalty=0.6,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True,
        top_p=top_p,
    )
    title_list = []
    for i, beam_output in enumerate(outputs):
        title = postprocess(tokenizer.decode(beam_output, skip_special_tokens=True))
        title_list.append(title)
    return title_list

if __name__ == "__main__":
    device = torch.device("cpu")  # 如果使用GPU 为 "cuda"
    model_name = "/data/zmp/progect/LLM/finetuning/mt5/outputs/model_files_1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    # 原标题：植物对盐碱胁迫的响应机制研究进展
    text = """盐碱胁迫是制约植物生长发育的主要非生物胁迫之一,也是制约农作物生产和生态环境建设的严峻问题。研究作物的耐盐碱机理,对开发和有效利用盐碱地有重要的现实意义。许多研究将盐碱胁迫笼统称为盐胁迫,实际上这是两种不同的非生物胁迫,且碱胁迫对植物的伤害要大于盐胁迫。总结性阐述了盐碱胁迫对植物的危害。从生物量、光合作用、离子平衡和膜透性等方面分析了植物对盐碱胁迫的响应机制,并结合最新研究从多角度综述了植物的抗盐碱机理,包括合成渗透调节物质、提高抗氧化酶活性、对离子的选择性吸收及p H平衡和诱导抗盐碱相关基因表达。提出了抗盐碱性的途径,即外源物质的加入、与真菌的协同效应、利用生物技术手段、培育耐盐碱品种和抗性锻炼。最后针对植物适应盐碱逆境方面的研究进行了展望,提出了当前研究需要解决的问题和突破口,旨在为提高植物耐盐碱能力、增加作物产量提供一定的理论依据。"""
    title_list = predict(model, tokenizer, text, title_num=3, sample=False, top_p=0.6, input_length=512, output_length=128, device=device)
    print(title_list)  # ['植物对盐碱胁迫的响应机制及抗盐碱机理研究进展', '植物对盐碱胁迫的响应机制及其抗盐碱机理研究进展', '植物耐盐碱机理及抗盐碱机理研究进展']
