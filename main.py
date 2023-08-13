import copy
import json
import logging
import multiprocessing
from multiprocessing.managers import ListProxy
import traceback
from typing import Any, Optional
import pandas as pd
import time
import openai
import os
import sys
import tiktoken
from tiktoken import Encoding

from tqdm import tqdm
from gpt3_interactor import CompletionTypeEnum, Gpt3Interactor
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from prompt.FileContentPrompt import FileContentPrompt
from prompt.FileNamePrompt import FileNamePrompt
from utils.file_operations import *
from utils.list_operations import *

#ENGINE = "gpt-3.5-turbo"
ENGINE = "gpt-3.5-turbo-16k"
logger = logging.getLogger(__name__)
GENERATE_MAX_PROMPT_LENGTH = 12000
GENERATE_MAX_GENERATED_LENGTH = 12000
DEBUG_MAX_PROMPT_LENGTH = 12000
DEBUG_MAX_GENERATED_LENGTH = 500
# 进程数
PROCESS_COUNT = 5
# 每隔多少个样本保存一次检查点文件
SAVE_ITER = 20
# .replace("\\", "/"): 防止windows下路径出错
CHECKPOINT_FOLDER = os.path.join("temporary", f"md_translate_{ENGINE}").replace("\\", "/")

INPUT_FOLDER = os.path.join("data", "input").replace("\\", "/")
OUTPUT_FOLDER = os.path.join("data", "output").replace("\\", "/")


def worker_helper(args_dict):
    return worker(**args_dict)

def worker(index:int, filename_prompt:str, content_prompt:str, interactor_manager_list:list[Gpt3Interactor])->tuple[int, dict]:
    interactor:Gpt3Interactor = interactor_manager_list.pop(0)
    encoding = tiktoken.encoding_for_model(ENGINE)
    content_prompt_tokens_count = len(encoding.encode(content_prompt))
    temp_generated_dict:dict[str, dict|None] = \
    {
        "filename": None,
        "content": None,
        "response": None,
    }
    

    input_messages:list[dict[str, str]] = \
    [
        {
            "role": "system", 
            # 你是一位专业的程序员与中英翻译人员。你精通包括 Python 在内的多种编程语言，并且知道如何恰当的将许多专有名词从中文翻译成英文，以及知道什么部分应该翻译、什么部分不需要翻译。下面将提供一份中文的程序文档，请你将它的文件名和文件内容翻译为英文。
            "content": "You are a professional programmer and translator fluent in both Chinese and English. You are proficient in multiple programming languages, including Python, and you know how to accurately translate many specialized terms from Chinese to English. You also understand what parts should be translated and what parts do not need translation. Below is a Chinese programming document, please translate its file name and content into English."
        },
        {
            "role": "user", 
            "content": filename_prompt
        }
    ]
    
    response = interactor.generate(
        engine=ENGINE, 
        prompt=input_messages, 
        max_tokens=GENERATE_MAX_GENERATED_LENGTH,
        completion_type=CompletionTypeEnum.ChatCompletion,
        stop = ["---"],
        )
    temp_generated_dict["filename_response"] = response
    temp_generated_dict["filename"] = response["choices"][0]["message"]["content"]
    
    input_messages.append\
    (
        {
            "role": response["choices"][0]["message"]["role"], 
            "content": response["choices"][0]["message"]["content"]
        }
    )
    input_messages.append\
    (
        {
            "role": "user", 
            "content": content_prompt
        }
    )
    
    response = interactor.generate(
        engine=ENGINE, 
        prompt=input_messages, 
        max_tokens=GENERATE_MAX_GENERATED_LENGTH,
        completion_type=CompletionTypeEnum.ChatCompletion,
        stop = [],
        )
    temp_generated_dict["content_response"] = response
    temp_generated_dict["content"] = response["choices"][0]["message"]["content"]
    
    return (index, temp_generated_dict)


def read_files_in_folder(folder_path)->list[tuple[str, str]]:
    result:list[tuple[str, str]] = []
    try:
        # 遍历文件夹下的所有文件
        for filename in os.listdir(folder_path):
            file_path:str = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                # 将文件名和内容组成元组，添加到结果列表
                result.append((filename, content))
    except Exception as e:
        print("Error:", e)
    return result

def main()->None:
    logging.basicConfig(
            format='%(asctime)s-%(levelname)s-%(name)s-: %(message)s',
            datefmt=f'%m/%d/%Y %H:%M:%S',
            level=logging.INFO)    
    logger.setLevel(logging.INFO)
    logger.info("Starting...")
    
    # Load openai keys
    with open("keys.txt", encoding='utf-8') as f:
        keys = [line.strip() for line in f.readlines()]
    # key 分成 THREADS 份
    keys_split = divide_list_into_n_parts(keys, PROCESS_COUNT)
    interactor_list:list[Gpt3Interactor] = []
    for _keys in keys_split:
        interactor = Gpt3Interactor(
            api_keys=_keys, 
            model_name=ENGINE,
            logger=logger
        )
        interactor_list.append(interactor)
    # 多进程准备
    manager = multiprocessing.Manager()
    # interactor_manager_list 进程间通信分配Gpt3交互器，要的自己pop，用完了append
    interactor_manager_list:'ListProxy[Gpt3Interactor]' = manager.list(interactor_list)
    pool = multiprocessing.Pool(processes=PROCESS_COUNT)
        
        
    # 读取数据
    inputs = read_files_in_folder(INPUT_FOLDER)
    
    fnp = FileNamePrompt()
    fcp = FileContentPrompt()
    
    workloads_list:list[dict] = []        
    for index, (filename, content) in enumerate(inputs):
        workloads_list.append\
        (
            {
                "index": index, 
                "filename_prompt": fnp.Render({"filename": filename}),
                "content_prompt": fcp.Render({"content": content}),
                
                "interactor_manager_list": interactor_manager_list,
            }
        )
        
    
    # 结果暂存
    generated_dict:dict[int, dict[str, Any]] = {}
    generated_count = 0
    # 读取checkpoint，剔除已经生成的样本
    if os.path.exists(CHECKPOINT_FOLDER):
        ORIGINAL_DATASET_SIZE = len(workloads_list)
        already_generated:dict[str, dict[str, Any]] = load_and_merge_multiple_dicts_or_lists(CHECKPOINT_FOLDER, {})
        logger.info(f"检测到已经生成的样本，共 {len(already_generated)} 个。将从数据集中剔除它们。")
        for already_generated_index_str in already_generated.keys():
            # 保存后再读取index会变成字符串，转换一下
            already_generated_index = int(already_generated_index_str)
            for workload in workloads_list:
                if workload["index"] == already_generated_index:
                    if already_generated[already_generated_index_str]['CODEX'][0] != workload['row']['question']:
                        print(f"Index {already_generated_index}: 已经生成的样本的原NL与数据集中的样本不匹配，已经生成的样本NL为：{already_generated[already_generated_index_str]['CODEX'][0]}，数据集中对应 INDEX 的样本NL为：{workload['row']['question']}。")
                        print("为了数据完整性，程序将退出。请检查数据集和已经生成的样本。")
                        exit()
                    workloads_list.remove(workload)
                    generated_dict[already_generated_index] = already_generated[already_generated_index_str]
                    break  
        generated_count = len(generated_dict)      
        logger.info(f"数据集原大小：{ORIGINAL_DATASET_SIZE}，现大小：{len(workloads_list)}")
    else:
        os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
        
    # 启动任务
    with tqdm(total=len(workloads_list), desc=f"推理") as pbar:
        temp_generated_dict = {}
        try:
            iterer = pool.imap_unordered(worker_helper, workloads_list, chunksize=1)
        except Exception as e:
                traceback.print_exc()
        for result in iterer:
            try:
                index = result[0]
                temp_generated_dict_content = result[1]
                #generated_dict[index] = result_dict
                temp_generated_dict[index] = temp_generated_dict_content
                generated_count += 1
                # 保存检查点文件
                try:
                    if generated_count % SAVE_ITER == 0 and generated_count > 0:
                        logger.warning(f"保存检查点文件中，请勿关闭")
                        save_variable_to_json_file(temp_generated_dict, os.path.join(CHECKPOINT_FOLDER, f"{generated_count}.json"), logger=logger)
                        generated_dict.update(temp_generated_dict)
                        temp_generated_dict.clear()
                        logger.warning(f"保存检查点文件完毕")
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"保存检查点文件失败 {e}")
                pbar.update() 
            except Exception as e:
                traceback.print_exc()
    # 推理结束保存检查点文件
    try:
        logger.warning(f"保存检查点文件中，请勿关闭")
        save_variable_to_json_file(temp_generated_dict, os.path.join(CHECKPOINT_FOLDER, f"{generated_count}.json"), logger=logger)
        generated_dict.update(temp_generated_dict)
        logger.warning(f"保存检查点文件完毕")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"保存检查点文件失败 {e}")        
    pbar.update() 
    pool.close()
    pool.join()
    
    # 重新排序并提取结果
    sorted_result = sorted(generated_dict.items())
    total_result_list:list[dict] = []
    for (__index, value) in sorted_result:
        total_result_list.append(value)
    
        
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    for result_dict in total_result_list:
            file_path = os.path.join(OUTPUT_FOLDER, result_dict["filename"].strip())
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(result_dict["content"])
        
        
    save_variable_to_json_file(total_result_list, f"result/md_translate_{ENGINE}.json", logger=logger)
    
    

if __name__ == '__main__':
    main()