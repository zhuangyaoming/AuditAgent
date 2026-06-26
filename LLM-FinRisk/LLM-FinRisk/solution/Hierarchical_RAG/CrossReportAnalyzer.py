from pathlib import Path
from Parser import SingleParser
from prompt import construct_term_trend_prompt, construct_cross_report_prompt
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed  # 新增并行处理模块
import requests
from openai import OpenAI
import logging
from logging.handlers import RotatingFileHandler
import json

# 在类定义前添加日志配置
def setup_logging(log_path:str):
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 文件日志（自动滚动，最大10MB，保留3个备份）
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=20*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    # 获取根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

TERM_LIST = ["存货","应收账款","货币资金","其他应收款","商誉","固定资产","预付款项","长期股权投资","在建工程","未分配利润",
             "其他非流动资产","无形资产","营业收入和营业成本","财务费用", "其他应付款"]
MAX_RETRIES = 3
DEFAULT_SYSTEM_ROLE = "你是一个财务审计专家。"
MODEL_CONFIGS = {"qwq-32b": {"model": "qwq-32b"},
                 "deepseek-r1": {"model": "deepseek-r1"}}
API_CONFIGS = {"qwq-32b": {"api_key": "123",
                           "api_url": "http://127.0.0.1:8000/v1/chat/completions",
                           "headers": {"Content-Type": "application/json", "Authorization": "Bearer 123",},
                           "base_url": "http://127.0.0.1:8000/v1"},
               "deepseek-r1": {"url":"https://cloud.infini-ai.com/maas/v1/chat/completions",
                               # TODO: Set your API key via environment variable.
                               # Original key backed up in api_keys_backup.txt (EXCLUDED from git).
                               "headers":{"Content-Type": "application/json", "Authorization": f"Bearer YOUR_API_KEY"}}}

class CrossAnalyzer:
    def __init__(self, report_paths, model_name) -> None:
        self.report_paths = report_paths
        self.model_name = model_name
    def _chat_with_model(self, prompt, name, flag):
        if self.model_name == "qwq-32b":
            """调用大模型API"""
            # client = OpenAI(**API_CONFIGS[self.model_name])
            # response = client.chat.completions.create(
            #     model=MODEL_CONFIGS[self.model_name]["model"],
            #     messages=[{"role": "system", "content": DEFAULT_SYSTEM_ROLE},
            #             {"role": "user", "content": prompt}],
            #     stream=False, temperature=0.7)
            # content = response.choices[0].message.content
            # think_end = content.find("</think>")
            # if think_end != -1:
            #     think_content = content[:think_end].strip()
            #     answer_content = content[think_end+8:].strip()
            # else:
            #     think_content = ""
            #     answer_content = content
            # return flag, name, think_content, answer_content
            current_messages = [{"role": "system", "content": DEFAULT_SYSTEM_ROLE},
                                {"role": "user", "content": prompt}]
            current_data = {"model": self.model_name, "messages": current_messages,
                            "stream": False, "temperature": 0.7}
            try:
                response = requests.post(API_CONFIGS[self.model_name]["api_url"], headers=API_CONFIGS[self.model_name]["headers"], 
                                         data=json.dumps(current_data))
                response.raise_for_status()
                # 获取模型输出
                json_data = response.json()
                content = json_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                # 提取推理内容和评估结果
                think_end = content.find("</think>")
                if think_end != -1:
                    think_content = content[:think_end].strip()
                    output_content = content[think_end + 8:].strip()  # 8 是 </think> 的长度
                else:
                    think_content = ""
                    output_content = content
                return flag, name, think_content, output_content
            except Exception as e:
                print(f"处理风险评估时发生错误: {str(e)}")
                return flag, name, "", ""
        elif self.model_name == "deepseek-r1":
            data = {"model": self.model_name,
                    "messages": [{"role": "system", "content": DEFAULT_SYSTEM_ROLE}, {"role": "user", "content": prompt}],
                    "stream": False, "temperature": 0.6, "top_p": 0.9, "force_think": True}
            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    response = requests.post(API_CONFIGS[self.model_name]["url"], headers=API_CONFIGS[self.model_name]["headers"], json=data)
                    # 检查是否为429错误
                    if response.status_code == 429:
                        wait_time = 300  # 5分钟 = 300秒
                        retries += 1
                        if retries <= MAX_RETRIES:
                            print(f"遇到429错误（请求过多），等待{wait_time}秒后进行第{retries}次重试...")
                            # 使用倒计时等待
                            self._countdown_wait(wait_time)
                            continue
                        else:
                            print(f"已达到最大重试次数（{MAX_RETRIES}次），放弃请求")
                            return flag, name, "", ""
                    # 检查其他HTTP错误
                    response.raise_for_status()
                    json_data = response.json()
                    reasoning_content = json_data.get('choices', [{}])[0].get('message', {}).get('reasoning_content', '')
                    output_content = json_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    return flag, name, reasoning_content, output_content
                except requests.exceptions.HTTPError as e:
                    # 处理其他HTTP错误
                    print(f"调用模型时发生HTTP错误: {str(e)}")
                    return flag, name, "", ""
                except Exception as e:
                    print(f"调用模型时发生错误: {str(e)}")
                    return flag, name, "", ""
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析并清洗API响应"""
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return cleaned
    def _parse(self, ):
        max_tokens = int(64000 / len(self.report_paths))
        content_list, sections_list = [], []
        for i, report_path in enumerate(self.report_paths):
            current_parser = SingleParser(report_path=report_path, max_tokens=max_tokens)
            current_content, sections = current_parser._parse_pdf()
            content_list.append(current_content)
            sections_list.append(sections)
        return content_list, sections_list
    def _analyze(self, sections_list, is_analyze=True):
        tasks = []
        section_dict = {}
        # content_str = ""
        # for idx, sections in enumerate(sections_list):
        #     report_path = self.report_paths[idx]
        #     for title, section_content in sections:
        #         if title not in list(section_dict.keys()):
        #             section_dict[title] = {}
        #         section_dict[title][report_path.name] = section_content
        # for section_key, section_value in section_dict.items():
        #     if section_key in TERM_LIST:
        #         for report_key, report_value in section_value.items():
        #             content_str = content_str + f"{section_key}--\n\n{report_key}\n\n{report_value}\n\n"
        # current_prompt = construct_cross_report_prompt(report_content=content_str)
        # tasks.append(("cross", current_prompt))
        for idx, sections in enumerate(sections_list):
            report_path = self.report_paths[idx]
            for title, section_content in sections:
                if title not in list(section_dict.keys()):
                    section_dict[title] = {}
                section_dict[title][report_path.name] = section_content
        for section_key, section_value in section_dict.items():
            if section_key != "季报":
                content = ""
                for report_key, report_value in section_value.items():
                    content = content + f"{section_key}--{report_key}--\n{report_value}\n\n"
                if "季报" in list(section_dict.keys()):
                    for key, value in section_dict["季报"].items():
                        content = content + f"{section_key}--{key}--\n{value}\n\n"
                current_prompt = construct_term_trend_prompt(report_content=content)
                tasks.append(("cross", section_key, current_prompt))
        if is_analyze == False:
            return tasks
        results = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), 30)) as executor:
            futures = [executor.submit(self._chat_with_model, prompt, name, flag) for flag, name, prompt in tasks]
            for future in as_completed(futures):
                try:
                    flag, name, reason_content, output_content = future.result()
                    output_content = self._parse_response(output_content)
                    results.append((flag, name, output_content))
                    logging.info(f"记录 {name} 已完成，当前进度: {len(results)}/{len(tasks)}")
                except Exception as e:
                    results.append(("cross", name, ""))
                    logging.error(f"处理记录 {name} 时发生错误: {str(e)}", exc_info=True)
        merge_trend_content = ""
        for term, trend in results:
            merge_trend_content = merge_trend_content + f"{term}--\n{trend}\n\n"
        cross_report_prompt = construct_cross_report_prompt(report_content=merge_trend_content)
        flag, name, final_reason, final_output = self._chat_with_model(prompt=cross_report_prompt, name="", flag="cross")
        return final_output

if __name__ == "__main__":
    log_path = "/data1/baisongran/LLM-FinRisk/solution/Hierarchical_RAG/logs/cross_analyzer.log"
    report_paths = ["/data1/baisongran/llm-financial-risk/dataset/financial-report/002388/2022年年度报告.pdf",
                    "/data1/baisongran/llm-financial-risk/dataset/financial-report/002388/2023年半年度报告.pdf",
                    "/data1/baisongran/llm-financial-risk/dataset/financial-report/002388/2023年年度报告.pdf"]
    report_paths = [Path(p) for p in report_paths]
    model_name = "deepseek-r1"
    url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer YOUR_API_KEY"}
    setup_logging(log_path=log_path)
    analyzer = CrossAnalyzer(report_paths=report_paths, model_name=model_name, url=url, headers=headers)
    sections_list = analyzer._parse()[1]
    result = analyzer._analyze(sections_list=sections_list)
    print(result)