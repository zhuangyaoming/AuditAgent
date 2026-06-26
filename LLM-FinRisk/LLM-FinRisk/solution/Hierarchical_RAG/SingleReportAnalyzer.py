from pathlib import Path
from openai import OpenAI
import requests
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed  # 新增并行处理模块
from prompt import construct_single_report_prompt
from Parser import SingleParser
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

DEFAULT_SYSTEM_ROLE = "你是一个财务审计专家。"
MAX_RETRIES = 3
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

class SingleAnalyzer:
    def __init__(self, report_paths, model_name, ) -> None:
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
    # def _chat_with_ds(self, prompt: str, name, flag) -> tuple[str, str]:
            data = {"model": self.model_name,
                    "messages": [{"role": "system", "content": DEFAULT_SYSTEM_ROLE},
                                {"role": "user", "content": prompt}],
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
    def _analyze(self, content_list, is_analyze=True):
        tasks = []
        # content_str = ""
        # for idx, content in enumerate(content_list):
        #     report_path = self.report_paths[idx]
        #     content_str = content_str + f"{report_path.name}--\n\n{content}\n\n"
        # current_prompt = construct_single_report_prompt(report_content=content_str)
        # tasks.append(("single", current_prompt))
        for idx, content in enumerate(content_list):
            report_path = self.report_paths[idx]
            current_prompt = construct_single_report_prompt(report_content=content)
            tasks.append(("single", report_path.name, current_prompt))
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
                    results.append(("single", name, ""))
                    logging.error(f"处理记录 {name} 时发生错误: {str(e)}", exc_info=True)
        return results
if __name__ == "__main__":
    log_path = "/data1/baisongran/LLM-FinRisk/solution/Hierarchical_RAG/logs/single_analyzer.log"
    report_paths = ["/data1/baisongran/llm-financial-risk/dataset/financial-report/002388/2022年年度报告.pdf",
                    "/data1/baisongran/llm-financial-risk/dataset/financial-report/002388/2023年半年度报告.pdf",
                    "/data1/baisongran/llm-financial-risk/dataset/financial-report/002388/2023年年度报告.pdf"]
    report_paths = [Path(p) for p in report_paths]
    model_name = "deepseek-r1"
    url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer YOUR_API_KEY"}
    setup_logging(log_path=log_path)
    analyzer = SingleAnalyzer(report_paths=report_paths, model_name=model_name, url=url, headers=headers)
    content_list = analyzer._parse()[0]
    results = analyzer._analyze(content_list=content_list)
    print(results)