from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import fitz
import tiktoken
import json
import re
ENCODING_NAME = "cl100k_base"
SECTION_PATTERNS = {
    "section_title": re.compile(r'^\s*(第[一二三四五六七八九十]+节)\s*([^\n]+)', re.MULTILINE),
    "subsection": re.compile(r'^\s*([一二三四五六七八九十（一）（二）（三）（四）（五）（六）（七）（八）（九）（十）(一)(二)(三)(四)(五)(六)(七)(八)(九)(十)]+[、\.\n]*)\s*([^\n]+)', re.MULTILINE),
    "subsection_l3": re.compile(r'^\s*[#]*\s*((?:\d+|[(（]\d+[)）]))[、.]+\s*([^\n]+)', re.MULTILINE)
}
DEFAULT_PATTERNS = {
    "chapter": [r'财务报告', r'财务会计报告'],
    "fintab_section": [r'财务报表'],
    "end_section": [r'合并范围的变更', r'关联关系及关联交易', r'关联方及关联交易', r'关联方关系', 
                   r'关联方关系及其交易', r'关联关系及其交易', r'资产证券化业务的会计处理', r'分部信息'],
    "end_chapter": [r'备查文件目录'],
    "merge_section": [r'合并报表主要项目注释', r'会计报表主要项目注释', r'财务报表主要项目注释',
                    r'合并财务报表项目注释', r'合并财务报表项目附注', r'合并财务报表主要项目注释', 
                    r'合并财务报表重要项目注释', r'合并会计报表主要项目附注', r'合并会计报表主要项目注释', 
                    r'合并会计报表主要项目注释（单位：人民币元）', r'合并会计报表主要项目注释(计量单位：人民币元) '],
}
TERM_LIST = ["存货","应收账款","货币资金","其他应收款","商誉","固定资产","预付款项","长期股权投资","在建工程","未分配利润",
             "其他非流动资产","无形资产","营业收入和营业成本","财务费用", "其他应付款"]

class SingleParser:
    def __init__(self, report_path, max_tokens) -> None:
        self.report_path = Path(report_path)
        self.max_tokens = max_tokens#int(64000)
        self.term_variants = TERM_LIST.copy()
    def _truncate_by_tokens(self, content_str, max_tokens=16000):
        encoding = tiktoken.get_encoding(ENCODING_NAME)
        tokens = encoding.encode(content_str)
        truncated_tokens = tokens[:max_tokens]
        truncated_str = encoding.decode(truncated_tokens)
        return truncated_str
    def _parse_full_content(self, start):
        with fitz.open(self.report_path) as doc:
            full_text = []
            for page_num in range(start, len(doc), 1):
                page = doc[page_num]
                full_text.append(page.get_text("text"))
            content = "\n".join(full_text)
        content_str = f"{self.report_path.name}\n{content}"
        truncated_content = self._truncate_by_tokens(content_str=content_str, max_tokens=self.max_tokens)
        return truncated_content
    def _find_page(self, doc, pattern, keywords, start=0):
        for page_num in range(start, len(doc)):
            text = doc[page_num].get_text("text")
            if not text:
                continue
            for match in pattern.findall(text):
                if any(kw == match[1].strip() for kw in keywords) and page_num > 5:
                    num = match[0].strip().translate(str.maketrans("", "", "、."))
                    return page_num, int(num) if num.isdigit() else None
        return None, None 
    def _check_end_patterns(self, text, current_page):
        """检查结束模式"""
        # 子章节结束模式
        for match in SECTION_PATTERNS["subsection"].findall(text):
            if match[1].strip() in DEFAULT_PATTERNS["end_section"]:
                return current_page - 1
        # 主章节结束模式
        for match in SECTION_PATTERNS["section_title"].findall(text):
            if match[1].strip() in DEFAULT_PATTERNS["end_chapter"]:
                return current_page - 1
        return current_page
    def _find_end_page(self, doc, start):
        end_page = len(doc) - 1
        for page_num in range(start, len(doc)):
            text = doc[page_num].get_text("text")
            if not text:
                continue
            end_page = self._check_end_patterns(text=text, current_page=page_num)
            if end_page != page_num:
                return end_page
        return end_page
    def _find_merge_boundary(self, doc):
        merge_start, _ = self._find_page(doc=doc, pattern=SECTION_PATTERNS["subsection"], 
                                         keywords=DEFAULT_PATTERNS["merge_section"])
        if not merge_start:
            return (None, None)
        merge_end, _ = self._find_page(doc=doc, pattern=SECTION_PATTERNS["subsection"], 
                                       keywords=DEFAULT_PATTERNS["end_section"], start=merge_start)
        if not merge_end:
            merge_end = self._find_end_page(doc=doc, start=merge_start)
        return (merge_start, merge_end)
    def _get_term_variants(self, term: str) -> List[str]:
        """获取会计科目别称"""
        variants_map = {
            "预付款项": ["预付款项", "预付帐款", "预付账款"],
            "存货": ["存货", "存货及存货跌价准备"],
            "营业收入和营业成本": ["营业收入和营业成本", "营业收入及营业成本", "营业收入、营业成本"]
        }
        return variants_map.get(term, [term])
    def _parse_pdf(self, ):
        if not self.report_path.exists():
            raise ValueError("输入的财报文件不存在")
        sections = []
        if "季度报告" in self.report_path.name:
            truncated_content = self._parse_full_content(start=0)
            sections.append(("季报", truncated_content))
            return truncated_content, sections
        with fitz.open(self.report_path) as doc:
            ### 按照commen terms提取
            start, end = self._find_merge_boundary(doc=doc)
            if start is not None and end is not None:
                merge_text = []
                for page_num in range(start, end+1):
                    page = doc[page_num]
                    merge_text.append(page.get_text("text"))
                content = "\n".join(merge_text)
                for term in TERM_LIST:
                    term_variants = self._get_term_variants(term)
                    self.term_variants.extend(term_variants)
                self.term_variants = list(set(self.term_variants))
                matches = list(SECTION_PATTERNS["subsection_l3"].finditer(content))
                # sections = []
                for i, current_match in enumerate(matches):
                    # 提取当前匹配项信息
                    title = current_match.group(2).strip().strip(":").strip()
                    number = current_match.group(1).strip().strip("#").strip()
                    # 前置条件校验
                    if (title not in self.term_variants or not number.isdigit() or any(c.isdigit() for c in title)):
                        continue
                    current_number = int(number)
                    start_pos = current_match.end()
                    end_pos = None
                    # 遍历后续匹配项查找下一个相邻编号
                    for next_match in matches[i+1:]:
                        title_next = next_match.group(2).strip().strip(":").strip()
                        number_next = next_match.group(1).strip().strip("#").strip()
                        if (number_next.isdigit() and not any(c.isdigit() for c in title_next)):
                            if int(number_next) == current_number + 1:
                                end_pos = next_match.start()
                                term_content = content[start_pos:end_pos].strip()
                                sections.append((title, term_content))
                                break
                    # 处理最后一个章节无后续匹配的情况
                    if end_pos is None:
                        continue
                selected_content = []
                for title, term_content in sections:
                    if title in self.term_variants:
                        selected_content.append(f"{self.report_path.name}--{title}\n{term_content}")
                content_str = "\n\n".join(selected_content)
                if selected_content == []:
                    content_str = f"{self.report_path.name}--合并财务报表项目注释\n{content}"
                truncated_content = self._truncate_by_tokens(content_str=content_str, max_tokens=self.max_tokens)
            else:
                truncated_content = self._parse_full_content(start=int(len(doc) / 2))
                sections.append(("全文", truncated_content))
        return truncated_content, sections