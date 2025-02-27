# 표준 라이브러리
import re
import os
import gc
import time
from datetime import datetime
from typing import List, Dict, Optional

# 서드파티 라이브러리
import torch
import yaml
from llama_cpp import Llama
from pymongo import MongoClient
from PySide6.QtCore import QRunnable
from dotenv import load_dotenv

# local
from utils.logger import get_logger
from utils.intend import load_intent_model, predict_intent
from utils.valid import valid_input, change_parameter, remove_stop
from config.config import LANGUAGE


logger = get_logger()

massage_llms = None

def load_llm_model(task):
    global massage_llms

    if task == "온열로봇":
        massage_llms = Llama(
            model_path="/home/ft_models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            n_ctx=4096,
            n_gpu_layers=-1,
            temperature=0.1,
            stop=["\n\n", "\n"],
            f16_kv=True,
            verbose=False,
            chat_format="llama-3",
        )
        logger.info("Base 모델 로드 완료")

class LLMWorker(QRunnable):
    """
    LLM 분석 처리하는 작업 스레드
    """

    def __init__(self, task, model, prompt, sentence, recognized_signal, lang = LANGUAGE):
        super().__init__()
        self.task = task
        self.model = model
        self.sentence = sentence
        self.recognized_signal = recognized_signal
        self.lang = lang
        self.prompt_path = self.set_prompt(prompt)
        
        self.llm = massage_llms
    
    def run(self, **kwargs):
        messages = []
        try:
            if self.task == "온열로봇":
                sentence = valid_input(self.task, self.sentence, True)
                if sentence is None:
                        self.abort_on_invalid_sentence()
                        return
                    
                template = self.load_prompt("action_response")
                messages = [
                    {"role": "system", "content": template},
                    {
                        "role": "user",
                        "content": self.sentence
                    }
                ]
            
        except Exception as e:
            logger.error(f"LLM data format error: {e}")

        logger.debug(f"request  llm: {self.task}")

        buffer = ""
        action = None
        pattern_count = 0
        action_start = "`["
        action_state = False
        sentence_end_pattern = re.compile(r'[.!?\n]')
        and_pattern = re.compile(r", $")
        kwargs['stream'] = True

        try:
            for chunk in self.llm.create_chat_completion(messages=messages, **kwargs):
                if 'choices' in chunk and chunk['choices'][0].get('delta', {}).get('content'):
                    content = chunk['choices'][0]['delta']['content']
                    buffer += content
                    pattern_count += len(sentence_end_pattern.findall(content))
                    print(content, end='', flush=True)

                    content = re.sub(r"\[|\]|`", "", content)

                    if pattern_count >= 2:
                        self.recognized_signal.emit(False, content, buffer, None)
                        buffer = ""
                        pattern_count = 0
                    elif action_start in buffer and action_state is False:
                        self.recognized_signal.emit(False, "\n\n", buffer.split("`")[0], None)
                        action_state = True
                    else:
                        if action_state is True:
                            if buffer[-2] == ")":
                                content = content.replace(", ", "\n").replace(",", "\n")
                            elif buffer[(-len(content)-1)] == ",":
                                content = content.strip()
                            else:
                                content = content.replace("), ", ")\n").replace("),", ")\n")

                        if and_pattern.search(buffer) is None or (action_state is True and content != " "):
                            self.recognized_signal.emit(False, content, None, None)
        except Exception as e:
            logger.error(f"LLM error: {e}")

        logger.debug(f"llm result: {buffer}")
        buffer = remove_stop(buffer)

        if action_start in buffer:
            action = buffer.split("```")[1].replace("),", ");")

            try:
                action = change_parameter(self.task, action)
            except Exception as e:
                logger.error("change_parameter Error: ", e)

        if action_state is True:
            buffer = ""

        self.recognized_signal.emit(True, None, buffer, action)

        torch.cuda.empty_cache()
        gc.collect()


    def set_prompt(self, prompt):
        dir_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        return os.path.join(dir_path, "prompts", self.task, "prompt.yml")


    def load_prompt(self, prompt_type: str) -> str:
        """Load a specific prompt type from the prompt file."""
        try:
            logger.debug(f"prompt file path: {self.prompt_path}, prompt_type: {prompt_type}")
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                prompt = data.get(f"{self.task}_prompts", {}).get(prompt_type, "")
            return prompt
        except Exception as e:
            logger.error(f"Error load_prompt: {e}")
            return ""

    def abort_on_invalid_sentence(self):
        logger.debug(f"llm: {self.task} > abort on invalid sentence")

        command = "Error - Invalid sentence."
        for char in command:
            self.recognized_signal.emit(False, char, None, None)
            time.sleep(0.1)

        self.recognized_signal.emit(True, None, command, None)
