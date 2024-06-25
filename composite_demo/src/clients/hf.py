"""
HuggingFace client.
"""
import os
import threading
from collections.abc import Generator
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from client import Client, process_input, process_response
from conversation import Conversation
from torch.nn import DataParallel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch




class HFClient(Client):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        # 使用 accelerate 库进行更细粒度的显存管理
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     trust_remote_code=True,
        #     torch_dtype=torch.float16,
        #     device_map="cuda",
        # ).eval()
        self.model = load_checkpoint_and_dispatch(
            self.model,
            model_path,
            device_map="auto",  # 可以改为 'balanced' 或者手动指定 device_map
            # no_split_module_classes=["GPTJBlock"],  # 根据你的模型结构调整
            dtype=torch.float16
        ).eval()
        # self.model = DataParallel(self.model, device_ids=list(range(torch.cuda.device_count()))).eval()

    def generate_stream(
        self,
        tools: list[dict],
        history: list[Conversation],
        **parameters,
    ) -> Generator[tuple[str | dict, list[dict]]]:
        try:
            chat_history = process_input(history, tools)
            model_inputs = self.tokenizer.apply_chat_template(
                chat_history,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
            streamer = TextIteratorStreamer(
                tokenizer=self.tokenizer,
                timeout=5,
                skip_prompt=True,
            )
            generate_kwargs = {
                **model_inputs,
                "streamer": streamer,
                "eos_token_id": [151329, 151336, 151338],
                "do_sample": True,
            }
            generate_kwargs.update(parameters)
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()
            total_text = ""
            for token_text in streamer:
                total_text += token_text
                yield process_response(total_text, chat_history)
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            if "device-side assert triggered" in str(e):
                print("CUDA error detected. Please check the input data and model parameters.")
                # 启用 CUDA_LAUNCH_BLOCKING 以进行同步错误报告
                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
                raise

# 调用代码时设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
