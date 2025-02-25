import os
import platform
import requests
from pathlib import Path
import openvino_genai as ov_genai
import sys

SUPPORTED_OPTIMIZATIONS = ["INT4", "INT4-AWQ", "INT4-NPU", "INT8", "FP16"]
genai_chat_template_qwen_1_5B = "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}"

import huggingface_hub as hf_hub

# hub_api = hf_hub.HfApi()
# hf_hub.snapshot_download(ov_model_hub_id, local_dir=model_dir)
# print(f"✅ {precision} {model_id} model downloaded and can be found in {model_dir}"
# return model_dir


def deepseek_partial_text_processor(partial_text, new_text):
    partial_text += new_text
    return partial_text.split("</think>")[-1]

#model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#model_dir = "models/models--AIFunOver--Qwen2.5-14B-Instruct-1M-openvino-fp16"

from pathlib import Path

model_name = "DeepSeek-R1-Distill-Llama-8B"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_dir = "../models/DeepSeek-R1-Distill-Llama-8B-int4-ov"
#model_dir = "../models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-fp16-ov"
device = "GPU"

# optimum-cli export openvino --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --task text-generation-with-past --weight-format int4 "./DeepSeek-R1-Distill-Llama-8B-int4-ov"
# not working optimum-cli export openvino --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --task text-generation-with-past --weight-format int4 "./deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-int4-ov"
# not working optimum-cli export openvino --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --task text-generation-with-past --weight-format fp16 "./deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-fp16-ov"
# optimum-cli export openvino --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --task text-generation-with-past --weight-format int8 "models/DeepSeek-R1-Distill-Llama-8B-int8-ov"
## ************************

# if not model_dir.exists():
#     ! optimum-cli export openvino --model $model_id --task text-generation-with-past --weight-format int4 $model_dir
#
# # convert OV tokenizer if needed
# convert_tokenizer ./DeepSeek-R1-Distill-Llama-8B-int4-ov --with-detokenizer -o ./DeepSeek-R1-Distill-Llama-8B-int4-ov



pipe = ov_genai.LLMPipeline(model_dir, device)

pipe.get_tokenizer().set_chat_template(genai_chat_template_qwen_1_5B)

generation_config = ov_genai.GenerationConfig()

generation_config.max_new_tokens = 128


def streamer(subword):
    print(subword, end="", flush=True)
    sys.stdout.flush()
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False


input_prompt = "What is OpenVINO?"
print(f"Input text: {input_prompt}")
result = pipe.generate(input_prompt, generation_config, streamer)
print("result \n")
print(result)



