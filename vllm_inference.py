import gc
import time
import torch
import json
import os
import sys
from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
# from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import asyncio
from loguru import logger

# INFO 레벨 이상의 로그만 출력
logger.remove()  # 기존 핸들러 제거
logger.add(lambda msg: None, level="INFO")  # INFO 로그는 무시
logger.add(sys.stderr, level="WARNING")     # WARNING 이상만 출력

prompt = """
"""
def save_to_json(data):
    file_path = "/home/result/massage_4bit.json"
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"Saved results to {file_path}")

async def run_inference(llm, prompt, user_input, tokenizer):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ]
    formatted_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    example_input = {
        "prompt": formatted_messages,
        "stream": True, # assume the non-streaming case
        "request_id": 0,
    }

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=256,
        top_k=40,
        top_p=0.95,
        min_p=0.05,
        best_of=1,
        n=1,
        stop=["H:", "user:", "\n\n", "\n"],  # 문자열을 만나면 생성 중지
        # guided_decoding=guided_decoding_params,
        stop_token_ids=eos_token_id,
    )

    ttft = None
    gen_result = []
    token_count = 0
    full_response = ""

    results_generator = engine.generate(
        formatted_messages,
        sampling_params,
        example_input["request_id"])

    start = time.perf_counter()
    async for request_output in results_generator:
        current_time = time.perf_counter()
        if ttft is None:
            ttft = current_time - start
            print(f"\n첫 토큰 생성 시간 (TTFT): {ttft:.4f}초")
        new_text = request_output.outputs[0].text
        delta = new_text[len(full_response):]  # 새로 추가된 부분만 추출
        print(delta, end="", flush=False)
        full_response = new_text
        gen_result.append(delta)

    end = time.perf_counter()
    latency = end - start
    response_text = ''.join(gen_result)
    token_count = len(gen_result)
    print(f"gen_result:{gen_result}")
    if token_count > 1 and (latency - ttft) > 0:
        tps = token_count / (latency - ttft)
    else:
        tps = 0

    print(f"\n질문: {user_input}")
    print(f"response_text:{response_text}")
    print(f"지연시간:{latency}")
    print(f"ttft :{ttft}")
    print(f"tps:{tps}")
    print(f"token_count:{token_count}")
    return response_text, latency, ttft, tps, token_count

# guided_decoding_params = GuidedDecodingParams(
#         regex=".+\s*(```\[.*?\]```)?",
#         #backend="outlines",
#         )

if __name__ == "__main__":
    model_path = "/home/models/massage_vllm_16bit"
    json_data="/home/dataset/massage_v2_test.json"
    with open(json_data, 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    latency_ = []
    ttft_ = []
    tps_ = []
    token_count_ = []
    results = []  # JSON 저장용

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=model_path,
            max_model_len=8192,
            trust_remote_code= True,  # use for hf model
            gpu_memory_utilization= 0.60,
            tensor_parallel_size= 1,
            enforce_eager= True,
            # dtype= "float16",
            quantization = "bitsandbytes",
            load_format="bitsandbytes",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    for doc in test_data:
        user_input = doc['human']
        response, latency, ttft, tps, token_count = asyncio.run(run_inference(engine, prompt, user_input, tokenizer))

        latency_.append(latency)
        ttft_.append(ttft)
        tps_.append(tps)
        token_count_.append(token_count)
        results.append({
            "user_input": user_input,
            "expected_output": doc['output'],
            "result_text": response
        })

    print("-" * 50)
    avg_latency = round(sum(latency_) / len(latency_), 2)
    avg_tpot = round(sum(tps_) / len(tps_), 2)
    avg_ttft = round(sum(ttft_) / len(ttft_), 2)
    avg_token = sum(token_count_) / len(token_count_)

    print(f"ttft val : {avg_ttft}")
    print(f"latency val : {avg_latency}")
    print(f"tpot val : {avg_tpot}")
    print(f"token val : {avg_token}")

    save_to_json(results)
