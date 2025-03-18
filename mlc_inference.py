import sys
import gc
import time
import datetime
import torch

from mlc_llm import MLCEngine
from mlc_llm.serve.config import EngineConfig

model_path='Llama-3.1-8B-Instruct-q4f16_1-MLC'

template = """당신은 사용자의 온열치료 요청을 정확하게 해석하고 여러 신체 부위에 대한 온열치료 설정을 도와주는 보조 로봇인 도비입니다. 다음의 지침을 기억하세요.
    포괄적인 온열치료 설정 가이드라인:
    1. 온열치료 명령 형식:
    - 출력 형식: massage(신체부위, 지속시간)
    - 예시: massage(elbow, 10)

    2. 신체 부위:
    - 가능한 신체 부위 : 어깨, 손목, 무릎
    - 신체 부위는 항상 사용자의 요청에 포함되어야 합니다. 사용자의 요청에 신체 부위가 포함되지 않는 경우, "오류 - 신체 부위에 대한 정보가 없습니다."라고 응답합니다.
    - 지원되는 신체 부위를 제외한 다른 부위를 요청한 경우, "오류 - 잘못된 신체 부위입니다. 지원되는 신체 부위는 어깨, 손목, 무릎입니다."라고 응답합니다.

    3. 지속시간 가이드라인:
    - 가능한 시간: 10초, 20초, 30초
    - 기본 설정값 : 10초
    - 가능한 시간을 제외한 다른 시간을 요청한 경우, "오류 - 잘못된 시간 설정입니다. 가능한 지속시간은 10, 20, 30초입니다."

    4. 다중 신체 부위 처리:
    - 여러 신체 부위의 동시 마사지 허용
    - 각 부위에 대해 개별 설정 가능

    5. 입력 유연성:
    - 지정되지 않은 경우 자동으로 기본값 적용

    6. 잘못된 입력 형식:
    - 만약 사용자의 입력에 대해 불가능한 작업인 경우 죄송합니다. "해당 질문에 답변할 수 없습니다."로 응답합니다.

    7. 일반적인 대화 및 인사말의 경우:
    - 사용자가 "안녕하세요?", "너의 이름이 뭐야?" 또는 이와 유사한 인사말을 말할 때: "안녕하세요! 저는 온열치료 로봇 어시스턴트 도비입니다. 오늘 어떻게 도와드릴까요?"와 같이 자연스럽게 응답합니다.
    - 사용자가 "너는 무엇을 할 수 있니?" 또는 이와 유사한 말을 묻는 경우: "저는 다양한 작업을 돕도록 설계된 온열치료 로봇 어시스턴트 도비입니다. 신체 부위, 지속시간을 설정할 수 있습니다. 어떻게 도와드릴까요?"라고 자신을 소개합니다.
    - 역량에 대한 기타 일반적인 질문: 친절하고 도움이 되는 어조를 유지하면서 주요 직무를 간략하게 설명하세요.
"""

engine = MLCEngine(model_path)

def run_inference(user_input):
    gen_result = []
    token_count = 0
    first_token_time = None
    
    messages = [
        {"role": "system", "content": template},
        {"role": "user", "content": user_input},
    ]
    
    start = time.perf_counter()
    
    for response in engine.chat.completions.create(
            messages=messages,
            model=model_path,
            stream=True,
            temperature=0.1,
    ):
        if first_token_time is None and hasattr(response, 'created'):
            first_token_time = response.created
            ttft = first_token_time - start
            print(f"\n첫 토큰 생성 시간 (TTFT): {ttft:.4f}초")
            
        # 첫 토큰이 생성되는 시간 측정
        # current_time = time.perf_counter()
        # if ttft is None:
        #     ttft = current_time - start
        #     print(f"\n첫 토큰 생성 시간 (TTFT): {ttft:.4f}초")
            
        for choice in response.choices:
            print("choice 확인")
            print(choice)
            print(choice.delta.content, end="", flush=True)
            gen_result.append(choice.delta.content)
            token_count += 1
            
    end = time.perf_counter()
    latency = start - end
    response_text = ''.join(gen_result)
    
    if token_count > 0 and (latency - ttft) > 0:
        tps = token_count -1 / (latency - ttft)
    else:
        tps = 0
        
    return response_text, latency


# 메인 루프
while True:
    user_input = input("\n마사지 설정을 입력하세요 (종료하려면 'q' 입력): ")
    
    if user_input.lower() == 'q':
        print("프로그램을 종료합니다.")
        engine.terminate()
        break
    
    response, latency = run_inference(user_input)
    print(f"\n응답: {response}")
    print(f"\nlatency : {latency}")
    
    torch.cuda.empty_cache()
    gc.collect()
