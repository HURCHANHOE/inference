import json
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util
import torch
import gc

# 유사도 검사 function
def analyze_pairwise_similarity(result_text, expected_output, st_model):
    # 임베딩 생성
    result_embedding = st_model.encode(result_text)
    output_embedding = st_model.encode(expected_output)
    
    # 유사도 계산
    similarity = util.pytorch_cos_sim(result_embedding, output_embedding)[0][0]
    
    print(f"생성된 답변: {result_text}")
    print(f"예상 답변: {expected_output}")
    print(f"유사도: {similarity:.4f}")
    
    return similarity

def load_model():
    
    model_path = "/app/models/massage_8B_v4.gguf"
    # model_path = "/app/models/massage_7B_v1.gguf"

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        temperature=0.1,
        stop=["\n\n", "\n"],
        f16_kv=True,
        verbose=True,
        chat_format="llama-3",
    )
    return llm

def generate(llm, user_input):
    PROMPT_TEMPLATE = """당신은 사용자의 온열치료 요청을 정확하게 해석하고 여러 신체 부위에 대한 온열치료 설정을 도와주는 보조 로봇인 도비입니다. 다음의 지침을 기억하세요.
    포괄적인 온열치료 설정 가이드라인:
    1. 온열치료 명령 형식:
    - 출력 형식: massage(신체부위, 지속시간)
    - 예시: massage(elbow, 1)

    2. 신체 부위:
    - 가능한 신체 부위 : 어깨, 손목, 무릎
    - 신체 부위는 항상 사용자의 요청에 포함되어야 합니다. 사용자의 요청에 신체 부위가 포함되지 않는 경우, "오류 - 신체 부위에 대한 정보가 없습니다."라고 응답합니다.
    - 지원되는 신체 부위를 제외한 다른 부위를 요청한 경우, "오류 - 잘못된 신체 부위입니다. 지원되는 신체 부위는 어깨, 손목, 무릎입니다."라고 응답합니다.

    3. 지속시간 가이드라인:
    - 가능한 시간: 1분, 2분, 3분
    - 기본 설정값 : 1분
    - 가능한 시간을 제외한 다른 시간을 요청한 경우, "오류 - 잘못된 시간 설정입니다. 가능한 지속시간은 1, 2, 3분 입니다."

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

    result = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    return result

if __name__ == "__main__":
    model = load_model()
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    results = []
    similarities = []

    with open('/app/dataset/massage_v2_test.json', 'r') as js:
        test_data = json.load(js)
    
    # for user_input in user_inputs:t
    for doc in test_data:
        user_input = doc['human']
        label_output = doc['output']
        result = generate(model, user_input)
        result_text = result["choices"][0]['message']['content']
        
        if "```" in label_output:
            if "```" in result_text:
                function_response = result_text.split("```")[1]
                if function_response == label_output.split("```")[1] :
                    similarity=1
                else:
                    similarity=0
            else:
                similarity=0
        else:
            if "```" in result_text:
                similarity=0
            else:
                # 유사도 분석 및 저장
                similarity_tensor = analyze_pairwise_similarity(result_text, label_output, st_model)
                similarity = similarity_tensor.item()
                similarities.append(similarity)
            
        # Create a dictionary for each iteration
        result_dict = {
            "input": user_input,
            "output": result_text,
            "label": label_output,
            "similarity" : similarity
        }
        results.append(result_dict)
        
        # Print for monitoring
        print(f'\ninput::{user_input}\noutput::{result_text}\nlabel::{label_output}')
    
    # Save results to JSON file
    output_path = '/app/result/massage_llama_results2.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to {output_path}")
    
    del model, st_model
    torch.cuda.empty_cache()
    gc.collect()
