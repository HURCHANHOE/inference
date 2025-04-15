import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
# from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import asyncio
import json
import time
from transformers import AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

# 1. 문서 로드
def load_documents(directory_path: str):
    try:
        for filename in os.listdir(directory_path):
            documents = PyPDFLoader(directory_path + '/' + filename).load()
        print(f"총 {len(documents)}개의 문서를 로드했습니다.")
        return documents
    except Exception as e:
        print(f"문서 로드 중 오류 발생: {e}")
        return []

# 2. 문서 청킹
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"총 {len(chunks)}개의 청크로 분할했습니다.")
    return chunks

# 3. 벡터 DB 생성 및 저장
def create_and_save_vectordb(chunks, embedding_model, persist_directory):
    # 임베딩 모델 초기화
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    embeddings = Llama(model_path=embedding_model, embedding=True)
    collection_name="baseball_rule"
    # ChromaDB 생성
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(name=collection_name, embedding_function=embeddings)
        print(f"기존 컬렉션을 로드했습니다: {collection_name}")
    except:
        collection = client.create_collection(name=collection_name, embedding_function=embeddings)
        print(f"새 컬렉션을 생성했습니다: {collection_name}")

    if chunks:
        ids = [f"doc_{i}" for i in range(len(chunks))]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # 청크를 컬렉션에 추가
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"총 {len(chunks)}개의 청크가 벡터 DB에 저장되었습니다.")
    # 변경사항 저장
    # client.persist()
    print(f"벡터 DB가 {persist_directory}에 저장되었습니다.")
    return collection

# 4. 기존 벡터 DB 로드
def load_vectordb(embedding_model, persist_directory):
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    embeddings = Llama(model_path=embedding_model, embedding=True)
    collection_name="baseball_rule"
    client = chromadb.PersistentClient(path=persist_directory)
    
    collection = client.get_collection(name=collection_name, embedding_function=embeddings)
    return collection

# 5. VLLM을 사용한 추론
class LlamaCPPInference:
    def __init__(self, model_name):
        self.model = Llama(
            model_path=model_name,
            n_ctx=2048,
            n_gpu_layers=-1,
            temperature=0.1,
            stop=["\n\n", "\n"],
            f16_kv=True,
            verbose=False,
            chat_format="llama-3",
            draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10, max_ngram_size=1), # num_pred_tokens is the number of tokens to predict 10 is the default and generally good for gpu, 2 performs better for cpu-only machines.
            logits_all=True,
        )
    
    def generate_response(self, query, prompt, **kwargs) :
        # input값 구성
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        tokens = self.model.tokenize(formatted_messages.encode('utf-8'))
        context_length = len(tokens)

        print(f"context length 확인 : {context_length}")

        # 추론
        first_token_time = None
        gen_result = []
        token_count = 0

        kwargs['stream'] = True
        start = time.perf_counter()
        for chunk in self.model.create_chat_completion(messages=messages, **kwargs):
            current_time = time.perf_counter()
            if first_token_time is None:
                first_token_time = current_time - start
            if 'choices' in chunk and chunk['choices'][0].get('delta', {}).get('content'):
                content = chunk['choices'][0]['delta']['content']
                print(content, end='', flush=True)
                gen_result.append(content)

        end = time.perf_counter()
        latency = end - start
        response_text = ''.join(gen_result)
        token_count = len(gen_result)
        print(f"gen_result:{gen_result}")
        if token_count > 1 and (latency - first_token_time) > 0:
            tps = token_count / (latency - first_token_time)
        else:
            tps = 0

        return response_text, latency, first_token_time, tps, token_count
    
# 6. RAG(Retrieval-Augmented Generation) 파이프라인
def rag_pipeline(query: str, collection, llama_cpp_inference):
    # 유사한 문서 검색
    k=3
    logger.info("리트리버 문서 시작")
    retriever_start = time.perf_counter()
    retrieved_docs = collection.query(
        query_texts=[query],
        n_results=k,
    )
    retriever_end = time.perf_counter()
    logger.info("리트리버 문서 끝")
    print(f"리트리버 걸린 시간 확인: {retriever_end - retriever_start}")
    print(f"retrieved_docs:{retrieved_docs}")
    # 리트리버 컨텍스트 구성
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # print(f"context 길이 확인:{len(context)}")
    # 프롬프트 구성
    prompt = f"""
{context}

당신은 사용자의 질문에 대해 친절하고 답변하는 챗봇입니다. 위의 정보를 바탕으로 다음 지침을 준수하여 답변하세요.

가이드라인:
1. 당신은 대화를 자연스럽게 유지하기 위해 과거 주제를 기억합니다.
2. 정확하지 않은 정보는 아는 척하지 말고, 솔직하게 모른다고 하거나 자연스럽게 대화를 다른 방향으로 >유도해야 합니다. 절대 추측하지 >말고 사실처럼 말하지 마세요.
"""
    # print(f"프롬프트 길이 확인:{len(prompt)}")

    # 응답 생성
    response, latency, ttft, tps, token_count = llama_cpp_inference.generate_response(query, prompt)
    return response, retriever_end - retriever_start, latency, ttft, tps, token_count

# 메인 실행 코드
if __name__ == "__main__":
    # 문서 디렉토리 경로 (langchain 문서가 있는 경로로 변경)
    DOCS_DIR = "./langchain_docs"
    PERSIST_DIR = "./chroma_db"

    # 모델 정의
    embedding_model_name="/home/models/bge-m3"
    embedding_model_name="/home/models/bge-m3-q8_0.gguf"
    base_model_name = "/home/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    # base_model_name="/home/models/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
    # base_model_name = "/home/models/EXAONE-Deep-7.8B-Q8_0.gguf"
    # base_model_name = "/home/models/Phi-4-mini-instruct.Q8_0.gguf"
    
    # 벡터 DB가 이미 존재하는지 확인
    if not os.path.exists(PERSIST_DIR):
        # 문서 로드 및 처리
        logger.info("문서 로드 시작")
        documents = load_documents(DOCS_DIR)

        logger.info("문서 청킹 시작")
        chunks = chunk_documents(documents)

        logger.info("벡터 스토어 생성 및 저장 시작")
        collection = create_and_save_vectordb(chunks, embedding_model_name, persist_directory=PERSIST_DIR)
    else:
        # 기존 벡터 DB 로드
        print(f"기존 벡터 DB를 {PERSIST_DIR}에서 로드합니다.")
        collection = load_vectordb(embedding_model_name, persist_directory=PERSIST_DIR)

    # VLLM 추론 엔진 초기화 (모델명은 필요에 따라 변경)
    logger.info("llama cpp 엔진 로드 시작")
    llama_cpp_inference = LlamaCPPInference(model_name=base_model_name)

    # 대화형 모드
    # while True:
    #     user_query = input("\n질문을 입력하세요 (종료하려면 'q' 입력): ")
    #     if user_query.lower() == 'q':
    #         break
    user_inputs=[
        "안녕하세요 두산베어스의 감독은 누구야", #warm up
        "한경기에 비디오 판독을 몇번 쓸 수 있나요?",
        "정규시즌은 총 몇경기를 치루나요?",
        "토요일에 경기를 보러가고 싶은데 언제 시작하는지 알 수 있나요?",
        "몇위까지 가을야구에 진출하나요?",
        "우최되면 티켓은 환불받을 수 있는지 알려줘줘",
        "두산베어스의 단장은 누구야야",
        "준플레이오프 경기는 총 몇경기인가요?",
        "올스타전 선정 기준이 어떻게 되나요?",
        "올스타전은 언제 해요?",
        "abs 판정에 불만을 가질경우 항의는 어떻게 하나요?"
    ]

    retriever_times = []
    latencies = []
    ttfts = []
    tps_values = []
    total_tokens = []

    # 모든 쿼리에 대해 파이프라인 실행
    for i, user_query in enumerate(user_inputs):
        result, retriever_time, latency, ttft, tps, total_token = rag_pipeline(user_query, collection, llama_cpp_inference)

        # 첫 번째 항목("warm up")은 건너뛰기
        if i > 0:  # i가 0보다 크면 첫 번째 항목이 아님
            retriever_times.append(retriever_time)
            latencies.append(latency)
            ttfts.append(ttft)
            tps_values.append(tps)
            total_tokens.append(total_token)

    # 평균 계산
    avg_retriever_time = sum(retriever_times) / len(retriever_times) if retriever_times else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0
    avg_total_token = sum(total_tokens) / len(total_tokens) if total_tokens else 0

    # 결과 출력
    print(f"평균 검색 시간: {avg_retriever_time:.4f}초")
    print(f"평균 지연 시간: {avg_latency:.4f}초")
    print(f"평균 첫 토큰 생성 시간(TTFT): {avg_ttft:.4f}초")
    print(f"평균 초당 토큰 수(TPS): {avg_tps:.2f}")
    print(f"평균 총 토큰 수: {avg_total_token:.1f}")
