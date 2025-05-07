#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AWS 리소스 질의 스크립트
- 로컬 임베딩 모델을 사용하여 저장된 인덱스에 질의
- Name 태그가 없는 VPC도 CIDR 블록 정보를 사용하여 식별
- 대화형 모드 또는 커맨드라인 인수 모드 지원
- 로컬 또는 원격 LLM 선택 가능
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path

# LlamaIndex 관련 패키지
from llama_index.core import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 기본 폴더 설정
BASE_DIR = Path("aws_resources")
INDEX_DIR = BASE_DIR / "index"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# 필요한 디렉토리 존재 확인
if not BASE_DIR.exists():
    print(f"경고: 기본 디렉토리 {BASE_DIR}가 존재하지 않습니다. aws_indexer_embedding.py를 먼저 실행해주세요.")
    BASE_DIR.mkdir(exist_ok=True)
    
LOGS_DIR.mkdir(exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "aws_query.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """OpenAI API를 사용하는 래퍼 클래스"""
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        try:
            from openai import OpenAI
            self.api_key = api_key
            self.model_name = model
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI 클라이언트 초기화 완료 (모델: {model})")
        except ImportError:
            logger.error("openai 패키지가 설치되어 있지 않습니다. 'pip install openai' 명령으로 설치해주세요.")
            raise

    def complete(self, prompt):
        """텍스트 완성 메서드"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            return f"오류 발생: {e}"

class LocalLLM:
    """로컬 LLM을 사용하는 래퍼 클래스"""
    def __init__(self, model_url="http://localhost:8000/v1", model_name="llama3"):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=model_url, api_key="not-needed")
            self.model_name = model_name
            logger.info(f"로컬 LLM 클라이언트 초기화 완료 (URL: {model_url}, 모델: {model_name})")
        except ImportError:
            logger.error("openai 패키지가 설치되어 있지 않습니다. 'pip install openai' 명령으로 설치해주세요.")
            raise

    def complete(self, prompt):
        """텍스트 완성 메서드"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"로컬 LLM API 오류: {e}")
            return f"오류 발생: {e}"

def setup_embed_model():
    """로컬 임베딩 모델을 초기화하고 전역 설정에 적용합니다."""
    try:
        # 로컬 임베딩 모델 사용
        # 옵션 1: 경량 모델 (빠름)
        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        
        # 옵션 2: 중간 크기 모델 (균형잡힌 성능)
        # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 옵션 3: 대형 모델 (높은 성능, 느림)
        # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
        
        # 전역 설정에 모델 지정
        Settings.embed_model = embed_model
        logger.info("로컬 임베딩 모델 설정 완료")
    except Exception as e:
        logger.error(f"임베딩 모델 설정 중 오류: {e}")
        raise

def load_index(persist_dir):
    """저장된 인덱스를 로드합니다."""
    try:
        logger.info(f"{persist_dir}에서 인덱스 로드 중")
        
        # 임베딩 모델 설정
        setup_embed_model()
        
        # 인덱스 로드를 위한 스토리지 컨텍스트 설정
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        
        # 인덱스 로드
        index = load_index_from_storage(storage_context)
        logger.info("인덱스 로드 완료")
        return index
    except Exception as e:
        logger.error(f"인덱스 로드 중 오류 발생: {e}")
        return None

def format_document_content(doc, idx):
    """문서 내용을 보기 좋게 포맷팅합니다."""
    try:
        # JSON 문자열 파싱
        content = json.loads(doc.get_content())
        
        # 메타데이터 추출
        metadata = doc.metadata
        doc_type = metadata.get("document_type", "unknown")
        
        # 문서 유형에 따라 다르게 포맷팅
        if doc_type == "summary":
            return f"요약 문서 {idx}:\n{json.dumps(content, indent=2, ensure_ascii=False)}"
        elif doc_type == "service_summary":
            service = metadata.get("service", "unknown")
            return f"{service} 서비스 요약 {idx}:\n{json.dumps(content, indent=2, ensure_ascii=False)}"
        else:
            # 리소스 문서의 경우 중요 정보 추출하여 보기 좋게 포맷팅
            service = content.get("service", "unknown")
            resource_type = content.get("resource_type", "unknown")
            resource_id = content.get("resource_id", "unknown")
            
            # 리소스 유형에 따른 특별 처리
            if service == "ec2" and resource_type == "vpc":
                # VPC의 경우 Name 태그가 없으면 CIDR 블록과 ID를 조합
                tags = content.get("tags", {})
                details = content.get("details", {})
                cidr = details.get("CidrBlock", "알 수 없음")
                name = tags.get("Name", f"VPC-{cidr} ({resource_id})")
                return f"리소스 {idx} - {service}:{resource_type}:{name}:\n{json.dumps(content, indent=2, ensure_ascii=False)}"
            else:
                # 일반적인 경우
                tags = content.get("tags", {})
                name = tags.get("Name", resource_id)
                return f"리소스 {idx} - {service}:{resource_type}:{name}:\n{json.dumps(content, indent=2, ensure_ascii=False)}"
    except Exception as e:
        logger.error(f"문서 내용 포맷팅 중 오류: {e}")
        # 오류 발생 시 원본 내용 그대로 반환
        return f"문서 {idx}:\n{doc.get_content()}"

def query_aws_resources(query_text, index, model_type, model_params, max_docs=10):
    """AWS 리소스 인덱스에 질의합니다."""
    try:
        logger.info(f"질의 시작: {query_text}")
        
        # VPC 관련 질문인지 확인
        vpc_related = any(keyword in query_text.lower() for keyword in ['vpc', '가상 사설 클라우드', '네트워크', 'cidr'])
        
        # 1. 검색 수행
        retriever = index.as_retriever(similarity_top_k=max_docs)
        relevant_docs = retriever.retrieve(query_text)
        logger.info(f"{len(relevant_docs)}개의 관련 문서 검색됨")
        
        # 2. 문서 분류
        summary_docs = []
        resource_docs = []
        vpc_docs = []  # VPC 문서를 따로 분류
        
        for doc in relevant_docs:
            metadata = doc.metadata
            doc_type = metadata.get("document_type", "")
            
            # VPC 리소스 특별 처리
            service = metadata.get("service", "")
            resource_type = metadata.get("resource_type", "")
            
            if service == "ec2" and resource_type == "vpc":
                vpc_docs.append(doc)
            elif doc_type in ["summary", "service_summary"]:
                summary_docs.append(doc)
            else:
                resource_docs.append(doc)
        
        logger.info(f"요약 문서: {len(summary_docs)}개, VPC 문서: {len(vpc_docs)}개, 기타 리소스 문서: {len(resource_docs)}개")
        
        # VPC 관련 질문이고 VPC 문서가 없는 경우, 추가 검색 시도
        if vpc_related and not vpc_docs:
            logger.info("VPC 관련 질문이지만 VPC 문서가 없습니다. VPC 특화 검색을 시도합니다.")
            try:
                # VPC 특화 검색 쿼리 생성
                vpc_query = "VPC CIDR network AWS"
                vpc_retriever = index.as_retriever(similarity_top_k=3)
                additional_docs = vpc_retriever.retrieve(vpc_query)
                
                # VPC 문서만 필터링
                for doc in additional_docs:
                    metadata = doc.metadata
                    service = metadata.get("service", "")
                    resource_type = metadata.get("resource_type", "")
                    
                    if service == "ec2" and resource_type == "vpc":
                        # 중복 방지
                        if doc.doc_id not in [d.doc_id for d in vpc_docs]:
                            vpc_docs.append(doc)
                
                logger.info(f"VPC 특화 검색으로 {len(vpc_docs)}개의 VPC 문서를 찾았습니다.")
            except Exception as e:
                logger.error(f"VPC 특화 검색 중 오류 발생: {e}")
        
        # 3. 문서 내용 포맷팅
        doc_texts = []
        
        # 요약 문서 추가
        if summary_docs:
            summary_texts = []
            for i, doc in enumerate(summary_docs):
                summary_texts.append(format_document_content(doc, i+1))
            
            doc_texts.append("# 요약 정보\n" + "\n\n".join(summary_texts))
        
        # VPC 문서 추가 (VPC를 우선적으로 표시)
        if vpc_docs:
            vpc_texts = []
            for i, doc in enumerate(vpc_docs):
                vpc_texts.append(format_document_content(doc, i+1))
            
            doc_texts.append("# VPC 정보\n" + "\n\n".join(vpc_texts))
        
        # 기타 리소스 문서 추가
        if resource_docs:
            resource_texts = []
            for i, doc in enumerate(resource_docs):
                resource_texts.append(format_document_content(doc, i+1))
            
            doc_texts.append("# 기타 리소스 정보\n" + "\n\n".join(resource_texts))
        
        # 문서를 줄바꿈으로 결합
        docs_content = "\n\n".join(doc_texts)
        
        # 4. 프롬프트 생성
        prompt = f"다음은 AWS 리소스 정보에 관한 질문입니다: \"{query_text}\"\n\n"
        prompt += "아래 정보를 바탕으로 자세히 답변해주세요:\n\n"
        prompt += docs_content + "\n\n"
        prompt += "답변 형식:\n"
        prompt += "1. 명확하고 정확하게 응답해주세요.\n"
        prompt += "2. 질문에 관련된 AWS 리소스 정보만 포함해주세요.\n"
        prompt += "3. 기술적으로 올바른 정보를 제공해주세요.\n"
        prompt += "4. JSON 데이터에서 필요한 정보를 추출하여 이해하기 쉽게 정리해주세요.\n"
        prompt += "5. 다양한 리소스 타입(EC2, S3, VPC, RDS 등)의 정보를 균형있게 포함해주세요.\n"
        prompt += "6. 특정 리소스 타입에만 편중되지 않도록 해주세요.\n"
        prompt += "7. 전체 AWS 환경의 포괄적인 개요를 제공해주세요.\n"
        prompt += "8. VPC 관련 질문에는 반드시 Name 태그가 없는 VPC도 CIDR 블록과 ID 정보를 사용하여 식별해 응답해주세요.\n"
        prompt += "9. 특히 중요한 점: 제공된 문서에서 관련 정보를 찾지 못한 경우, 정직하게 해당 정보가 없다고 말하세요."
        
        # VPC 관련 질문인 경우 추가 지시사항
        if vpc_related:
            prompt += "\n10. 이 질문은 VPC와 관련된 질문입니다. VPC 정보를 특별히 자세히 설명해주세요.\n"
            prompt += "11. Name 태그가 없는 VPC의 경우 CIDR 블록과 ID를 함께 표시하여 VPC를 명확히 식별할 수 있게 해주세요.\n"
            prompt += "12. VPC에 연결된 서브넷, 라우팅 테이블, 보안 그룹 등의 관련 정보도 포함해주세요."
        
        # 5. 모델 유형에 따른 응답 생성
        if model_type == "openai":
            # OpenAI API 사용
            api_key = model_params.get("api_key")
            if not api_key:
                logger.warning("경고: OpenAI API 키가 설정되지 않았습니다.")
                return "OpenAI API 키가 설정되지 않았습니다. API 키를 설정하거나 다른 모델을 사용해주세요."
            
            model = model_params.get("model", "gpt-3.5-turbo")
            openai_client = OpenAIClient(api_key=api_key, model=model)
            response = openai_client.complete(prompt)
            
        elif model_type == "local":
            # 로컬 LLM 사용
            model_url = model_params.get("model_url", "http://localhost:11434/v1")
            model_name = model_params.get("model_name", "llama3")
            
            local_llm = LocalLLM(model_url=model_url, model_name=model_name)
            response = local_llm.complete(prompt)
            
        else:
            # 지원되지 않는 모델
            logger.error(f"지원되지 않는 모델 유형: {model_type}")
            return f"지원되지 않는 모델 유형: {model_type}. 현재 지원되는 모델: openai, local"
        
        logger.info("질의 응답 완료")
        return response
    except Exception as e:
        logger.error(f"질의 중 오류 발생: {e}")
        return f"질의 처리 중 오류가 발생했습니다: {e}"

def interactive_mode(index, model_type, model_params, max_docs=10):
    """대화형 모드로 질의를 처리합니다."""
    print("\n=== AWS 리소스 질의 대화형 모드 ===")
    print("질문을 입력하시거나, 'exit', 'quit', 'q'를 입력하여 종료하세요.")
    print("특수 명령어:")
    print("  vpc: VPC 리소스를 우선적으로 검색합니다.")
    print("  summary: 전체 AWS 환경 요약을 보여줍니다.")
    print("  help: 도움말을 표시합니다.\n")
    
    while True:
        query = input("\n질문을 입력하세요 (종료: q) > ")
        
        # 종료 명령어
        if query.lower() in ['exit', 'quit', 'q']:
            print("대화형 모드를 종료합니다.")
            break
        
        # 특수 명령어 처리
        if query.lower() == 'help':
            print("\n=== 도움말 ===")
            print("- 일반 질문: AWS 리소스에 대한 질문을 자유롭게 입력하세요.")
            print("- 특수 명령어:")
            print("  vpc: VPC 리소스를 우선적으로 검색합니다.")
            print("  summary: 전체 AWS 환경 요약을 보여줍니다.")
            print("  help: 이 도움말을 표시합니다.")
            print("  exit, quit, q: 프로그램을 종료합니다.")
            print("- 팁: VPC 관련 질문에서는 Name 태그가 없는 VPC도 CIDR 블록으로 식별됩니다.")
            continue
        elif query.lower() == 'vpc':
            query = "모든 VPC 정보를 보여주고 CIDR 블록으로 구분해서 설명해줘"
        elif query.lower() == 'summary':
            query = "전체 AWS 환경의 모든 리소스를 요약해줘"
        
        print("\n처리 중...\n")
        response = query_aws_resources(query, index, model_type, model_params, max_docs)
        print(f"답변:\n{response}\n")
        print("-" * 80)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AWS 리소스 질의 스크립트")
    
    parser.add_argument(
        "--index-dir", "-i", 
        default=None,
        help="인덱스가 저장된 디렉토리 (기본값: aws_resources/index)"
    )
    
    parser.add_argument(
        "--model", "-m", 
        default="local",
        choices=["openai", "local"],
        help="사용할 AI 모델 (기본값: local)"
    )
    
    parser.add_argument(
        "--openai-key", "-k", 
        help="OpenAI API 키 (모델이 'openai'인 경우 필요)"
    )
    
    parser.add_argument(
        "--openai-model", 
        default="gpt-3.5-turbo",
        help="OpenAI 모델 이름 (기본값: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--local-url", 
        default="http://localhost:8000/v1",
        help="로컬 LLM API URL (기본값: http://localhost:8000/v1)"
    )
    
    parser.add_argument(
        "--local-model", 
        default="llama3",
        help="로컬 LLM 모델 이름 (기본값: llama3)"
    )
    
    parser.add_argument(
        "--query", "-q", 
        help="직접 질의할 텍스트. 생략 시 대화형 모드로 실행됩니다."
    )
    
    parser.add_argument(
        "--max-docs", "-d", 
        type=int,
        default=10,
        help="검색할 최대 문서 수 (기본값: 10)"
    )
    
    parser.add_argument(
        "--vpc-focus", "-v",
        action="store_true",
        help="VPC 관련 정보에 초점을 맞춥니다."
    )
    
    args = parser.parse_args()
    
    # 인덱스 디렉토리 설정
    index_dir = Path(args.index_dir) if args.index_dir else INDEX_DIR
    
    # 인덱스 디렉토리 확인
    if not index_dir.exists():
        print(f"오류: 인덱스 디렉토리 {index_dir}가 존재하지 않습니다.")
        print(f"먼저 aws_indexer_embedding.py를 실행하여 인덱스를 생성해야 합니다.")
        return 1
    
    # 인덱스 로드
    print(f"인덱스를 로드하는 중: {index_dir}")
    index = load_index(str(index_dir))
    if index is None:
        print(f"오류: 인덱스를 로드할 수 없습니다.")
        return 1
    
    # 모델 파라미터 설정
    model_params = {}
    if args.model == "openai":
        # OpenAI API 키 확인
        api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("경고: OpenAI API 키가 설정되지 않았습니다.")
            print("--openai-key 옵션을 사용하거나 OPENAI_API_KEY 환경 변수를 설정해주세요.")
            return 1
        
        model_params = {
            "api_key": api_key,
            "model": args.openai_model
        }
    else:  # local
        # sentence-transformers 패키지 확인
        try:
            import sentence_transformers
        except ImportError:
            print("경고: sentence-transformers 패키지가 설치되어 있지 않습니다.")
            print("pip install sentence-transformers 명령을 실행하여 패키지를 설치한 후 다시 시도해주세요.")
            return 1
        
        # openai 패키지 확인
        try:
            import openai
        except ImportError:
            print("경고: openai 패키지가 설치되어 있지 않습니다.")
            print("pip install openai 명령을 실행하여 패키지를 설치한 후 다시 시도해주세요.")
            return 1
        
        model_params = {
            "model_url": args.local_url,
            "model_name": args.local_model
        }
    
    # 쿼리 모드 선택
    if args.query:
        # VPC 초점 옵션 처리
        if args.vpc_focus:
            if "vpc" not in args.query.lower():
                args.query = f"VPC 관련 정보: {args.query}"
        
        # 단일 질의 모드
        print(f"질문: {args.query}")
        response = query_aws_resources(args.query, index, args.model, model_params, args.max_docs)
        print(f"답변:\n{response}")
    else:
        # 대화형 모드
        try:
            interactive_mode(index, args.model, model_params, args.max_docs)
        except KeyboardInterrupt:
            print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
