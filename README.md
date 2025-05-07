# 개선방향 : 인덱싱부터 제대로 처리 필요 
# RAG , GRAPH 처리 방식 이해 필요 

# AWS Resource Indexer & Query System

이 프로젝트는 **AWS 리소스 전체를 자동 수집**하고, **리소스 간의 관계까지 추적**하여, 
로컬 임베딩(벡터) 모델과 LlamaIndex를 활용해 **자연어로 AWS 환경을 질의/검색/요약**할 수 있는 시스템입니다.

---

## 주요 개념

### 1. **AWS 리소스 자동 수집**
- Resource Groups Tagging API와 각 서비스별 boto3 API를 활용해
  - EC2, VPC, Subnet, Security Group, RDS, S3 등 **모든 리소스**를 수집합니다.
- **태그 없는 VPC**도 region별 describe_vpcs()로 누락 없이 수집합니다.

### 2. **리소스 간 관계 추적**
- VPC-Subnet-Instance-SG-RDS 등 **AWS 리소스 간의 실제 관계**를 추적/요약합니다.
- 예: 어떤 VPC에 어떤 인스턴스/서브넷/보안그룹이 연결되어 있는지 등

### 3. **LlamaIndex Document 변환**
- 각 리소스/요약/관계 정보를 LlamaIndex의 Document 객체로 변환합니다.
- Document의 `text`에는 상세 정보(JSON), `metadata`에는 요약/식별/관계 정보가 담깁니다.

### 4. **로컬 임베딩(벡터) 인덱스 구축**
- HuggingFace 임베딩 모델로 각 Document를 벡터로 변환하여 인덱스(index 폴더)에 저장합니다.
- LlamaIndex를 사용해 벡터 인덱스를 구축합니다.

### 5. **자연어 질의 & RAG 구조**
- 사용자의 자연어 질문을 임베딩(벡터)로 변환 → 인덱스에서 관련 문서 검색 →
  검색된 문서를 LLM(로컬/클라우드)에 프롬프트로 전달 → LLM이 답변 생성
- 이 구조를 **RAG(Retrieval Augmented Generation)**라고 부릅니다.

---

## 폴더/파일 구조

```
aws_resources/
├── data/
│   ├── aws_resources.json         # 모든 리소스의 상세 정보/관계/요약 (Document 리스트)
│   └── resources_summary.json     # 리소스 개수, 서비스별/타입별/관계별 요약
├── index/
│   └── ... (LlamaIndex 벡터 인덱스 파일)
├── logs/
│   └── aws_indexer.log           # 실행 로그
```

- **aws_indexer_embedding.py**: 리소스 수집/관계 추적/Document 변환/인덱스 구축 스크립트
- **aws_query.py**: 인덱스에 자연어로 질의하고, LLM을 통해 답변 생성

---

## 동작 원리 요약

1. **리소스 수집/인덱싱**
   - `python aws_indexer_embedding.py`
   - AWS 리소스 전체 수집 → 관계 추적 → Document 변환 → 인덱스/데이터 저장

2. **자연어 질의**
   - `python aws_query.py --query "내 VPC 목록을 알려줘"`
   - 질문 임베딩 → 벡터 인덱스에서 관련 문서 검색 → LLM에 프롬프트로 전달 → 답변 생성

---

## 주요 개념 설명

### ● **Document & Metadata**
- 각 AWS 리소스/요약/관계 정보는 Document 객체로 저장
- `text`: 상세 정보(JSON)
- `metadata`: 식별자, 타입, 리전, 관계 등 요약 정보

### ● **벡터 인덱스(Vector Index)**
- Document의 text를 임베딩 모델로 벡터화하여 인덱스에 저장
- 질문도 벡터로 변환, 인덱스에서 유사도 기반으로 관련 문서 검색

### ● **RAG 구조**
- Retrieval Augmented Generation
- "질문 → 벡터 검색 → 관련 문서 추출 → LLM 프롬프트로 전달 → 답변 생성"
- LLM은 전체 데이터가 아니라, 검색된 문서만 참고

### ● **관계 추적(Relationships)**
- VPC, Subnet, SG, RDS 등 리소스 간의 연결/소속 관계를 metadata와 요약 문서에 포함
- 예: "이 인스턴스가 속한 VPC/서브넷/SG는?", "VPC별 리소스 분포는?" 등 질의에 강함

---

## 사용 예시

1. **리소스 인덱싱**
   ```bash
   python aws_indexer_embedding.py
   ```
2. **자연어 질의**
   ```bash
   python aws_query.py --query "내 VPC 목록을 알려줘"
   ```

---

## 참고/한계
- LLM의 답변 품질은 문서 구조, 프롬프트, 임베딩 모델에 따라 달라질 수 있습니다.
- 벡터 검색은 의미 기반이므로, 정확한 수치/구조 정보는 후처리나 프롬프트 개선이 필요할 수 있습니다.
- Document의 text/metadata 구조를 개선하면 검색/질의 품질이 더 좋아집니다.




