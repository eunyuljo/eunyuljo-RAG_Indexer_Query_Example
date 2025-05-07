#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AWS 리소스 수집 및 인덱싱 스크립트
- Resource Groups Tagging API를 사용하여 모든 AWS 리소스를 수집
- 수집된 리소스를 LlamaIndex Document 객체로 변환
- 리소스 간 연관 관계 추적 및 문서에 포함
- 로컬 임베딩 모델을 사용하여 벡터 인덱스를 구축
- 로컬에서 자연어 질의 처리
"""

import boto3
import json
import os
import shutil
from collections import defaultdict
import logging
import sys
from pathlib import Path

# LlamaIndex 관련 패키지
from llama_index.core import Document, Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 기본 폴더 설정
BASE_DIR = Path("aws_resources")
INDEX_DIR = BASE_DIR / "index"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# 필요한 디렉토리 생성
BASE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "aws_indexer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 서비스 이름 매핑 테이블 추가
SERVICE_CLIENT_MAP = {
    "ec2": "ec2",
    "s3": "s3",
    "rds": "rds",
    "dynamodb": "dynamodb",
    "lambda": "lambda",
    "elasticfilesystem": "efs",
    "elasticloadbalancing": "elbv2",  # Application/Network Load Balancer. 필요시 'elb'도 추가 가능
    # 필요시 추가 서비스 매핑
}

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

def get_all_vpcs(region):
    ec2 = boto3.client('ec2', region_name=region)
    vpcs = []
    paginator = ec2.get_paginator('describe_vpcs')
    for page in paginator.paginate():
        vpcs.extend(page['Vpcs'])
    return vpcs

def collect_all_aws_resources():
    """Resource Groups Tagging API를 사용하여 모든 AWS 리소스를 수집합니다."""
    try:
        logger.info("AWS Resource Groups Tagging API 클라이언트 초기화")
        resourcegroupstaggingapi = boto3.client('resourcegroupstaggingapi')
        
        # 모든 리소스 가져오기 (페이지네이션 적용)
        all_resources = []
        paginator = resourcegroupstaggingapi.get_paginator('get_resources')
        
        # 빈 태그도 포함하도록 설정
        for page in paginator.paginate(ResourcesPerPage=100, ResourceTypeFilters=[]):
            all_resources.extend(page['ResourceTagMappingList'])
        
        logger.info(f"총 {len(all_resources)}개의 AWS 리소스 수집 완료")
        
        # 서비스별 리소스 분석
        service_counts = defaultdict(int)
        for resource in all_resources:
            arn = resource['ResourceARN']
            arn_parts = arn.split(':')
            service = arn_parts[2] if len(arn_parts) > 2 else "unknown"
            service_counts[service] += 1
        
        # 서비스별 리소스 수 로깅
        for service, count in service_counts.items():
            logger.info(f"서비스 {service}: {count}개 리소스")
        
        # 리소스 처리를 통해 상세 정보 추출
        processed_resources = []
        
        # 서비스별 처리를 위한 리소스 그룹화
        resources_by_service = defaultdict(list)
        for resource in all_resources:
            arn = resource['ResourceARN']
            arn_parts = arn.split(':')
            service = arn_parts[2] if len(arn_parts) > 2 else "unknown"
            region = arn_parts[3] if len(arn_parts) > 3 else ""
            resources_by_service[(service, region)].append(resource)
        
        # 서비스별로 처리
        for (service, region), resources in resources_by_service.items():
            logger.info(f"서비스 {service} (리전 {region})의 {len(resources)}개 리소스 처리 시작")
            for resource in resources:
                try:
                    # ARN에서 서비스 및 리소스 타입, ID 추출
                    arn = resource['ResourceARN']
                    arn_parts = arn.split(':')
                    
                    service = arn_parts[2] if len(arn_parts) > 2 else ""
                    region = arn_parts[3] if len(arn_parts) > 3 else ""
                    account = arn_parts[4] if len(arn_parts) > 4 else ""
                    
                    # ARN 형식에 따라 리소스 타입과 ID 추출 방식이 다름
                    resource_type = ""
                    resource_id = ""
                    
                    if len(arn_parts) > 5:
                        resource_path = arn_parts[5]
                        # 서비스별 ARN 형식 처리
                        if service == "ec2":
                            if "/" in resource_path:
                                resource_parts = resource_path.split('/')
                                resource_type = resource_parts[0] if resource_parts else ""
                                resource_id = resource_parts[-1] if len(resource_parts) > 1 else ""
                            else:
                                resource_type = resource_path
                        elif service == "s3":
                            resource_type = "bucket"
                            resource_id = resource_path
                        elif service == "rds":
                            if ":" in resource_path:
                                resource_parts = resource_path.split(':')
                                resource_type = resource_parts[0] if resource_parts else ""
                                resource_id = resource_parts[-1] if len(resource_parts) > 1 else ""
                            else:
                                resource_type = "instance"
                                resource_id = resource_path
                        elif service == "dynamodb":
                            if "/" in resource_path:
                                resource_parts = resource_path.split('/')
                                resource_type = resource_parts[0] if resource_parts else "table"
                                resource_id = resource_parts[-1] if len(resource_parts) > 1 else ""
                            else:
                                resource_type = "table"
                                resource_id = resource_path
                        elif service == "lambda":
                            if ":" in resource_path:
                                resource_parts = resource_path.split(':')
                                resource_type = "function"
                                resource_id = resource_parts[-1] if len(resource_parts) > 1 else ""
                            else:
                                resource_type = "function"
                                resource_id = resource_path
                        else:
                            # 기본 처리: 슬래시로 분리된 경우
                            if "/" in resource_path:
                                resource_parts = resource_path.split('/')
                                resource_type = resource_parts[0] if resource_parts else ""
                                resource_id = resource_parts[-1] if len(resource_parts) > 1 else ""
                            # 콜론으로 분리된 경우
                            elif ":" in resource_path:
                                resource_parts = resource_path.split(':')
                                resource_type = resource_parts[0] if resource_parts else ""
                                resource_id = resource_parts[-1] if len(resource_parts) > 1 else ""
                            # 분리자가 없는 경우
                            else:
                                resource_id = resource_path
                    
                    # 태그 가져오기
                    tags = {tag['Key']: tag['Value'] for tag in resource.get('Tags', [])}
                    
                    # 상세 정보 가져오기
                    detailed_info = get_resource_details(arn, service, resource_type, resource_id, region)
                    
                    # 리소스 정보 구성
                    resource_info = {
                        'arn': arn,
                        'service': service,
                        'resource_type': resource_type,
                        'resource_id': resource_id,
                        'region': region,
                        'account_id': account,
                        'tags': tags,
                        'details': detailed_info
                    }
                    
                    processed_resources.append(resource_info)
                    logger.debug(f"리소스 처리 완료: {arn}")
                    
                except Exception as e:
                    logger.error(f"리소스 {resource.get('ResourceARN', '알 수 없음')} 처리 중 오류: {e}")
                    continue
            
            logger.info(f"서비스 {service} (리전 {region})의 리소스 처리 완료")
        
        # region 목록 추출 (이미 수집된 리소스에서 region 추출)
        regions = set()
        for resource in processed_resources:
            if resource['service'] == 'ec2' and resource['resource_type'] == 'vpc':
                regions.add(resource['region'])

        # region별로 describe_vpcs()로 모든 VPC 수집
        for region in regions:
            ec2 = boto3.client('ec2', region_name=region)
            vpcs = ec2.describe_vpcs()['Vpcs']
            for vpc in vpcs:
                vpc_id = vpc['VpcId']
                # 이미 수집된 VPC인지 확인
                if not any(r['resource_id'] == vpc_id and r['region'] == region for r in processed_resources):
                    # 태그 변환
                    tags = {tag['Key']: tag['Value'] for tag in vpc.get('Tags', [])}
                    resource_info = {
                        'arn': f"arn:aws:ec2:{region}:{vpc.get('OwnerId', '')}:vpc/{vpc_id}",
                        'service': 'ec2',
                        'resource_type': 'vpc',
                        'resource_id': vpc_id,
                        'region': region,
                        'account_id': vpc.get('OwnerId', ''),
                        'tags': tags,
                        'details': vpc
                    }
                    processed_resources.append(resource_info)

        logger.info(f"총 {len(processed_resources)}개 AWS 리소스 처리 완료")
        return processed_resources
        
    except Exception as e:
        logger.error(f"AWS 리소스 수집 중 오류 발생: {e}")
        return []

def get_resource_details(arn, service, resource_type, resource_id, region):
    """리소스 유형에 따라 특정 AWS 리소스의 상세 정보를 가져옵니다."""
    try:
        # 서비스별 클라이언트 초기화 (리전 지정)
        boto3_service = SERVICE_CLIENT_MAP.get(service, service)
        client = boto3.client(boto3_service, region_name=region) if region else boto3.client(boto3_service)
        
        # 서비스 및 리소스 유형에 따라 상세 정보 가져오기
        if service == 'ec2':
            if resource_type == 'vpc':
                try:
                    response = client.describe_vpcs(VpcIds=[resource_id])
                    return response['Vpcs'][0] if response.get('Vpcs') else {}
                except Exception as e:
                    logger.error(f"VPC {resource_id} 정보 가져오기 실패: {e}")
                    return {}
            elif resource_type == 'subnet':
                try:
                    response = client.describe_subnets(SubnetIds=[resource_id])
                    return response['Subnets'][0] if response.get('Subnets') else {}
                except Exception as e:
                    logger.error(f"서브넷 {resource_id} 정보 가져오기 실패: {e}")
                    return {}
            elif resource_type == 'security-group':
                try:
                    response = client.describe_security_groups(GroupIds=[resource_id])
                    return response['SecurityGroups'][0] if response.get('SecurityGroups') else {}
                except Exception as e:
                    logger.error(f"보안 그룹 {resource_id} 정보 가져오기 실패: {e}")
                    return {}
            elif resource_type == 'route-table':
                try:
                    response = client.describe_route_tables(RouteTableIds=[resource_id])
                    return response['RouteTables'][0] if response.get('RouteTables') else {}
                except Exception as e:
                    logger.error(f"라우팅 테이블 {resource_id} 정보 가져오기 실패: {e}")
                    return {}
            elif resource_type == 'instance':
                try:
                    response = client.describe_instances(InstanceIds=[resource_id])
                    reservations = response.get('Reservations', [])
                    if reservations and reservations[0].get('Instances'):
                        return reservations[0]['Instances'][0]
                    return {}
                except Exception as e:
                    logger.error(f"인스턴스 {resource_id} 정보 가져오기 실패: {e}")
                    return {}
            elif resource_type == 'network-interface':
                try:
                    response = client.describe_network_interfaces(NetworkInterfaceIds=[resource_id])
                    return response['NetworkInterfaces'][0] if response.get('NetworkInterfaces') else {}
                except Exception as e:
                    logger.error(f"네트워크 인터페이스 {resource_id} 정보 가져오기 실패: {e}")
                    return {}
            elif resource_type == 'volume':
                try:
                    response = client.describe_volumes(VolumeIds=[resource_id])
                    return response['Volumes'][0] if response.get('Volumes') else {}
                except Exception as e:
                    logger.error(f"볼륨 {resource_id} 정보 가져오기 실패: {e}")
                    return {}
        
        elif service == 's3':
            # S3 버킷 처리
            try:
                # 버킷 위치 정보
                location_response = client.get_bucket_location(Bucket=resource_id)
                location = location_response.get('LocationConstraint')
                
                # 버킷 태그 가져오기
                tags = {}
                try:
                    tags_response = client.get_bucket_tagging(Bucket=resource_id)
                    tags = {tag['Key']: tag['Value'] for tag in tags_response.get('TagSet', [])}
                except Exception:
                    # 태그가 없는 경우 무시
                    pass
                
                # 버킷 정책 확인
                policy = None
                try:
                    policy_response = client.get_bucket_policy(Bucket=resource_id)
                    policy = policy_response.get('Policy')
                except Exception:
                    # 정책이 없는 경우 무시
                    pass
                
                return {
                    'Location': location,
                    'Tags': tags,
                    'Policy': policy
                }
            except Exception as e:
                logger.error(f"S3 버킷 {resource_id} 정보 가져오기 실패: {e}")
                return {}
        
        elif service == 'rds':
            # RDS 인스턴스 처리
            try:
                response = client.describe_db_instances(DBInstanceIdentifier=resource_id)
                return response['DBInstances'][0] if response.get('DBInstances') else {}
            except Exception as e:
                logger.error(f"RDS 인스턴스 {resource_id} 정보 가져오기 실패: {e}")
                return {}
        
        elif service == 'lambda':
            # Lambda 함수 처리
            try:
                response = client.get_function(FunctionName=resource_id)
                return response.get('Configuration', {})
            except Exception as e:
                logger.error(f"Lambda 함수 {resource_id} 정보 가져오기 실패: {e}")
                return {}
        
        elif service == 'dynamodb':
            # DynamoDB 테이블 처리
            try:
                response = client.describe_table(TableName=resource_id)
                return response.get('Table', {})
            except Exception as e:
                logger.error(f"DynamoDB 테이블 {resource_id} 정보 가져오기 실패: {e}")
                return {}
        
        # 추가 서비스 처리는 필요에 따라 확장
        
        return {}
    except Exception as e:
        logger.error(f"리소스 {arn} 상세 정보 가져오기 중 오류: {e}")
        return {}

def organize_resources_by_type(resources_data):
    """리소스를 서비스 및 타입별로 그룹화하고, 연관 관계를 추적합니다."""
    # 기본 그룹화
    organized = {
        "by_service": defaultdict(list),
        "by_type": defaultdict(list),
        "by_region": defaultdict(list),
        # 연관 관계 추적을 위한 추가 그룹화
        "by_vpc": defaultdict(list),      # VPC ID 기준 그룹화
        "by_subnet": defaultdict(list),   # 서브넷 ID 기준 그룹화 
        "by_sg": defaultdict(list)        # 보안 그룹 ID 기준 그룹화
    }
    
    # 리소스 매핑 (ID -> ARN)
    id_to_arn = {}
    
    for resource in resources_data:
        service = resource.get('service', 'unknown')
        resource_type = resource.get('resource_type', 'unknown')
        region = resource.get('region', 'unknown')
        resource_id = resource.get('resource_id', '')
        arn = resource.get('arn', '')
        
        # ID -> ARN 매핑 추가
        if resource_id:
            id_key = f"{service}:{region}:{resource_type}:{resource_id}"
            id_to_arn[id_key] = arn
        
        # 서비스별 그룹화
        organized["by_service"][service].append(resource)
        
        # 리소스 타입별 그룹화
        type_key = f"{service}:{resource_type}"
        organized["by_type"][type_key].append(resource)
        
        # 리전별 그룹화
        organized["by_region"][region].append(resource)
        
        # 연관 관계 그룹화
        details = resource.get('details', {})
        
        # EC2 인스턴스 관계
        if service == 'ec2' and resource_type == 'instance':
            # VPC 관계
            vpc_id = details.get('VpcId')
            if vpc_id:
                organized["by_vpc"][vpc_id].append(resource)
            
            # 서브넷 관계
            subnet_id = details.get('SubnetId')
            if subnet_id:
                organized["by_subnet"][subnet_id].append(resource)
            
            # 보안 그룹 관계
            if 'SecurityGroups' in details:
                for sg in details['SecurityGroups']:
                    sg_id = sg.get('GroupId')
                    if sg_id:
                        organized["by_sg"][sg_id].append(resource)
        
        # 서브넷 관계
        elif service == 'ec2' and resource_type == 'subnet':
            vpc_id = details.get('VpcId')
            if vpc_id:
                organized["by_vpc"][vpc_id].append(resource)
        
        # 보안 그룹 관계
        elif service == 'ec2' and resource_type == 'security-group':
            vpc_id = details.get('VpcId')
            if vpc_id:
                organized["by_vpc"][vpc_id].append(resource)
        
        # RDS 인스턴스 관계
        elif service == 'rds' and resource_type == 'instance':
            # VPC 보안 그룹 관계
            if 'VpcSecurityGroups' in details:
                for sg in details['VpcSecurityGroups']:
                    sg_id = sg.get('VpcSecurityGroupId')
                    if sg_id:
                        organized["by_sg"][sg_id].append(resource)
            
            # 서브넷 그룹 관계
            if 'DBSubnetGroup' in details and 'Subnets' in details['DBSubnetGroup']:
                for subnet in details['DBSubnetGroup']['Subnets']:
                    subnet_id = subnet.get('SubnetIdentifier')
                    if subnet_id:
                        organized["by_subnet"][subnet_id].append(resource)
                
                # VPC 관계
                vpc_id = details['DBSubnetGroup'].get('VpcId')
                if vpc_id:
                    organized["by_vpc"][vpc_id].append(resource)
    
    # 그룹화된 정보 요약 생성
    summary = {
        "total_resources": len(resources_data),
        "services": {},
        "resource_types": {},
        "regions": {},
        # 관계 요약 추가
        "vpc_resources": {},
        "subnet_resources": {},
        "sg_resources": {}
    }
    
    # 서비스별 요약
    for service, resources in organized["by_service"].items():
        summary["services"][service] = len(resources)
    
    # 리소스 타입별 요약
    for type_key, resources in organized["by_type"].items():
        summary["resource_types"][type_key] = len(resources)
    
    # 리전별 요약
    for region, resources in organized["by_region"].items():
        summary["regions"][region] = len(resources)
    
    # 연관 관계 요약
    for vpc_id, resources in organized["by_vpc"].items():
        summary["vpc_resources"][vpc_id] = len(resources)
    
    for subnet_id, resources in organized["by_subnet"].items():
        summary["subnet_resources"][subnet_id] = len(resources)
    
    for sg_id, resources in organized["by_sg"].items():
        summary["sg_resources"][sg_id] = len(resources)
    
    logger.info(f"리소스 그룹화 완료: {len(summary['services'])}개 서비스, {len(summary['resource_types'])}개 리소스 타입")
    logger.info(f"관계 그룹화 완료: {len(summary['vpc_resources'])}개 VPC, {len(summary['subnet_resources'])}개 서브넷, {len(summary['sg_resources'])}개 보안 그룹")
    
    return organized, summary

def create_llama_documents(resources_data, organized_resources):
    """수집된 AWS 리소스 데이터를 LlamaIndex Document 객체로 변환합니다."""
    documents = []
    logger.info("Document 객체 변환 시작")
    
    # 1. 전체 요약 문서 생성
    summary_doc = Document(
        text=json.dumps({
            "document_type": "summary",
            "total_resources": len(resources_data),
            "services": {k: len(v) for k, v in organized_resources["by_service"].items()},
            "resource_types": {k: len(v) for k, v in organized_resources["by_type"].items()},
            "regions": {k: len(v) for k, v in organized_resources["by_region"].items()},
            # 관계 요약 추가
            "vpc_resources": {k: len(v) for k, v in organized_resources.get("by_vpc", {}).items()},
            "subnet_resources": {k: len(v) for k, v in organized_resources.get("by_subnet", {}).items()},
            "sg_resources": {k: len(v) for k, v in organized_resources.get("by_sg", {}).items()}
        }, indent=2, ensure_ascii=False, default=str),
        metadata={
            "document_type": "summary",
            "title": "AWS 리소스 요약"
        }
    )
    documents.append(summary_doc)
    logger.info("전체 요약 문서 생성 완료")
    
    # 2. 서비스별 요약 문서 생성
    for service, resources in organized_resources["by_service"].items():
        service_summary = {
            "document_type": "service_summary",
            "service": service,
            "resource_count": len(resources),
            "resource_types": {},
            "regions": {}
        }
        
        # 서비스 내 리소스 타입별 카운트
        resource_types = defaultdict(int)
        regions = defaultdict(int)
        
        for resource in resources:
            resource_type = resource.get('resource_type', 'unknown')
            region = resource.get('region', 'unknown')
            resource_types[resource_type] += 1
            regions[region] += 1
        
        service_summary["resource_types"] = dict(resource_types)
        service_summary["regions"] = dict(regions)
        
        service_doc = Document(
            text=json.dumps(service_summary, indent=2, ensure_ascii=False, default=str),
            metadata={
                "document_type": "service_summary",
                "service": service,
                "title": f"{service} 서비스 요약"
            }
        )
        documents.append(service_doc)
    
    logger.info("서비스별 요약 문서 생성 완료")
    
    # 3. 개별 리소스 문서 생성
    for data in resources_data:
        try:
            # 리소스 유형에 기반한 메타데이터 생성
            metadata = {
                "document_type": "resource",
                "arn": data.get("arn", "unknown"),
                "resource_id": data.get("resource_id", "unknown"),
                "resource_type": data.get("resource_type", "unknown"),
                "service": data.get("service", "unknown"),
                "region": data.get("region", "unknown")
            }
            
            # 서비스별 특별 처리
            service = data.get("service", "unknown")
            resource_type = data.get("resource_type", "unknown")
            resource_id = data.get("resource_id", "unknown")
            details = data.get("details", {})
            
            # 연관 관계 정보 추가
            # 1. EC2 인스턴스 관계
            if service == 'ec2' and resource_type == 'instance':
                # VPC 관계
                if details.get('VpcId'):
                    metadata["vpc_id"] = details['VpcId']
                
                # 서브넷 관계
                if details.get('SubnetId'):
                    metadata["subnet_id"] = details['SubnetId']
                
                # 보안 그룹 관계
                if 'SecurityGroups' in details:
                    sg_ids = [sg.get('GroupId') for sg in details['SecurityGroups'] if sg.get('GroupId')]
                    if sg_ids:
                        metadata["security_group_ids"] = sg_ids
            
            # 2. 서브넷 관계
            elif service == 'ec2' and resource_type == 'subnet':
                if details.get('VpcId'):
                    metadata["vpc_id"] = details['VpcId']
            
            # 3. 보안 그룹 관계
            elif service == 'ec2' and resource_type == 'security-group':
                if details.get('VpcId'):
                    metadata["vpc_id"] = details['VpcId']
            
            # 4. RDS 인스턴스 관계
            elif service == 'rds' and resource_type == 'instance':
                # VPC 관계
                if 'DBSubnetGroup' in details and details['DBSubnetGroup'].get('VpcId'):
                    metadata["vpc_id"] = details['DBSubnetGroup']['VpcId']
                
                # 서브넷 관계
                if 'DBSubnetGroup' in details and 'Subnets' in details['DBSubnetGroup']:
                    subnet_ids = [subnet.get('SubnetIdentifier') for subnet in details['DBSubnetGroup']['Subnets'] 
                                if subnet.get('SubnetIdentifier')]
                    if subnet_ids:
                        metadata["subnet_ids"] = subnet_ids
                
                # 보안 그룹 관계
                if 'VpcSecurityGroups' in details:
                    sg_ids = [sg.get('VpcSecurityGroupId') for sg in details['VpcSecurityGroups'] 
                             if sg.get('VpcSecurityGroupId')]
                    if sg_ids:
                        metadata["security_group_ids"] = sg_ids
            
            # 관계 정보를 기반으로 연결된 리소스 검색
            relationship_info = {}
            
            # VPC 기반 관계
            if "vpc_id" in metadata:
                vpc_id = metadata["vpc_id"]
                # 같은 VPC에 있는 다른 리소스 수 계산
                vpc_resources = organized_resources["by_vpc"].get(vpc_id, [])
                relationship_info["same_vpc_resources_count"] = len(vpc_resources)
                
                # VPC 내 리소스 유형 분포
                vpc_resource_types = defaultdict(int)
                for res in vpc_resources:
                    res_service = res.get('service', 'unknown')
                    res_type = res.get('resource_type', 'unknown')
                    vpc_resource_types[f"{res_service}:{res_type}"] += 1
                
                if vpc_resource_types:
                    relationship_info["vpc_resource_types"] = dict(vpc_resource_types)
            
            # 서브넷 기반 관계
            if "subnet_id" in metadata:
                subnet_id = metadata["subnet_id"]
                # 같은 서브넷에 있는 다른 리소스 수 계산
                subnet_resources = organized_resources["by_subnet"].get(subnet_id, [])
                relationship_info["same_subnet_resources_count"] = len(subnet_resources)
            
            # 보안 그룹 기반 관계
            if "security_group_ids" in metadata:
                sg_resources_count = 0
                sg_resource_types = defaultdict(int)
                
                for sg_id in metadata["security_group_ids"]:
                    # 같은 보안 그룹을 사용하는 다른 리소스 계산
                    sg_resources = organized_resources["by_sg"].get(sg_id, [])
                    sg_resources_count += len(sg_resources)
                    
                    # 보안 그룹 사용 리소스 유형 분포
                    for res in sg_resources:
                        res_service = res.get('service', 'unknown')
                        res_type = res.get('resource_type', 'unknown')
                        sg_resource_types[f"{res_service}:{res_type}"] += 1
                
                if sg_resources_count > 0:
                    relationship_info["security_group_resources_count"] = sg_resources_count
                
                if sg_resource_types:
                    relationship_info["security_group_resource_types"] = dict(sg_resource_types)
            
            # 관계 정보가 있으면 메타데이터에 추가
            if relationship_info:
                metadata["relationships"] = relationship_info
            
            # VPC 특별 처리 - Name 태그가 없어도 CIDR 정보로 식별 가능하게
            if service == "ec2" and resource_type == "vpc":
                # CIDR 정보 추출
                details = data.get("details", {})
                cidr = details.get("CidrBlock", "")
                if cidr:
                    metadata["cidr"] = cidr
                    # Name 태그가 없으면 CIDR로 대체
                    if "tags" in data and isinstance(data["tags"], dict) and "Name" in data["tags"]:
                        metadata["name"] = data["tags"]["Name"]
                    else:
                        metadata["name"] = f"VPC-{cidr}"
            # 일반적인 처리
            elif "tags" in data and isinstance(data["tags"], dict) and "Name" in data["tags"]:
                metadata["name"] = data["tags"]["Name"]
            
            # 원래 데이터에 관계 정보 추가
            enriched_data = data.copy()
            if relationship_info:
                enriched_data["relationship_info"] = relationship_info
            
            # Document 객체로 변환
            doc = Document(
                text=json.dumps(enriched_data, indent=2, ensure_ascii=False, default=str),
                metadata=metadata
            )
            documents.append(doc)
            logger.debug(f"{metadata['service']}:{metadata['resource_type']}:{metadata.get('name', metadata['resource_id'])} Document 생성 완료")
        except Exception as e:
            logger.error(f"문서 생성 중 오류 발생: {e}")
    
    logger.info(f"총 {len(documents)}개의 Document 객체 생성 완료")
    
    # 4. VPC 관계 요약 문서 생성 (추가)
    for vpc_id, resources in organized_resources.get("by_vpc", {}).items():
        if not resources:
            continue
            
        vpc_info = {
            "document_type": "vpc_summary",
            "vpc_id": vpc_id,
            "resource_count": len(resources),
            "resource_types": {},
            "regions": {}
        }
        
        # VPC 내 리소스 타입별 카운트
        resource_types = defaultdict(int)
        regions = defaultdict(int)
        
        for resource in resources:
            service = resource.get('service', 'unknown')
            resource_type = resource.get('resource_type', 'unknown')
            region = resource.get('region', 'unknown')
            
            type_key = f"{service}:{resource_type}"
            resource_types[type_key] += 1
            regions[region] += 1
        
        vpc_info["resource_types"] = dict(resource_types)
        vpc_info["regions"] = dict(regions)
        
        # VPC 자체 정보 찾기
        vpc_resource = None
        for resource in resources:
            if resource.get('service') == 'ec2' and resource.get('resource_type') == 'vpc' and resource.get('resource_id') == vpc_id:
                vpc_resource = resource
                break
        
        if vpc_resource:
            vpc_details = vpc_resource.get('details', {})
            vpc_info["cidr_block"] = vpc_details.get('CidrBlock', '')
            vpc_info["is_default"] = vpc_details.get('IsDefault', False)
            
            # 태그 정보
            if 'tags' in vpc_resource and isinstance(vpc_resource['tags'], dict):
                vpc_info["tags"] = vpc_resource['tags']
                if 'Name' in vpc_resource['tags']:
                    vpc_info["name"] = vpc_resource['tags']['Name']
        
        vpc_doc = Document(
            text=json.dumps(vpc_info, indent=2, ensure_ascii=False, default=str),
            metadata={
                "document_type": "vpc_summary",
                "vpc_id": vpc_id,
                "title": f"VPC {vpc_id} 리소스 요약"
            }
        )
        documents.append(vpc_doc)
    
    return documents

def build_index(documents, persist_dir=None):
    """LlamaIndex 인덱스를 생성하고 저장합니다."""
    if persist_dir is None:
        persist_dir = INDEX_DIR

    try:
        logger.info("인덱스 구축 시작")
        # 저장 디렉토리가 없으면 생성
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        # 임베딩 모델 설정
        setup_embed_model()
        # 인덱스 생성
        index = VectorStoreIndex.from_documents(documents)
        logger.info("인덱스 생성 완료")
        # 인덱스 저장
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"인덱스가 {persist_dir}에 저장되었습니다.")
        return index
    except Exception as e:
        logger.error(f"인덱스 생성 중 오류 발생: {e}")
        return None

def main():
    # 1. AWS 리소스 데이터 수집
    resources_data = collect_all_aws_resources()
    if not resources_data:
        logger.error("AWS 리소스 데이터 수집 실패. 종료합니다.")
        return 1

    # 2. 리소스 데이터 구성 및 요약
    organized_resources, resources_summary = organize_resources_by_type(resources_data)
    logger.info(f"리소스 요약: 총 {resources_summary['total_resources']}개 리소스, "
                f"{len(resources_summary['services'])}개 서비스, "
                f"{len(resources_summary['resource_types'])}개 리소스 타입")

    # 3. LlamaIndex Document 객체로 변환
    documents = create_llama_documents(resources_data, organized_resources)
    if not documents:
        logger.error("LlamaIndex Document 생성 실패. 종료합니다.")
        return 1

    # 4. 문서 저장
    data_path = DATA_DIR / "aws_resources.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump([doc.to_json() for doc in documents], f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"AWS 리소스 데이터가 {data_path}에 저장되었습니다.")

    # 5. 인덱스 구축
    index = build_index(documents)
    if not index:
        logger.error("인덱스 구축 실패. 종료합니다.")
        return 1

    # 6. 요약 정보 저장
    summary_path = DATA_DIR / "resources_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(resources_summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"리소스 요약 정보가 {summary_path}에 저장되었습니다.")

    logger.info(f"AWS 리소스 정보 인덱스 구축 완료 ({INDEX_DIR})")
    print(f"""
작업이 완료되었습니다!
- 인덱스 폴더: {INDEX_DIR}
- 데이터 폴더: {DATA_DIR}
- 로그 폴더: {LOGS_DIR}
""")

if __name__ == "__main__":
    sys.exit(main())