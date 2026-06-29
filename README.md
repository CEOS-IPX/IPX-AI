# 동의어 사전 파일 배치 안내

## 파일 위치

```
IPX-AI/
├── app/
│   └── resources/
│       └── synonyms_patent.txt
└── ...
```

## 원본 위치

Spring 레포(`backend-spring/infra/opensearch/synonyms_patent.txt`)의 동일 파일을 그대로 복사합니다.

각 레포에서의 사용 목적
1. Spring 레포: OpenSearch 컨테이너에 마운트되어 검색 시 동의어 필터로 활용 
2. Python 레포: 키워드 확장되어 가상 초록 생성 및 KIPRIS API 파라미터에 활용

## 동기화 규칙

원본은 Spring 레포에 두고, Python 레포에는 복사본을 둡니다. 동의어를 수정할 때는 양쪽 모두 업데이트합니다.

## 파일 형식

```txt
# 주석은 '#'으로 시작
# 한 줄에 하나의 동의어 그룹, 쉼표로 구분
# 빈 줄은 무시됨

# 단어 수준 동의어
배터리, 이차전지, secondary cell, 축전지, rechargeable battery
양극재, cathode material, 정극재

# 문제-기술 매핑
급속충전, 정전류정전압충전, 리튬플레이팅억제
에너지밀도, 고용량양극재, 실리콘음극재
```

