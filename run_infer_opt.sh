#!/bin/bash

# 설정할 모델 리스트
MODELS=(
#  "opt-1.3b"
#  "opt-2.7b"
#  "opt-6.7b"
  "opt-13b"
)

# 설정할 숫자 쌍 리스트
PERCENT_PAIRS=(
#  "100 0"
#  "90 10"
#  "80 20"
  "70 30"
  "60 40"
  "50 50"
  "40 60"
  "30 70"
  "20 80"
  "10 90"
  "0 100"
)

# 명령어 기본 템플릿
BASE_CMD="python3 -m flexllmgen.infer_opt"

# 반복 횟수 (N번 실행)
ITERATIONS=10  # N번 반복 실행

# 결과를 저장할 폴더 정의
OUTPUT_DIR="output"  # 결과 파일을 저장할 폴더
mkdir -p "$OUTPUT_DIR"  # output 폴더가 없으면 생성

# 각 모델에 대해 테스트 반복
for MODEL in "${MODELS[@]}"; do
    echo -e "\nTesting with model: $MODEL"

    # 현재 모델 이름으로 결과 파일 설정
    MODEL_OUTPUT_FILE="$OUTPUT_DIR/${MODEL}_infer_results.txt"

    # 모델별 결과 파일 초기화
    echo "Latency Test Results for Model: $MODEL (rounded to 2 decimal places):" > "$MODEL_OUTPUT_FILE"

    # 각 설정된 percent 쌍에 대해 테스트
    for PAIR in "${PERCENT_PAIRS[@]}"; do
        echo -e "\n  Testing with percent pair: $PAIR"

        # 실행할 명령어 조합
        CMD="$BASE_CMD --model facebook/$MODEL --percent $PAIR 100 0 100 0"
        
        # 합계 및 횟수 초기화
        total_latency_sum=0
        count=0

        # N번 명령어 실행
        for ((i=1; i<=ITERATIONS; i++)); do
            echo "    Run $i of $ITERATIONS with model $MODEL and percent pair $PAIR..."
            
            # 명령어 실행 및 total latency 값 추출
            RESULT=$($CMD | grep "total latency:" | awk -F ':' '{print $2}' | awk '{print $1}')
            
            # 추출된 값이 숫자인지 확인
            if [[ $RESULT =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                total_latency_sum=$(echo "$total_latency_sum + $RESULT" | bc)
                count=$((count + 1))
                echo "      Run $i: Total Latency = $RESULT s"
            else
                echo "      Run $i: Failed to get total latency"
            fi
        done

        # 평균 계산
        if [[ $count -gt 0 ]]; then
            # 소수점 이하 두 자리까지 반올림
            average_latency=$(echo "scale=2; $total_latency_sum / $count" | bc)
            echo "Command: $CMD -> Average Total Latency after $count runs: $average_latency s"
            echo "$CMD -> Average Total Latency after $count runs: $average_latency s" >> "$MODEL_OUTPUT_FILE"
        else
            echo "Command: $CMD -> Failed to collect valid data"
            echo "$CMD -> Failed to collect valid data" >> "$MODEL_OUTPUT_FILE"
        fi
    done

    echo -e "\nResults for model $MODEL saved to: $MODEL_OUTPUT_FILE"
done

echo -e "\nAll results are saved in the output folder: $OUTPUT_DIR"
