import matplotlib.pyplot as plt
import re

# 읽을 파일 경로
test_name = "opt-1.3b_infer"
file_path = "output/" + test_name + "_results.txt"
pdf_output_path = "output/" + test_name + "_results.pdf"

# 데이터 저장 리스트
x_labels = []  # X축 레이블 (percent pair)
y_values = []  # Y축 값 (Average latency)

try:
    # 파일 읽어오기
    with open(file_path, "r") as file:
        lines = file.readlines()

        for line in lines:
            # "Average Total Latency"가 포함된 줄에서 데이터를 추출
            if "Average Total Latency" in line:
                # X축 레이블 설정 (percent pair)
                match_pair = re.search(r"--percent (\d{1,3} \d{1,3})", line)
                if match_pair:
                    x_labels.append(match_pair.group(1))

                # Y축 값 설정 (마지막 숫자: Average latency)
                match_latency = re.search(r"(\d+\.\d+|\.\d+)\s*s$", line)
                if match_latency:
                    y_values.append(float(match_latency.group(1)))

    # Data validation
    if not x_labels or not y_values:
        print("No valid data found in the file. Please check the file content.")
        exit()

    # 라인 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, y_values, marker='o', linestyle='-', color='b', label='Average Latency')

    # 그래프 제목 및 레이블
    plt.title("Average Total Latency for " + test_name, fontsize=14)
    plt.xlabel("Percent Pair", fontsize=12)
    plt.ylabel("Average Latency (s)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 그래프를 PDF 파일로 저장
    plt.savefig(pdf_output_path, format="pdf")
    print(f"Graph has been saved to PDF successfully at: {pdf_output_path}")

    # 그래프 출력
    plt.show()

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
