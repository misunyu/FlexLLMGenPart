import torch
import time


# 설정: 모델 및 입력 크기
# batch_size = 32
# seq_len = 128
# hidden_dim = 768
# device = torch.device("cuda")

# 예제 모델 (Transformer 레이어)
class SimpleTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

def cpu_gpu_transfer_time(data):
    # 데이터 크기 설정 (예: 1GB)
    data_size = 1 * 1024 * 1024 * 1024  # 1GB
    dtype = torch.float32  # 데이터 타입 (float32)
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda")

    # CPU에서 데이터 생성
    data_cpu = torch.randn(data_size // torch.tensor([], dtype=dtype).element_size(), device=device_cpu)

    # 전송 시간 측정
    torch.cuda.synchronize()  # GPU 초기화 및 동기화
    start_time = time.time()

    # CPU → GPU 데이터 전송
    data_gpu = data_cpu.to(device_gpu)

    torch.cuda.synchronize()  # GPU 작업 완료 대기
    end_time = time.time()

    # 전송 시간 계산
    transfer_time = end_time - start_time
    transfer_speed = data_size / transfer_time / (1024 ** 3)  # GB/s

    print(f"Data transfer time: {transfer_time:.6f} seconds")
    print(f"Data transfer speed: {transfer_speed:.2f} GB/s")



def measure_unit_compute_time(model, input_shape, device="cuda"):
    """
    Measure the computation time per unit (element) for the given model.

    Args:
        model (torch.nn.Module): The model or layer to measure.
        input_shape (tuple): Shape of the input tensor (e.g., (batch_size, seq_len, hidden_dim)).
        device (str): Device for computation ("cuda" or "cpu").

    Returns:
        dict: Dictionary containing total time, unit time, and input size.
    """

    # 모델을 지정된 디바이스로 이동
    model.to(device)
    model.eval()  # 평가 모드

    # 입력 데이터 생성
    input_data = torch.randn(*input_shape, device=device)

    # 타이머 시작
    torch.cuda.synchronize()  # GPU 초기화
    start_time = time.time()

    # 모델 실행
    with torch.no_grad():  # 그래디언트 계산 제외
        _ = model(input_data)

    # 타이머 종료
    torch.cuda.synchronize()
    end_time = time.time()

    # 실행 시간 및 단위 크기당 시간 계산
    total_time = end_time - start_time  # 총 실행 시간
    input_size = torch.tensor(input_shape).prod().item()  # 입력 데이터 크기 (요소 수)
    unit_time = (total_time / input_size) * 1_000_000  # 단위 요소당 시간 (µs)

    # 결과 출력
    print(f"Total computation time: {total_time:.6f} seconds")
    print(f"Input size: {input_size} elements")
    print(f"Time per unit: {unit_time:.3f} µs/element")

    # 결과 반환
    return {
        "total_time": total_time,
        "unit_time_microseconds": unit_time,
        "input_size": input_size,
    }


def measure_pure_gpu_sync_time(func, *args, **kwargs):
    """
    Measure the pure GPU synchronization waiting time, excluding the function execution time.

    Args:
        func (callable): The function to execute GPU operations.
        *args: Arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        dict: Dictionary containing execution time, sync time, and pure wait time.
    """

    # 1. GPU 작업 수행 시간 측정 (동기화 제외)
    start_time = time.time()
    func(*args, **kwargs)  # GPU 작업 수행
    exec_end_time = time.time()

    # 2. GPU 동기화 시간 측정
    torch.cuda.synchronize()
    sync_end_time = time.time()

    # 3. 시간 계산
    execution_time = (exec_end_time - start_time)*1_000  # GPU 작업 수행 시간
    total_sync_time = (sync_end_time - start_time)*1_000  # GPU 작업 + 동기화 시간
    pure_wait_time = total_sync_time - execution_time  # 순수 대기 시간

    # 결과 출력
    print(f"Execution time (without sync): {execution_time:.6f} msec")
    print(f"Total sync time: {total_sync_time:.6f} msec")
    print(f"Pure GPU wait time: {pure_wait_time:.6f} msec")

    return {
        "execution_time": execution_time,
        "total_sync_time": total_sync_time,
        "pure_wait_time": pure_wait_time,
    }

def measure_stream_switching_overhead(num_switches=1000):
    """
    Measure the overhead caused by switching between multiple GPU streams.

    Args:
        num_switches (int): Number of stream switches to measure.

    Returns:
        float: Total overhead time in seconds.
    """

    # 두 개의 스트림 생성
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # GPU 동기화
    torch.cuda.synchronize()
    start_time = time.time()

    # 스트림 전환 반복
    for _ in range(num_switches):
        with torch.cuda.stream(stream1):
            pass  # 스트림 1 활성화
        with torch.cuda.stream(stream2):
            pass  # 스트림 2 활성화

    # GPU 동기화
    torch.cuda.synchronize()
    end_time = time.time()

    overhead_time = (end_time - start_time)*1_000
    print(f"Stream switching overhead for {num_switches} switches: {overhead_time:.6f} msec")
    return overhead_time


# 예제 GPU 작업
def example_gpu_operation(x):
    x_gpu = x.to("cuda")  # 데이터 전송
    x_gpu = x_gpu * 2  # 계산
    return x_gpu


if __name__ == "__main__":
    cpu_gpu_transfer_time(data=None)
    print("-------------------")

    # 예제: Transformer 레이어
    layer = SimpleTransformerLayer(hidden_dim=768)
    input_shape = (32, 128, 768)  # (batch_size, seq_len, hidden_dim)

    # 측정 함수 호출
    results = measure_unit_compute_time(layer, input_shape, device="cuda")

    print("-------------------")

    # 입력 데이터 생성
    x_cpu = torch.randn(1000000)

    # 순수 GPU 동기화 대기 시간 측정
    result = measure_pure_gpu_sync_time(example_gpu_operation, x_cpu)
    print("-------------------")

    overhead_time = measure_stream_switching_overhead(num_switches=10000)