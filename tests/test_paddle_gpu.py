import paddle
from paddleocr import PaddleOCR
import time
import multiprocessing

def test_cpu_performance():
    print("PaddlePaddle version:", paddle.__version__)
    print("Available CPU cores:", multiprocessing.cpu_count())
    print("MKL enabled:", paddle.device.get_device())
    
    # Initialize PaddleOCR with CPU optimization
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        cpu_threads=6,
        enable_mkldnn=False
    )
    
    # Test processing speed
    start_time = time.time()
    result = ocr.ocr("data/invoice.jpeg")
    end_time = time.time()
    
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print("Device used:", paddle.device.get_device())
    
    if result and result[0]:
        print(f"\nDetected {len(result[0])} text regions")

if __name__ == "__main__":
    test_cpu_performance()
