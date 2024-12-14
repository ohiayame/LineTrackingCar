import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard
import cv2
import threading
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# sudo 비밀번호를 변수로 설정
sudo_password = " "

# sudo 비밀번호로 셸 명령어를 실행하는 함수
def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)

# busybox가 설치되어 있는지 확인하고, 없으면 설치
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# devmem 명령어 정의
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]
# 각 devmem 명령어 실행
for command in commands:
    run_command(command)

# GPIO 핀 설정
servo_pin = 33  # 서보 모터 PWM핀
dc_motor_pwm_pin = 32  # DC 모터의 속도를 조절 PWM핀
dc_motor_dir_pin1 = 29  # GPIO 5, DC 모터의 방향을 제어 + (OUT1 + && IN1 + )
dc_motor_dir_pin2 = 31  # GPIO 6, DC 모터의 방향을 제어 - (OUT2 - && IN2 - )

GPIO.setmode(GPIO.BOARD) # Jetson Nano 핀 번호 사용하는 것을 명시
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)  # 초당 50(50Hz)번의 PWM 신호가 반복(20ms마다)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 초당 1000(1000Hz)번의 PWM 신호가 반복
servo.start(0)
dc_motor_pwm.start(0)

angle = 86

# 모델 로딩 함수
def load_model():
    # 사전 훈련된 모델 불러오기
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3개의 클래스 (0: 왼쪽, 1: 중간, 2: 오른쪽) 설정
    model.load_state_dict(torch.load("best_model.pth"))  # 학습된 모델 파일 경로
    model.eval()
    return model

# 이미지의 크기 조정
def img_filter(image):
    height, width, _ = image.shape
    # ROI 설정 (도로 영역만 남기기)
    roi_top = int(height * 0.3)  # 상단 60% 잘라내기 (도로만 남기기)
    roi = image[roi_top:height, :]
    
    old_width, old_height = roi.size  # PIL에서는 size로 (너비, 높이)를 가져옵니다.

    # 비율 유지하면서 크기 조정
    ratio = min(224 / old_width, 224 / old_height)
    new_width = int(old_width * ratio)
    new_height = int(old_height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 검은색 배경의 새 이미지 생성
    new_image = Image.new("RGB", (224, 224), (0, 0, 0))  # 검은색 배경
    # 리사이즈된 이미지를 중앙에 배치
    new_image.paste(resized_image, ((224 - new_width) // 2, (224 - new_height) // 2))
    

# 데이터 전처리 함수
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean=[0.2978, 0.3037, 0.2965], std=[0.3057, 0.3072, 0.3034])  # 정규화
    ])
    
    image = preprocess(img_filter(image)).unsqueeze(0)  # 배치 차원 추가
    return image

# 화살표를 추가하여 각도를 표시하는 함수

def draw_arrow(frame, angle):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    length = 50  # 화살표 길이
    angle_rad = np.radians(angle - 90)  # OpenCV 좌표계 기준으로 각도 변환
    end_x = int(center_x + length * np.cos(angle_rad))
    end_y = int(center_y - length * np.sin(angle_rad))
    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
    return frame
# 예측 함수
def predict(image, model, device):
    image = transform_image(image).to(device)  # 이미지를 텐서로 변환하고, GPU로 이동
    with torch.no_grad():
        outputs = model(image)
    _, preds = torch.max(outputs, 1)  # 가장 높은 확률을 가진 클래스 예측
    return preds.item()

# 서보 각도를 설정하는 함수
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)  # 회전 각도를 조절
    time.sleep(0.02)  # 서보가 위치에 도달할 시간 허용
    servo.ChangeDutyCycle(0)  # 진동을 피하기 위해 신호를 꺼줌

# 차량 제어 및 예측 기반 서보 모터 조정 함수
def car_control(model, device):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output_video.avi", fourcc, 20.0, (640, 480))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 모델 예측 수행
        pred_class = predict(frame, model, device)
        
        # 예측된 클래스에 따라 서보 모터 각도 조정
        if pred_class == 0:
            angle = 60  # 클래스 0 -> 서보 각도 60도 (왼쪽)
            print("Predicted Class: 0, Turning Left")
        elif pred_class == 1:
            angle = 86  # 클래스 1 -> 서보 각도 (중앙)
            print("Predicted Class: 1, Centered (90 degrees)")
        elif pred_class == 2:
            angle = 120  # 클래스 2 -> 서보 각도 120도 (오른쪽)
            print("Predicted Class: 2, Turning Right")
        
        # 화살표 추가
        frame_with_arrow = draw_arrow(frame, angle)
        
        # 프레임을 저장
        out.write(frame_with_arrow)

        # 영상 디스플레이
        # cv2.imshow("Car Control", frame_with_arrow)
        
        set_servo_angle(angle)  # 서보 모터 각도 설정
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)  # + 가 HIGH면 정방향으로 회전
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        dc_motor_pwm.ChangeDutyCycle(50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.02)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 메인 함수
if __name__ == "__main__":
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)

    # 차량 제어 및 예측 기반 서보 모터 조정 스레드
    car_thread = threading.Thread(target=car_control, args=(model, device), daemon=True)
    car_thread.start()

    car_thread.join()  # 차량 제어 스레드 종료 대기

    # GPIO 정리
    servo.stop()
    GPIO.cleanup()
