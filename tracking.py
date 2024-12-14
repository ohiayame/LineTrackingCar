import Jetson.GPIO as GPIO
import time
import subprocess
import threading
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2

# sudo 명령어 실행 함수
def run_command(command):
    full_command = f"sudo {command}"
    subprocess.run(full_command, shell=True, check=True)

# busybox가 설치되어 있는지 확인하고, 없으면 설치
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# devmem 명령 실행
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

for command in commands:
    run_command(command)

# GPIO 핀 설정
GPIO.setwarnings(False)  # GPIO 경고 비활성화
servo_pin = 33  # 서보 모터 PWM핀
dc_motor_pwm_pin = 32  # DC 모터의 속도를 조절 PWM핀
dc_motor_dir_pin1 = 29  # GPIO 5, DC 모터의 방향을 제어 + (OUT1 + && IN1 + )
dc_motor_dir_pin2 = 31  # GPIO 6, DC 모터의 방향을 제어 - (OUT2 - && IN2 - )

GPIO.setmode(GPIO.BOARD)  # Jetson Nano 핀 번호 사용하는 것을 명시
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)  # 초당 50(50Hz)번의 PWM 신호가 반복(20ms마다)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 초당 1000(1000Hz)번의 PWM 신호가 반복
servo.start(0)
dc_motor_pwm.start(0)

angle = 90

# 모델 로딩 함수
def load_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5개의 클래스 (0,1: 왼쪽, 2: 중간, 3,4: 오른쪽) 설정
    try:
        state_dict = torch.load("/home/aym/aym_car/best_model.pth")
        if "fc.weight" in state_dict and state_dict["fc.weight"].shape[0] != 5:
            print("Adjusting model output layer to match checkpoint.")
            model.fc = nn.Linear(num_ftrs, state_dict["fc.weight"].shape[0])
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        exit(1)
    model.eval()
    return model

# 이미지의 크기 조정
def img_filter(image):
    height, width, _ = image.shape
    roi_top = int(height * 0.3)
    roi = image[roi_top:height, :]
    roi = cv2.resize(roi, (224, 224))
    return roi

# 데이터 전처리 함수
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(img_filter(image)).unsqueeze(0)
    return image

# 예측 함수
def predict(image, model, device):
    image = transform_image(image).to(device)
    with torch.no_grad():
        outputs = model(image)
    _, preds = torch.max(outputs, 1)
    return preds.item()

# 서보 각도를 설정하는 함수
def set_servo_angle(angle):
    try:
        duty_cycle = max(2, min(12, 2 + (angle / 18)))  # 제한된 범위에서 duty_cycle 계산
        print(f"Setting servo angle to {angle} (Duty Cycle: {duty_cycle})")
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.02)
        servo.ChangeDutyCycle(0)
    except Exception as e:
        print(f"Error in set_servo_angle: {e}")

# 차량 제어 및 예측 기반 서보 모터 조정 함수
def car_control(model, device):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # OpenCV로 카메라 열기 (0번 카메라)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로 해상도
    cap.set(cv2.CAP_PROP_FPS, 30)           # 초당 프레임 수

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    try:
        while True:
            ret, frame = cap.read()  # 프레임 읽기

            if not ret:
                continue

            angle = 86  # 기본값 설정

            pred_class = predict(frame, model, device)
            if pred_class == 0:
                angle = 60
                print("Predicted Class: 0, Turning Left")
            elif pred_class == 1:
                angle = 75
                print("Predicted Class: 1, Turning Left")
            elif pred_class == 2:
                angle = 86
                print("Predicted Class: 2, Centered (90 degrees)")
            elif pred_class == 3:
                angle = 105
                print("Predicted Class: 3, Turning Right")
            elif pred_class == 4:
                angle = 120
                print("Predicted Class: 4, Turning Right")

            print(f"Current predicted angle: {angle}")
            set_servo_angle(angle)

            # DC모터 
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
            dc_motor_pwm.ChangeDutyCycle(70)

            cv2.imshow("Car Control - Camera Feed", frame)  # 카메라 피드를 화면에 표시

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 눌렀을 때 종료
                break

            time.sleep(0.02)
    finally:
        cap.release()  # 카메라 릴리즈
        cv2.destroyAllWindows()  # OpenCV 윈도우 종료

# 메인 함수
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("cuda is true")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)

    try:
        car_thread = threading.Thread(target=car_control, args=(model, device), daemon=True)
        car_thread.start()
        car_thread.join()
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        try:
            servo.stop()
        except Exception as e:
            print(f"Error stopping servo: {e}")

        try:
            dc_motor_pwm.stop()
        except Exception as e:
            print(f"Error stopping DC motor PWM: {e}")

        try:
            GPIO.cleanup()
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")
