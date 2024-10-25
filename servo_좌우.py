import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard

# sudo 비밀번호를 변수로 설정
sudo_password = "1212"

# sudo 비밀번호로 셸 명령어를 실행하는 함수
def run_command(command):
    try:
        full_command = f"echo {sudo_password} | sudo -S {command}"
        subprocess.run(full_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"명령어 '{command}' 실행 중 오류 발생: {e}")

# busybox가 설치되어 있는지 확인하고, 없으면 설치
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox가 이미 설치되어 있습니다.")
except subprocess.CalledProcessError:
    print("busybox를 찾을 수 없습니다. 설치 중...")
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

# 서보를 위한 GPIO 핀 설정
servo_pin = 33  # 서보 모터를 위한 PWM 핀

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

# 서보 PWM 설정
servo = GPIO.PWM(servo_pin, 50)  # 서보 모터를 위한 50Hz
servo.start(0)

# 서보 각도를 설정하는 함수
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # 서보가 위치에 도달할 시간 허용
    servo.ChangeDutyCycle(0)  # 진동을 피하기 위해 신호를 꺼줌

# 서보 모터를 멈추는 함수
def stop_servo():
    servo.ChangeDutyCycle(0)  # 서보의 PWM 신호 멈춤

# 키보드 입력에 따라 서보 모터를 제어하는 메인 루프
try:
    print("왼쪽 (←) 또는 오른쪽 (→) 화살표 키를 눌러 움직이세요. 스페이스 바를 눌러 각도를 입력하세요.")
    while True:
        if keyboard.is_pressed('left'):
            print("왼쪽으로 회전")
            set_servo_angle(45)  # 왼쪽으로 회전하는 각도
            time.sleep(0.5)  # 지정한 시간 동안 대기
        elif keyboard.is_pressed('right'):
            print("오른쪽으로 회전")
            set_servo_angle(135)  # 오른쪽으로 회전하는 각도
            time.sleep(0.5)  # 지정한 시간 동안 대기
        elif keyboard.is_pressed('space'):
            angle = input("각도 입력 (0부터 180까지): ")
            try:
                angle = int(angle)
                if 0 <= angle <= 180:
                    print(f"{angle}도로 설정 중...")
                    while keyboard.is_pressed('space'):  # 스페이스 바를 누르고 있는 동안
                        set_servo_angle(angle)  # 입력한 각도로 서보를 설정
                        time.sleep(0.1)  # 작은 지연으로 CPU 부하를 줄임
                else:
                    print("각도는 0부터 180 사이여야 합니다.")
            except ValueError:
                print("잘못된 입력입니다. 숫자를 입력하세요.")
        time.sleep(0.1)  # CPU 과부하 방지를 위한 짧은 지연
finally:
    print("정리 중...")
    # PWM 정지 및 GPIO 정리
    servo.stop()
    GPIO.cleanup()