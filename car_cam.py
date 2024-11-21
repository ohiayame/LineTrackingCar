import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard
import cv2
import threading

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

angle = 90

# 서보 각도를 설정하는 함수
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle) # 회전 각도를 조절
    time.sleep(0.02) # 서보가 위치에 도달할 시간 허용
    servo.ChangeDutyCycle(0) # 진동을 피하기 위해 신호를 꺼줌

# DC motor 방향과 속도 조절
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)  # + 가 HIGH면 정방향으로 회전
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)  # - 가 HIGH면 역방향으로 회전
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 영상 촬영
def video_capture():
    global angle
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    recoding = False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    save_dir = "cam/angle"
    image_number = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # 영상 녹화 처리
        if keyboard.is_pressed('r'):
            out = cv2.VideoWriter('cam/output2.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recoding = True
            print("Recoding started")
        if keyboard.is_pressed('q'):
            recoding = False
            out.release()
            print("Recoding stopped")
        if keyboard.is_pressed('c'):
            image_path = f"{save_dir}/{angle}_{image_number}.png"
            cv2.imwrite(image_path, frame)
            image_num += 1
        if recoding:
            out.write(frame) # frame 저장
            cv2.imshow('frame', frame)  # 영상 표시
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if recoding:
        out.release()
    cv2.destroyAllWindows()

# 키보드 입력에 따른 차량 제어
def car_control():
    # 서보 모터의 각도 초기화 -> 실제 모터도 90도로 세팅해야 함
    global angle
    set_servo_angle(angle)

    print("Use arrow keys to control. Press '0' to reset servo angle to 90.")
    while True:
        # DC모터 제어
        # up -> 속도 50, 전진
        if keyboard.is_pressed('up'):
            set_dc_motor(70, "forward")
            print("DC motor moving forward...")
        # down -> 속도 50, 후진
        elif keyboard.is_pressed('down'):
            set_dc_motor(50, "backward")
            print("DC motor moving backward...")
        else:
            set_dc_motor(0, "forward")  # Stop motor
            print("DC motor stopped.")

        # 서보모터 제어
        if 0 < angle < 180:
            # left -> 현제 각도에서 -1도한 각도로 설정
            if keyboard.is_pressed('left'):
                angle -= 1
                set_servo_angle(angle)
            # right -> 현제 각도에서 +1도한 각도로 설정
            elif keyboard.is_pressed('right'):
                angle += 1
                set_servo_angle(angle)
            print(f"Current angle: {angle} degrees")

        # 0 -> 90도로 설정
        if keyboard.is_pressed('0'):
            angle = 90
            set_servo_angle(angle)
            print("Servo angle reset to 90 degrees.")
            time.sleep(0.1)

        time.sleep(0.02)  # CPU 과부하 방지를 위한 짧은 지연

# 메인 함수
if __name__ == "__main__":
    # 영상 촬영 스레드
    video_thread = threading.Thread(target=video_capture, daemon=True)
    video_thread.start()

    # 차량 제어 스레드
    car_thread = threading.Thread(target=car_control, daemon=True)
    car_thread.start()

    # 메인 스레드가 종료되지 않도록 대기
    car_thread.join()
    video_thread.join()

    # GPIO 정리
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
