import Jetson.GPIO as GPIO
import time
import subprocess

# Set the sudo password as a variable for easy updating
sudo_password = ""

# Function to run shell commands with the sudo password
def run_command(command):
    # Form the full command with password input
    full_command = f"echo {sudo_password} | sudo -S {command}"
    # Execute the command in the shell
    subprocess.run(full_command, shell=True, check=True)

# Check if busybox is installed; if not, install it
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# Define devmem commands
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

# Execute each devmem command
for command in commands:
    run_command(command)

# Set up GPIO pins for servo 
# GPIO 핀 설정
servo_pin = 33  # PWM-capable pin for servo motor

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

# Configure PWM on servo 
# PWM(펄스 폭 변조)을 설정하여 서보 모터를 제어 50Hz로 설정하고 초기값을 0으로 시작
servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo motor
servo.start(0)

# Function to set servo angle
# 서보 각도 설정 함수
def set_servo_angle(angle):
    # Calculate duty cycle based on angle (0 to 180 degrees)
    duty_cycle = 2 + (angle / 18)
    # angle = 0일 때: duty_cycle = 2
    # angle = 90일 때: duty_cycle = 7
    # angle = 180일 때: duty_cycle = 12
    servo.ChangeDutyCycle(duty_cycle)  # 서보 모터의 각도를 설정
    time.sleep(0.5)  # Allow time for the servo to reach position
    servo.ChangeDutyCycle(0)  # Turn off signal to avoid jitter

    
# Example usage: Rotate servo 
try:
    # Rotate servo from 0 to 180 degrees and back to 0
    #for angle in range(0, 181, 10):  # Move from 0 to 180 in 10-degree steps
    #    set_servo_angle(angle)
    #for angle in range(180, -1, -10):  # Move from 180 back to 0 in 10-degree steps
    #    set_servo_angle(angle)
    while True:
        inputValue = int(input("put angle value : "))
        set_servo_angle(inputValue)
finally:
    # Stop all PWM and clean up GPIO
    servo.stop()
    GPIO.cleanup()









def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track
    
    return float(reward)