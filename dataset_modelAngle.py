import os
import cv2
import numpy as np
import shutil
import torch
from torchvision import transforms

# 화살표 그리기 함수
def draw_arrow(frame, predicted_angle, index):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    length = 50  # 화살표 길이

    # 왼쪽 0도, 위쪽 90도, 오른쪽 180도를 반영하기 위한 각도 변환
    # angle_rad = np.radians(180 - angle)  # OpenCV의 기준에서 변환
    predicted_angle_rad = np.radians(180 - predicted_angle)  # 예측된 각도 변환

    # end_x = int(center_x + length * np.cos(angle_rad))
    # end_y = int(center_y - length * np.sin(angle_rad))  # OpenCV의 y축은 아래로 증가하므로 음수로 변환
    # cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)

    predicted_end_x = int(center_x + length * np.cos(predicted_angle_rad))
    predicted_end_y = int(center_y - length * np.sin(predicted_angle_rad))
    cv2.arrowedLine(frame, (center_x, center_y), (predicted_end_x, predicted_end_y), (0, 0, 255), 3, tipLength=0.3)

    text = f"Angle: {predicted_angle:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])
    
    image = preprocess(image).unsqueeze(0)  # 배치 차원 추가
    return image

def predict(image, model):
    image = transform_image(image) # 이미지를 텐서로 변환하고, GPU로 이동
    with torch.no_grad():
        outputs = model(image)
    _, preds = torch.max(outputs, 1)  # 가장 높은 확률을 가진 클래스 예측
    return preds.item()

# 이미지 검증 함수
def validate_images(folder_path, model):
    files = sorted(os.listdir(folder_path))  # 폴더 내 파일 정렬
    deleat_folder = os.path.join(folder_path, "deleat")
    os.makedirs(deleat_folder, exist_ok=True)

    index = 0
    total_index = 0
    while 0 <= index < len(files):
        file_name = files[index]
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 필터링
            index += 1
            continue

        # 이미지 읽기
        image = cv2.imread(file_path)
        if image is None:
            print(f"Cannot read file: {file_name}")
            index += 1
            continue

        # 폴더 이름에서 각도 추출
        # try:
        #     angle = int(os.path.basename(folder_path))  # 폴더 이름이 각도라고 가정
        # except ValueError:
        #     print(f"Invalid folder name for angle: {folder_path}")
        #     break

        # 이미지 전처리
        # image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

        # 모델 예측
        model.eval()  # 평가 모드로 설정
        with torch.no_grad():
            # output = model(image_tensor)  # 모델 예측
            pred_class = predict(image, model)  # 예측된 각도 (예: 회전 각도)
            if pred_class == 0:
                predicted_angle = 60
            elif pred_class == 1:
                predicted_angle = 75    
            elif pred_class == 2:
                predicted_angle = 86
            elif pred_class == 3:
                predicted_angle = 105
            elif pred_class == 4:
                predicted_angle = 120
        # 화살표 추가 (실제 각도, 예측된 각도)
        annotated_image = draw_arrow(image.copy(), predicted_angle, total_index)

        # 이미지 표시
        cv2.imshow("Image Validation", annotated_image)
        key = cv2.waitKey(1)

        if key == ord('d'):  # 삭제 키
            print(f"Deleting {file_name}...")
            shutil.move(file_path, os.path.join(deleat_folder, file_name))
            files.pop(index)  # 파일 리스트에서 제거
        elif key == ord('l'):  # 오른쪽 화살표 (다음 사진)
            index += 1
            total_index += 1
        elif key == ord('j'):  # 왼쪽 화살표 (이전 사진)
            index = max(index - 1, 0)
            total_index -= 1
        elif key == 27:  # ESC 키로 종료
            break

    cv2.destroyAllWindows()

# 모델 정의 (모델 구조가 정확히 어떻게 되어 있는지에 따라 수정해야 함)
# 예시로 기본 CNN 모델을 사용했으나 실제 사용하는 모델 구조로 변경해야 함
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 모델 로드 및 이미지 전처리 정의
# 모델 정의
model = SimpleCNN(num_classes=5)  # 모델 정의
weights = torch.load("best_model.pth")
model.load_state_dict(weights, strict=False)  # strict=False로 로드 시, 이름이 맞지 않는 레이어는 무시하고 로드
model.eval()


# 실행 예시
folder_path = r"C:/Users/USER/LineTrackingCar/Data/ResizeData/validation/3"  # 이미지 폴더 경로
validate_images(folder_path, model)
