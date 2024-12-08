import cv2
import os
from PIL import Image


def resize_images_in_folder(input_folder, output_folder):
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 폴더 내 모든 파일 처리
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"): 
            # 원본 이미지 경로
            image_path = os.path.join(input_folder, filename)
            
            # 이미지 불러오기
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 불러오지 못했습니다: {image_path}")
                continue
            
            # 이미지 크기 확인
            height, width, _ = image.shape

            # ROI 설정 (도로 영역만 남기기)
            roi_top = int(height * 0.3)  # 상단 30% 잘라내기 (도로만 남기기)
            roi = image[roi_top:height, :]  # 아래쪽 70%만 사용
            
            ##
            # 자른 이미지 크기 확인
            new_height, new_width, _ = roi.shape

            # 비율 유지하면서 크기 조정
            ratio = min(224 / new_width, 224 / new_height)
            target_width = int(new_width * ratio)
            target_height = int(new_height * ratio)
            resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # 검은색 배경의 새 이미지 생성
            new_image = Image.new("RGB", (224, 224), (0, 0, 0))  # 검은색 배경
            # 리사이즈된 이미지를 중앙에 배치
            new_image.paste(resized_image, ((224 - target_width) // 2, (224 - target_height) // 2))
                

            # 결과 이미지 저장 경로
            output_path = os.path.join(output_folder, f"{filename}")

            # 결과 저장
            cv2.imwrite(output_path, new_image)
            print(f"전처리된 이미지를 저장했습니다: {output_path}")


# 예시: 폴더 경로와 원하는 크기 설정
dataset = ["training", "validation"]
lavel = ["65", "90", "120"]
target_width = 224
target_height = 224

for i in range(len(dataset)):
    for j in range(len(lavel)):
        
        input_folder = f'Dataset/{dataset[i]}/{lavel[j]}'  # 원본 이미지가 있는 폴더 경로
        output_folder = f'Dataset/ResizeData/{dataset[i]}/{lavel[j]}'  # 리사이즈된 이미지를 저장할 폴더 경로
    
        resize_images_in_folder(input_folder, output_folder)
print("모든 이미지의 전처리가 완료되었습니다.")