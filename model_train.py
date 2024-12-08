import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# 데이터 전처리 (이미지 크기 조정, 텐서 변환, 정규화)
transform = transforms.Compose([ # 이미지 크기 맞추기
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
])

# 데이터셋 경로 설정
data_dir = r'C:/Users/USER/LineTrackingCar/ResizeData'  # 데이터셋 경로

# 훈련, 검증, 테스트 데이터 로딩
train_data = datasets.ImageFolder(os.path.join(data_dir, 'training'), transform=transform)
val_data = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=transform)
# test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# 데이터로더 (배치 크기 설정)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ResNet18 모델 불러오기 (사전학습된 가중치 사용)
model = models.resnet18(pretrained=True)

# 출력층 수정 (데이터셋에 맞는 클래스 수로 변경)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_data.classes))  # train_data.classes는 클래스 수

# GPU가 사용 가능한 경우 GPU로 모델 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수와 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습 함수
# 모델 학습 함수 (검증 데이터를 훈련에 포함)
def train_with_validation(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 훈련 단계
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 최적화 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 역전파
            loss.backward()
            optimizer.step()

            # 통계
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            if total % 64 == 0:  # 특정 배치마다 예측 출력
                print(f"Predictions: {preds.cpu().numpy()}, Actual Labels: {labels.cpu().numpy()}")



        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / total
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 검증 단계 (검증 데이터로 성능 평가)
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                print(f"Validation Predictions: {preds.cpu().numpy()}, Actual Labels: {labels.cpu().numpy()}")


        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / val_total
        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 모델 성능이 향상되었으면 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    print(f'Best Validation Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    
    # 모델 저장
    torch.save(model.state_dict(), 'best_model.pth')
    print("Best model saved as 'best_model.pth'")
    return model

# 모델 학습 및 저장 (검증 데이터 사용)
model = train_with_validation(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)


# # 테스트 데이터로 성능 평가
# def evaluate_model(model, test_loader):
#     model.eval()
#     corrects = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             corrects += torch.sum(preds == labels.data)
#             total += labels.size(0)

#     accuracy = corrects.double() / total
#     print(f'Test Accuracy: {accuracy:.4f}')

# # 테스트 데이터로 평가
# evaluate_model(model, test_loader)
# 