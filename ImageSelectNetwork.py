import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os

class ImageSelectionNetworkEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageSelectionNetworkEfficientNet, self).__init__()
        # Pre-trained EfficientNet
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Replace the last fully connected layer
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes) 
    
    def forward(self, x):
        return self.model(x)

# 모델 생성
model = ImageSelectionNetworkEfficientNet(num_classes=2)

# 이미지 전처리: 이미지를 모델 입력 크기(224x224)에 맞게 변환
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 로드 및 전처리
img_dir = '../coco/images/train2017'
usage_file = './classification/usage.txt'
trash_file = './classification/trash.txt'
# 결과 저장 파일을 초기화
with open(usage_file, 'w') as uf, open(trash_file, 'w') as tf:
    # 이미지 디렉토리에서 모든 파일 읽기
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name) # 이미지 하나의 전체 경로
        
        # 이미지가 .jpg, .png 등의 확장자인지 확인
        if not img_path.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        try:
            # 이미지 로드 및 전처리
            img = Image.open(img_path)

            # 만약 흑백 이미지이면, 3채널로 변환
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_tensor = preprocess(img).unsqueeze(0)  # 배치 차원 추가 (1, 3, 224, 224)
            
            # 모델에 이미지 입력하여 추론
            with torch.no_grad():
                output = model(img_tensor)
            _, predicted = torch.max(output, 1)

            # 분류 결과에 따라 파일 경로를 각 파일에 저장
            # usage로 분류
            if predicted.item() == 0:  
                uf.write(f"{img_path}\n")
            # trash로 분류
            else:  
                tf.write(f"{img_path}\n")
                
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")