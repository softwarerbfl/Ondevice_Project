import os
import random
import shutil

# 원본 이미지 경로와 새로운 폴더 경로 설정
original_folder = '../coco/images/train2017'
new_folder = '../coco/images/train2017_10000'

# 새로운 폴더가 없다면 생성
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 원본 폴더에서 모든 이미지 파일 목록 가져오기 (jpg, png 확장자만)
all_images = [f for f in os.listdir(original_folder) if f.endswith(('.jpg', '.png'))]

# 원본 이미지 개수 확인
if len(all_images) < 20000:
    raise ValueError(f'원본 이미지 수가 10000개보다 적습니다: {len(all_images)}개')

# 임의로 10,000개의 이미지 선택
sampled_images = random.sample(all_images, 10000)

# 선택한 이미지를 새로운 폴더로 복사
for image in sampled_images:
    shutil.copy(os.path.join(original_folder, image), os.path.join(new_folder, image))

print(f'Successfully copied {len(sampled_images)} images to {new_folder}')
