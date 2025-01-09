import numpy as np
result = np.random.permutation(20)
print(result)

train=result[0:10]
print(train)

validation=result[10:15]
print(validation)

test=result[15:20]
print(test)

print('파일을 복사합니다 잠시만 기다려 주세요')

# 원본 이미지 파일이 들어 있는 폴더 (강아지와 고양이 각 2000개의 이미지)
origin_data_folder = '../datasets/cats_and_dogs'

# 학습용, 검증용, 테스트용을 위한 개별 폴더
target_folder = '../datasets/cats_and_dogs_random'

import os, shutil # shell utility
if os.path.exists(target_folder):
    # 해당 폴더가 존재하면 하위 폴더까지 모두 삭제한다.
    shutil.rmtree(target_folder)

os.mkdir(target_folder)    # 해당 폴더 생성

# 훈련용 데이터를 위한 폴더를 생성한다.
train_folder = os.path.join(target_folder, 'train')
os.mkdir(train_folder)

# 검증용 데이터를 위한 폴더를 생성한다.
validation_folder = os.path.join(target_folder, 'validation')
os.mkdir(validation_folder)

# 테스트용 데이터를 위한 폴더를 생성한다.
test_folder = os.path.join(target_folder, 'test')
os.mkdir(test_folder)

# 훈련용 고양이 사진 폴더
train_cats_folder = os.path.join(train_folder, 'cats')
os.mkdir(train_cats_folder)

# 훈련용 강아지 사진 폴더
train_dogs_folder = os.path.join(train_folder, 'dogs')
os.mkdir(train_dogs_folder)

# 검증용 고양이 사진 폴더
validation_cats_folder = os.path.join(validation_folder, 'cats')
os.mkdir(validation_cats_folder)

# 검증용 강아지 사진 폴더
validation_dogs_folder = os.path.join(validation_folder, 'dogs')
os.mkdir(validation_dogs_folder)

# 테스트용 고양이 사진 폴더
test_cats_folder = os.path.join(test_folder, 'cats')
os.mkdir(test_cats_folder)

# 테스트용 강아지 사진 폴더
test_dogs_folder = os.path.join(test_folder, 'dogs')
os.mkdir(test_dogs_folder)

#------------------------------------------------------------------------
image_su = 2000 # 강아지와 고양이 모두 2000개
cat_random = np.random.permutation(image_su)

# 고양이 이미지 1,000개 이미지를 train_cats_folder 폴더에 복사한다.
fnames = ['cat.{}.jpg'.format(i) for i in cat_random[0:1000]]
for fname in fnames:
    src = os.path.join(origin_data_folder, fname)
    dst = os.path.join(train_cats_folder, fname)
    shutil.copyfile(src, dst)

# 고양이 이미지 500개 이미지를 validation_cats_folder 폴더에 복사한다.
fnames = ['cat.{}.jpg'.format(i) for i in cat_random[1000:1500]]
for fname in fnames:
    src = os.path.join(origin_data_folder, fname)
    dst = os.path.join(validation_cats_folder, fname)
    shutil.copyfile(src, dst)

# 고양이 이미지 500개 이미지를 test_cats_folder 폴더에 복사한다.
fnames = ['cat.{}.jpg'.format(i) for i in cat_random[1500:2000]]
for fname in fnames:
    src = os.path.join(origin_data_folder, fname)
    dst = os.path.join(test_cats_folder, fname)
    shutil.copyfile(src, dst)

dog_random = np.random.permutation(image_su)

# 강아지 이미지 1,000개 이미지를 train_dogs_folder 폴더에 복사한다.
fnames = ['dog.{}.jpg'.format(i) for i in dog_random[0:1000]]
for fname in fnames:
    src = os.path.join(origin_data_folder, fname)
    dst = os.path.join(train_dogs_folder, fname)
    shutil.copyfile(src, dst)

# 강아지 이미지 500개 이미지를 validation_dogs_folder 폴더에 복사한다.
fnames = ['dog.{}.jpg'.format(i) for i in dog_random[1000:1500]]
for fname in fnames:
    src = os.path.join(origin_data_folder, fname)
    dst = os.path.join(validation_dogs_folder, fname)
    shutil.copyfile(src, dst)

# 강아지 이미지 500개 이미지를 test_dogs_folder 폴더에 복사한다.
fnames = ['dog.{}.jpg'.format(i) for i in dog_random[1500:2000]]
for fname in fnames:
    src = os.path.join(origin_data_folder, fname)
    dst = os.path.join(test_dogs_folder, fname)
    shutil.copyfile(src, dst)

print('파일 개수 확인')
print('훈련용 고양이 이미지 개수 : ', len(os.listdir(train_cats_folder)))
print('훈련용 강아지 이미지 개수 : ', len(os.listdir(train_dogs_folder)))

print('검증용 고양이 이미지 개수 : ', len(os.listdir(validation_cats_folder)))
print('검증용 강아지 이미지 개수 : ', len(os.listdir(validation_dogs_folder)))

print('테스트용 고양이 이미지 개수 : ', len(os.listdir(test_cats_folder)))
print('테스트용 강아지 이미지 개수 : ', len(os.listdir(test_dogs_folder)))