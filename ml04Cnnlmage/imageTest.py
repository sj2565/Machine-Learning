img_source = '../image/'   # 이미지가 있는 폴더 경로
img_target = '../myimage/'   # 이미지가 저장될 폴더 경로

img_dog = img_source + 'mydog.png'
print('원본 이미지 : ' +img_dog)

import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img

image32 = load_img(img_dog, target_size=(32, 32)) # 숫자가 높을수록 해상도가 좋아짐
print(type(image32))
plt.axis('off')
plt.figure()
plt.xticks([])   # 눈금 표시 안 함
plt.yticks([])
plt.imshow(image32)

filename = img_target + 'dog32.png'
plt.savefig(filename)
print(filename + '파일 저장됨')
#----------------------------------------------------------------------
image64 = load_img(img_dog, target_size=(64, 64)) # 숫자가 높을수록 해상도가 좋아짐
print(type(image64))
plt.axis('off')
plt.figure()
plt.xticks([])   # 눈금 표시 안 함
plt.yticks([])
plt.imshow(image64)

filename = img_target + 'dog64.png'
plt.savefig(filename)
print(filename + '파일 저장됨')
#----------------------------------------------------------------------
image128 = load_img(img_dog, target_size=(128, 128)) # 숫자가 높을수록 해상도가 좋아짐
print(type(image128))
plt.axis('off')
plt.figure()
plt.xticks([])   # 눈금 표시 안 함
plt.yticks([])
plt.imshow(image128)

filename = img_target + 'dog128.png'
plt.savefig(filename)
print(filename + '파일 저장됨')
#----------------------------------------------------------------------

from tensorflow.python.keras.preprocessing.image import img_to_array
arr_dog_128 = img_to_array(image128)
print(type(arr_dog_128))    # type 확인
print(arr_dog_128.shape)    # 128 * 128 * 3 배열 (칼라 : 3, 흑백 : 1)
# print(arr_dog_128)

from tensorflow.python.keras.preprocessing.image import array_to_img

# 저해상도 이미지를 생성해주는 함수
def drop_resolution(x, scale=3.0):
    img = array_to_img(x)
    size = (x.shape[0], x.shape[1])
    print('size : ', size)
    small_size = (int(size[0]/scale), int(size[1]/scale))
    print('small_size : ' , small_size)

    small_img = img.resize(small_size, 3)
    print('type(small_img) : ', type(small_img))
    plt.imshow(small_img)
    filename = img_target + 'drop_res_image(' + str(scale) + ').png'
    plt.savefig(filename)
    print(filename + '파일 저장됨')

drop_resolution(arr_dog_128)    # scale을 안 넣었으니 위에서 설정해 준 3.0
drop_resolution(arr_dog_128, scale=10.0)