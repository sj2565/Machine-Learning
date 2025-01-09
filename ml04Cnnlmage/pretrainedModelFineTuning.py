import os, json, pickle, math

from datetime import datetime
# from tensorflow.python.keras.applications.model import preprocess_input
# from tensorflow.python.keras.applications.model import VGG16
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger

from utility.utils import show_train_samples
from utility.utils import load_random_imgs
from utility.utils import show_test_samples
from utility.utils import plot_learningcurve_from_csv
from keras.applications.vgg16 import VGG16, preprocess_input

# 기존의 1000 클래스의 출력을 사용하지 않으므로
# include_top=False 옵션을 사용하여 출력층을 포함하지 않는 상태로 읽어 들입니다.
model = VGG16(include_top=False, input_shape=(224, 224, 3))
# model = VGG16()

# 모델의 요약 확인. 출력층이 포함되지 않은 것을 알 수 있다.
# print(model.summary())
# print( '-' * 40 )

# 모델을 편집해서 네트워크를 생성할 함수 정의
def build_transfer_model(model):
    # 호출한 모델을 사용해서 새로운 모델을 작성한다.
    model = Sequential(model.layers)

    # 호출한 가중치의 일부는 재학습하지 않도록 설정
    # 여기서는 추가한 층과 출력층에 가까운 층의 가중치만 재학습
    for layer in model.layers[:15]:
        layer.trainable = False

    # 추가할 출력 부분의 층을 구축
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

model = build_transfer_model(model)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

print(model.summary()) # 모델 요약 확인
print( '-' * 40 )

# 학습용 데이터의 내용을 랜덤으로 표시
img_dir = './img/shrine_temple'
show_train_samples(img_dir, classes=('shrine', 'temple'), seed=1)

# 생성기 생성(for 학습용 이미지)
idg_train = ImageDataGenerator(
    rescale=1/255.,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function = preprocess_input
)

# 훈련용 데이터(학습할 때 이용)
img_itr_train = idg_train.flow_from_directory(
    './img/shrine_temple/train', 
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# 검증용 데이터(학습할 때 이용)
img_itr_validation = idg_train.flow_from_directory(
    './img/shrine_temple/validation', 
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# 모델 저장용 폴더 준비 : 실행 일시를 나타내는 하위 폴더
model_dir = os.path.join(
    'models', datetime.now().strftime('%y%m%d_%H%M')
)

os.makedirs(model_dir, exist_ok=True)
print('model_dir:', model_dir)  # 저장할 폴더 이름 표시
print( '-' * 40 )

dir_weights = os.path.join(model_dir, 'weights')
os.makedirs(dir_weights, exist_ok=True)

model_json = os.path.join(model_dir, 'model.json')

with open(model_json, 'w') as f:
    json.dump(model.to_json(), f)

# 학습할 때의 정답 레이블 저장
model_classes = os.path.join(model_dir, 'classes.pkl')
with open(model_classes, 'wb') as f:
    pickle.dump(img_itr_train.class_indices, f)

# 콜백의 설정
cp_filepath =  os.path.join(dir_weights, 'ep_{epoch:02d}_ls_{loss:.1f}.h5')
cp = ModelCheckpoint(cp_filepath, monitor='loss', verbose=0,
                     save_best_only=False, save_weights_only=True, 
                     mode='auto', period=5)

csv_filepath =  os.path.join(model_dir, 'loss.csv')
csv = CSVLogger(csv_filepath, append=True)

# 미니 배치를 몇 개 학습하면 1에폭이 되는지 계산(학습할 때 지정할 필요가 있다)
batch_size = 16
steps_per_epoch = math.ceil(
    img_itr_train.samples/batch_size
)
validation_steps = math.ceil(
    img_itr_validation.samples/batch_size
)
n_epoch = 30

# 모델 학습
history = model.fit_generator(
    img_itr_train, 
    steps_per_epoch=steps_per_epoch, 
    epochs=n_epoch,  # 학습할 에폭수
    validation_data=img_itr_validation, 
    validation_steps=validation_steps,
    callbacks = [cp, csv]
)

# **예제 코드 6.23:학습한 모델을 사용해서 예측** 
# 학습 결과를 산출(추론)
test_data_dir = './img/shrine_temple/test/unknown'
x_test, true_labels = load_random_imgs(
    test_data_dir, 
    seed=1
)
x_test_preproc= preprocess_input(x_test.copy())/255.
probs = model.predict(x_test_preproc)

print(probs)
print( '-' * 40 )

# 클레스 레이플 확인
print('class_indices:', img_itr_train.class_indices)
print( '-' * 40 )

# 평가용 이미지 표시
show_test_samples(
    x_test, probs, 
    img_itr_train.class_indices, 
    true_labels
)

# 학습 곡선 표시
plot_learningcurve_from_csv(csv_filepath)