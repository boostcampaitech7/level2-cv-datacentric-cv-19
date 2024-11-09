# 📄 다국어 영수증 OCR Text Detection 대회

---

## 📌 프로젝트 소개
![image](https://github.com/user-attachments/assets/90416895-7430-4d0e-8d6b-325636ffb7b2)

본 프로젝트는 카메라로 촬영된 영수증 이미지에서 텍스트를 자동으로 인식하는 **OCR (Optical Character Recognition)** 기술을 기반으로 합니다. OCR은 사람이 직접 쓰거나 이미지 속 텍스트를 컴퓨터가 인식할 수 있도록 하는 컴퓨터 비전 기술입니다. 본 대회에서는 **중국어, 일본어, 태국어, 베트남어**로 작성된 영수증 이미지에서 **텍스트 검출 (text detection)** 작업을 수행합니다.

---

## 📊 평가 방법
![image](https://github.com/user-attachments/assets/f2534fc4-9684-48ce-ab43-1884079f3f75)

- 본 대회는 **DetEval 방식**으로 평가됩니다.
- **Area Recall** 및 **Area Precision**을 기반으로, 글자 영역을 정확하게 예측한 경우 점수가 부여됩니다.

  - **Area Recall** = (정답 박스와 예측 박스가 겹치는 영역) / (정답 박스의 영역)  
  - **Area Precision** = (정답 박스와 예측 박스가 겹치는 영역) / (예측 박스의 영역)

- **매칭 조건**:
![image](https://github.com/user-attachments/assets/8a066d84-4e7f-414b-8dd9-c23c8ac69974)

  - **One-to-one match**: 정답 박스 1개와 예측 박스 1개가 매칭
  - **One-to-many match**: 정답 박스 1개와 예측 박스 여러 개가 매칭될 경우 패널티 적용
  - **Many-to-one match**: 정답 박스 여러 개와 예측 박스 1개가 매칭될 경우 패널티 적용

```plaintext
|-- data
|   |-- chinese_receipt
|   |-- japanese_receipt
|   |-- pickle_data
|   |-- sample_submission.csv
|   |-- thai_receipt
|   `-- vietnamese_receipt
|-- dataset.py
|-- detect.py
|-- deteval.py
|-- east_dataset.py
|-- inference.py
|-- loss.py
|-- model.py
|-- pickle_data
|   |-- chinese_receipt_test.pkl
|   |-- chinese_receipt_train.pkl
|   |-- japanese_receipt_test.pkl
|   |-- japanese_receipt_train.pkl
|   |-- thai_receipt_test.pkl
|   |-- thai_receipt_train.pkl
|   |-- vietnamese_receipt_test.pkl
|   `-- vietnamese_receipt_train.pkl
|-- pickle_dataset.py
|-- pickle_train.py
|-- pths
|   `-- vgg16_bn-6c64b313.pth
|-- requirements.txt
|-- to_pickle.py
|-- train.py

```
## 🏃‍♂️ 모델 학습

### 기본 경로에서 실행
```plaintext
python train.py
```
### 데이터 경로 지정
```plaintext
python train.py --data_dir PATH_TO_DATA
```
### 하이퍼파라미터 조정 예시
```plaintext
python train.py --batch_size 2 --max_epoch 1 --save_interval 1
```

## 📚 참고 문헌
Zhou et al. “East: an efficient and accurate scene text detector.” CVPR 2017. Link
Simonyan and Zisserman. “Very deep convolutional networks for large-scale image recognition.” ICLR 2015. Link

## 🔍 모델 추론
```plaintext
python inference.py --data_dir PATH_TO_DATA
```
