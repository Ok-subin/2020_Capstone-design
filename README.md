# 2020_Capstone design
 
# OCR (with EMNIST Training Model)

 &nbsp;&nbsp;&nbsp;2020학년도 2학기 [데이터분석 캡스톤 디자인] 수업의 프로젝트 과제로 EMNIST 학습을 통한 효율적인 이미지 속 문자 인식 모델을 구성하기로 한다.
&nbsp;&nbsp;&nbsp;MNIST 데이터의 숫자를 분류하는 연구는 이미지를 분석하고 학습시켜 분류시키는 컴퓨터 비전 연구 분야에서 기초적인 연구로 많이 학습된다. 이에 대한 연구는 많이 진행되었고, 이를 효율적으로 학습시키는 모델들 역시 다양하게 구성되어 있다. 이번 캡스톤 디자인에서는 MNIST 데이터를 변형시킨 숫자와 문자로 구성된 EMNIST 데이터를 학습시키는 모델을 구현한 후, 이미지 속에서 문자 영역을 찾아내어 bounding box로 씌우고 그 문자를 인식하려고 한다. 
 &nbsp;&nbsp;&nbsp;즉, 이번 프로젝트의 큰 과제는 1) 이미지 속에서 문자 영역을 추출하는 것과 2) 추출한 문자를 인식하는 것이다. 두 알고리즘을 통합하여 하나의 모델로 완성시키는 것을 목표로 한다.


## 1. 모델 설계

&nbsp;&nbsp;모델의 전반적인 구조는 [그림 1]과 같다.<br/>
![model](model.JPG)<br/>
[그림 1] 모델의 구조

**1.1 이미지 속 문자 영역 추출**
&nbsp;&nbsp;첫 번째 과정은 이미지 속에서 문자를 찾아내는 과정이다. 현재 프로젝트에서는 오픈소스인 CRAFT와 Tesseract 오픈소스를 사용해본 후, 성능을 평가했다. EMNIST가 손글씨 데이터이기 때문에 손글씨 이미지를 인식시켜본 결과, CRAFT가 더 성능이 우수하다고 판단되어 이를 사용하기로 했다.
<br/>

**1.2 추출한 문자 인식**
&nbsp;&nbsp;이전 과정을 통해 추출한 문자 영역을 문자 인식 모델에 입력하여 어떤 문자/숫자인지 판별해내는 것이 최종 목표였기에, 문자 인식 모델을 구현해야 한다. 모델 학습을 위해 사용한 dataset은 EMNIST의 ByMerge dataset이며, CNN 구조를 활용한다. CNN의 여러 모델 중 1) ResNet, 2) VGGNet, 3) Inception 모델을 사용해보고, 성능을 비교해 모델을 완성시킨다.
<br/>

**#EMNIST dataset**
&nbsp;&nbsp;본 모델에서 문자 인식 모델을 구현하기 위해 사용한 EMNIST dataset은 ‘NIST Special Database 19’로부터의 필기 숫자/문자 데이터를 28x28 픽셀 이미지 형식으 로 바꾼 dataset으로, 구조는 MNIST dataset과 동일하다. 해당 dataset은 ByClass, ByMerge, Balanced, Letters, Digits, MNIST인 6가지의 다른 형태의 dataset으로 나뉜다. 
&nbsp;&nbsp;6가지 dataset 중에서 데이터 개수가 814,255개로 가장 많은 ByClass와 ByMerge 중에서 유사한 대소문자 클래스를 합병한 ByMerge dataset을 사용한다. ByMerge dataset의 구성 예시는 [그림 2]와 같다. 각 이미지 아래의 label 값은 숫자, 알파벳 label에 대한 아스키코드 값이다.

![emnist](/emnist.png)<br/>
[그림 2] EMNIST - ByMerge Dataset의 구성 예시


## 2. 모델 구현
**2.1 이미지 속 문자 영역 추출**
&nbsp;&nbsp;오픈소스 코드 상에는 word level 영역의 이미지를 저장하는 코드는 존재하지만, 본 프로젝트를 위해 필요한 character level 영역의 이미지를 저장하는 코드는 없기 때문에 해당 함수를 추가 구현했다. 아래의 함수를 추가로 생성하면서 임계값을 수정해주면 기존의 word level로 추출되던 문자 이미지를 character level로 추출할 수 있게 된다. 아래 함수는 CRAFT 프로그램 내의 'detection.py' 등 기존에 word level 결과를 생성하던 곳에 추가해주어야 한다. 


get_result_img 함수 추가
```
def get_result_img(image, score_text, score_link, text_threshold=0.68, link_threshold=0.4, low_text=0.08, ratio_w=1.0, ratio_h=1.0):  
    copyimg = image   
    boxes = getDetBoxes2(score_text, score_link, text_threshold, link_threshold, low_text, s=False)  
    boxes = adjustResultCoordinates2(copyimg, boxes, 1.0, 1.0)  
    file_utils.saveResult('text_image_char.jpg', image, boxes, dirname='.')
```


main 함수
```
import easyocr  
import PIL  
from PIL import ImageDraw  
im = PIL.Image.open(이미지 주소)  
import numpy as np  
import matplotlib.pyplot as plt  
  
// Reader class로 초기화  
result = reader.readtext(이미지 주소)  
  
// main 실행 함수
resultImage = draw_boxes(im, result)  
resultImage = resultImage.convert("RGB")    --> RGB 값으로 변환해주어야 이미지 색상이 제대로 추출됨을 주의해야 한다.
resultImage.save(저장 이미지 주소)
```
&nbsp;&nbsp;main 함수는 이후 2.2의 문자 인식 모델과 합쳐져 하나의 모델로 구현될 때, 수정된다.
<br/>


**2.2 추출한 문자 인식**
&nbsp;&nbsp;CNN 모델 몇 가지를 구현한 후, EMNIST dataset에 입력했을 때 성능이 가장 좋은 모델을 사용하기로 결정했다. 각 모델을 구현한 후, 정확도를 높이기 위해 layer의 추가/삭제를 통한 구조의 변경 또는 learning rate, batch size 등의 parameter의 조정의 과정을 거쳤다. 해당 과정을 통해 생성된 가장 높은 validation accuracy를 가진 모델을 선택한다.
&nbsp;&nbsp;ResNet 18, VGGNet 16, Inception의 구현을 통해 EMNIST dataset을 학습시킨 결과의 validation accuracy는 표1과 같다. 소숫점 셋째자리에서 반올림한 값이다. 큰 차이는 나지 않지만, VGGNet이 accuracy가 가장 높은데다가 layer를 변경하기 쉽게 구현되어 있어서 VGGNet을 모델로 선택하였다.


|Model|Validation Accuracy|
|:------:|:---:|
|ResNet 18|87.86%|
|VGGNet 16|테스트2|
|VGGNet 18|테스트2|
|Inception|테스트2|
<br/>[표 1] CNN 모델의 validation accuracy

<br/>

**2.3 최종 모델 구현**
&nbsp;&nbsp; **2.1**과 **2.2**에서 구현된 각각의 모델을 하나로 통합해주는 과정이다. 


## 3. 결과

&nbsp;&nbsp; 모델의 실험 결과를 위한 2개의 이미지가 있다. [그림 3]는 테스트를 위해 다운받은 이미지, [그림 4]은 테스트를 위해 직접 쓴 손글씨 이미지이다.

**원본 이미지**

![original 01](./ReadMe/original_01.jpg)<br/>
[그림 3] 원본이미지 1

![original 02](./ReadMe/original_02.jpg)<br/>
[그림 4] 원본이미지 2

<br/>

**3.1 이미지 속 character 영역 인식 및 추출**
&nbsp;&nbsp;[그림 5], [그림 6]는 각 원본이미지에 대해서 CRAFT 과정을 거친 후, bounding box를 그려 출력한 결과이다.

![boudingBox_01](./ReadMe/boudingBox_01.jpg)<br/>
[그림 5] 원본이미지 1에서 character 영역 인식 및 bounding box 생성

![boudingBox_02](./ReadMe/boudingBox_02.jpg)<br/>
[그림 6] 원본이미지 2에서 character 영역 인식 및 bounding box 생성

<br/>
&nbsp;&nbsp;생성된 각 character level의 bounding box별로 잘라서 별도의 character 이미지로 모두 저장한다. [그림 7], [그림 8]은 두 이미지의 character가 별도로 폴더에 저장된 모습이다.

![character_01](./ReadMe/character_01.png)<br/>
[그림 7] 원본이미지 1의 각 character 영역을 잘라 별도의 이미지로 저장

![character_02](./ReadMe/character_02.jpg)<br/>
[그림 8] 원본이미지 2의 각 character 영역을 잘라 별도의 이미지로 저장


## 4. 사용 방법
&nbsp;&nbsp;프로그램을 사용하려면 소스코드 전체를 다운받은 후, 'modelMain.py' 파일을 수정해야 한다. 수정사항은 다음과 같으며, 수정이 필요한 부분은 해당 파일 내부에 주석으로 표시를 해놓았다.
 
 &nbsp;&nbsp;&nbsp;&nbsp;1) 테스트할 이미지의 주소 : imageAdd 변수
 &nbsp;&nbsp;&nbsp;&nbsp;2) character별 이미지의 저장 주소 : saveAdd 변수
 &nbsp;&nbsp;&nbsp;&nbsp;3) bounding box가 그려진 이미지의 저장 주소 : boudingBoxAdd 변수
 &nbsp;&nbsp;&nbsp;&nbsp;4) 사용할 모델의 h5 파일이 저장된 주소 : model 변수에 저장될 함수 load_model의 parameter 값
   
