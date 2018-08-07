# JellyLab Project

# 1. Korean_handle

- 한국어를 다루기 위한 Class 입니다.
- class KoreanHandler
  - 한글을 자모로 분리합니다.
- class KoreanDistance
  - 자모로 바꾼 한글 정보의 편집거리를 구해주는 알고리즘입니다.

# 2. Jellpy

- korean_handle과 다른 여러 모듈들을 이용해 한국어 띄어쓰기를 위한 module
- 자세한 사항은 Jellypy안의 README.md 를 참고해주세요.

# 3. Crawl_datapreprocess

- 다음 뉴스 기사를 crawling하기 위한 자동화 툴입니다.
- crawling 후 전처리 코드도 포함되어있습니다.

# 4. fully_connected

- 한국어 intent classification을 위한 fully connnected layer(Tensorflow) 입니다.

# 5. Skin_detection

### Human Skin detection Algorithm

HSV와 HSL 색 공간을 이용하여 Human Skin Color 의 Range를 정의한 후, Pixel 단위로 range 밖의 색들은 검은색으로 처리하여 사진에서 Skin 만 남길 수 있습니다.

### Requirements

- Python 3


### 세부사항

detection_a.py와 detection_skin.py 가 같이 쓰이고, detection_hsl.py와 detection_skin_hsl.py 가 같이 쓰입니다.

- detection_a.py 는 hsv 색 공간을 이용한 RangeColor detector입니다.
- detection_skin_hsl.py는 hsl 색 공간을 이용한 RangeColor detector 입니다.

### How to Run

```python
python detection_skin.py
```



# 6. Seq2Seq

- 한국어 intent classification을 위한 seq2seq 입니다.
- encoder로 일반 채팅 데이터가 들어가고, decoder로 intent를 output으로 내도록 learning 합니다/
- Testset에서는 60~70%의 성능을 나타냈지만, 실전데이터에서는 좋지 못한 성능을 냈습니다. 
- 한국어 딥러닝 쉽지 않았습니다.



# 7. Cnn_nlp

- 한국어 intent classification을 위한 cnn 입니다.
- Seq2Seq과 마찬가지로 일반 채팅데이터를 단어 단위로 잘라 convolution Layer로 넣어주고, output으로는 16개의 intent를 output으로 내도록 learning 합니다.
- intent의 개수가 너무 많아서 learning이 잘 되지는 않았습니다.
- 한국어 딥러닝 어렵습니다.

