# JellyLab Project

# 1. Korean_handle

- 한국어를 다루기 위한 Class
- class KoreanHandler
  - 한글을 자모로 분리
- class KoreanDistance
  - 자모로 바꾼 한글 정보의 편집거리를 구해주는 알고리즘

# 2. Crawl_datapreprocess

- 다음 뉴스 기사를 crawling하기 위한 자동화 툴
- crawling 후 전처리 코드

# 3. fully_connected

- 한국어 intent classification을 위한 fully connnected layer(Tensorflow)

# 4. Skin_detection

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
