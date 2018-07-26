# Human Skin detection Algorithm

HSV와 HSL 색 공간을 이용하여 Human Skin Color 의 Range를 정의한 후, Pixel 단위로 range 밖의 색들은 검은색으로 처리하여 사진에서 Skin 만 남길 수 있습니다.

# Requirements

- Python 3

  

# 세부사항

detection_a.py와 detection_skin.py 가 같이 쓰이고, detection_hsl.py와 detection_skin_hsl.py 가 같이 쓰입니다.

- detection_a.py 는 hsv 색 공간을 이용한 RangeColor detector입니다.
- detection_skin_hsl.py는 hsl 색 공간을 이용한 RangeColor detector 입니다.

# How to Run

```python
python detection_skin.py
```

