# Jellypy

한국어 띄어쓰기를 위한 모듈입니다. open source인 https://github.com/haven-jeon/PyKoSpacing 을 이용하였고, PyKospacing 이 미처 띄어쓰지 못한 부분들을 konlpy(twitter)를 이용해 마무리했습니다. 

미리 한국어 json 사전을 만들어 한국어 편집거리를 이용하여 오타를 약간이나마 수정할 수 있도록 만들었습니다.

Default는 오타를 수정하지 않으므로, 원한다면 permitChange=True로 설정하시면 됩니다.

하지만 오타를 띄어쓰기 교정 후에 사용하는 것이므로, 띄어쓰기가 부정확해져 오타 교정 또한 부정확해집니다.

## Dependency(사전에 필요한 모듈)

```python
pip install git+https://github.com/haven-jeon/PyKoSpacing.git
pip install hgtk
pip install konlpy
pip install json
```

 이외 등등..

# Requires

```
python 3.x
```



## Example

```
아버지가방에들어가신다 ->  아버지가 방에 들어가신다 
너트와이스라고알아? -> 너 트와이스라고 알아 ?
너는무슨일을하는챗봇이니? -> 너는 무슨 일을 하는 챗봇이니 ?
너는취미가뭐야? -> 너는 취미가 뭐야 ?
```

```python
from korean_spacing import koreanSpacing
toSpace = "너는취미가뭐야"
koreanspacing = koreanSpacing(toSpace)
print(koreanspacing.makeOutput())
```

