# -*- coding: utf-8 -*-
import re
from gensim.models import Word2Vec
"""
    초성 중성 종성 분리 하기
	유니코드 한글은 0xAC00 으로부터
	초성 19개, 중상21개, 종성28개로 이루어지고
	이들을 조합한 11,172개의 문자를 갖는다.
	한글코드의 값 = ((초성 * 21) + 중성) * 28 + 종성 + 0xAC00
	(0xAC00은 'ㄱ'의 코드값)
	따라서 다음과 같은 계산 식이 구해진다.
	유니코드 한글 문자 코드 값이 X일 때,
	초성 = ((X - 0xAC00) / 28) / 21
	중성 = ((X - 0xAC00) / 28) % 21
	종성 = (X - 0xAC00) % 28
	이 때 초성, 중성, 종성의 값은 각 소리 글자의 코드값이 아니라
	이들이 각각 몇 번째 문자인가를 나타내기 때문에 다음과 같이 다시 처리한다.
	초성문자코드 = 초성 + 0x1100 //('ㄱ')
	중성문자코드 = 중성 + 0x1161 // ('ㅏ')
	종성문자코드 = 종성 + 0x11A8 - 1 // (종성이 없는 경우가 있으므로 1을 뺌)
"""
# 유니코드 한글 시작 : 44032, 끝 : 55199
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JONGSUNG_LIST_1 = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
SPECIAL_LIST = ['!', '?', '/', '^', '#','*']
class koreanHandler:
    def __init__(self, word):
        self.word = word

    # Word 를 자음과 모음 단위로 split
    def splitWord(self):
        split_keyword_list = list(self.word)
        result = list()
        for keyword in split_keyword_list:
            if keyword in CHOSUNG_LIST or keyword in JUNGSUNG_LIST or keyword in JONGSUNG_LIST:
                result.append(keyword)
            elif re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
                char_code = ord(keyword) - BASE_CODE
                char1 = int(char_code / CHOSUNG)
                result.append(CHOSUNG_LIST[char1])
                char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
                result.append(JUNGSUNG_LIST[char2])
                char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
                result.append(JONGSUNG_LIST[char3])
        # return result
        return "".join(result).replace(" ","")

    # Word에서 자음과 모음을 삭제
    def deleteJM(self):
        split_keyword_list = list(self.word)
        result = list()
        puctuation = ""
        for keyword in split_keyword_list:
          if keyword in CHOSUNG_LIST or keyword in JUNGSUNG_LIST or keyword in JONGSUNG_LIST_1:
            continue
          elif keyword in SPECIAL_LIST:
            puctuation = keyword
          else:
            result.append(keyword)
        return "".join(result), puctuation


"""
koreanHandler로 쪼갠 한글자모 String 두 개 사이의 편집거리를 구하는 알고리즘
"""
class koreanDistance:
    def __init__(self, wordF, wordS):
        self.wordF = wordF
        self.wordS = wordS
        

    def __len__(word):
        return len(word)

    def distanceWords(self):
        lenF = len(self.wordF)
        lenS = len(self.wordS)
        dist = []
        for i in range(0, lenF+1):
            line = []
            for j in range(0, lenS+1):
                line.append(0)
            dist.append(line)

        for i in range(1, lenF+1):
            dist[i][0] = i
        for j in range(1, lenS+1):
            dist[0][j] = j

        for j in range(1, lenS+1):
            for i in range(1, lenF+1):
                if self.wordF[i-1]==self.wordS[j-1]:
                    dist[i][j] = dist[i-1][j-1]
                else:
                    dist[i][j] = min([dist[i-1][j-1], dist[i-1][j], dist[i][j-1]])+1
        return dist[lenF][lenS]



# protoTyping 
"""
미리 만들어 놓은 Word2Vec에 저장되어있는 vocab만 이용하여, 그 vocab들과 편집거리를 구한다.
그 후 그 편집거리와 가장 짧은 단어로 대체해준다. 오타 교정을 해낼 수 있다.
"""
if __name__ == '__main__':
  word2vec = Word2Vec.load('sum_all_news1')
  vocab = list(word2vec.wv.vocab)

  while(True):
     print('start')
     user = input('입력 : ')
     koreanhandler = koreanHandler(user)
     user_input = koreanhandler.splitWord()
     min_value = 10000
     key = ''
     same_key = []
     for voca in vocab:
         koreanhandler_1 = koreanHandler(voca)
         keyword = koreanhandler_1.splitWord()
         koreandistance = koreanDistance(user_input,keyword)
         if koreandistance.distanceWords() < min_value:
             min_value = koreandistance.distanceWords()
             key = voca
             same_key = []
             same_key.append(voca)
         elif koreandistance.distanceWords() == min_value:
             same_key.append(voca)
     print(same_key)

