# -*- coding: utf-8 -*-
from pykospacing import spacing
from korean_handler import koreanHandler
import hgtk
from konlpy.tag import Twitter
from korean_handler import koreanHandler,koreanDistance
import json

"""
오타 교정을 위해 미리 만들어놓은 json파일을 로드
"""
with open('chosung_dict_v2.json') as f:
	chosung_json = json.load(f)

"""
json파일에서 word(input)과 편집거리가 가장 짧은 단어들의 list들을 return
"""
def changeWord(word):
	koreanhandler = koreanHandler(word)
	_splitWord = koreanhandler.splitWord()
	min_value = 10000
	key = ''
	same_key = []
	chosung_input = ''
	if len(word) == 1:
		chosung_input = hgtk.letter.decompose(word)[0]
	else:
		chosung_d1 = hgtk.letter.decompose(word[0])[0]
		chosung_d2 = hgtk.letter.decompose(word[1])[0]
		chosung_input = chosung_d1 + chosung_d2
	for voca in chosung_json[chosung_input]:
		koreanhandler_1 = koreanHandler(voca)
		keyword = koreanhandler_1.splitWord()
		koreandistance = koreanDistance(_splitWord,keyword)
		if koreandistance.distanceWords() < min_value:
			min_value = koreandistance.distanceWords()
			key = voca
			same_key = []
			same_key.append(voca)
		elif koreandistance.distanceWords() == min_value:
			same_key.append(voca)
	return same_key

"""
여러 규칙들을 통해서 spacing(package)에서 거르지 못한 띄어쓰기들을 konlpy(twitter)를 이용하여 추가적으로 띄어쓰기 해준다.
"""
def makeAnswer(spacingList, answerList, konlpyList2, _josa, permitChange):
	_str = "".join(konlpyList2)
	for i, target in enumerate(spacingList):
		if _str+_josa == target:
			changeStr = changeWord(_str)
			if not _str in changeStr:
				# If permitChange is True, change the word to right word in dictionary
				# If permitChange is False, then do not change the word.
				# Default value is False, if you want, change the value
				if permitChange and len(konlpyList2) == 1:
					konlpyList2[0] = changeStr[0]
				konlpyList2[-1] = konlpyList2[-1] + _josa
				if i ==0 :
					answerList = konlpyList2+ spacingList[1:len(spacingList)+1]
				else:
					answerList = spacingList[0:i] + konlpyList2  + spacingList[i+1:len(spacingList)+1]
		elif _josa != "" and _str+_josa == target[0:len(_str+_josa)] and len(_str+_josa) != len(target):
			tempList = target.split(_josa)
			if tempList != None:
				if i==0:
					answerList =  [tempList[0]+_josa, tempList[1]] + spacingList[1:len(spacingList)+1]
				else:
					answerList = spacingList[0:i] + [tempList[0]+_josa, tempList[1]] + spacingList[i+1:len(spacingList)+1]

	return answerList


"""
spacing을 쉽게 하기 위한 class 모듈
"""
class koreanSpacing:
	# Initializing variables to need
	def __init__(self, inputStr, permitChange=False):
		self.twitter = Twitter()
		self.inputStr = inputStr

		self.answerList = list()
		self.konlpyList = list()
		self.konlpyList2 = list()
		self.spacingList = list()

		self.spacingStr = ""
		self.changeThing = ""
		self._josa = ""

		self.koreanhandler = koreanHandler(self.inputStr)
		self.space_string, self.punctuation = self.koreanhandler.deleteJM()
		self.spacingStr = spacing(self.space_string)
		self.spacingList = self.spacingStr.split(" ")
		self.konlpyList = self.twitter.pos(self.spacingStr)
		self.answerList = self.spacingList

		self.permitChange = permitChange
	# A function to make output
	def makeOutput(self):
		_iter = 0	
		print("hello")
		while _iter < len(self.konlpyList):
			if self.konlpyList[_iter][1] == 'Noun' or self.konlpyList[_iter][1]=='Adjective' or self.konlpyList[_iter][1]=='Exclamation':
				if _iter != len(self.konlpyList)-1 and self.konlpyList[_iter+1][1] == 'Josa':
					self._josa = self.konlpyList[_iter+1][0]
				self.konlpyList2.append(self.konlpyList[_iter][0])
				# self.konlpyList2.append(self._josa)
				self.answerList = makeAnswer(self.spacingList, self.answerList, self.konlpyList2, self._josa, self.permitChange)
			else:
				self.konlpyList2 = list()
			_iter = _iter + 1

		for _list in self.answerList:
			self.changeThing = self.changeThing + _list
			self.changeThing = self.changeThing + " "

		self.changeThing = self.changeThing + self.punctuation
		return self.changeThing