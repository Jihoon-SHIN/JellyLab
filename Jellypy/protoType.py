# -*- coding: utf-8 -*-
from pykospacing import spacing
from korean_handler import koreanHandler
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g
import re
import hgtk
from konlpy.tag import Twitter
from korean_handler import koreanHandler,koreanDistance
import json
from konlpy import jvm
from konlpy.tag import Kkma, Twitter, Komoran
from konlpy.utils import pprint

twitter = Twitter()

def changeWord(user):
	koreanhandler = koreanHandler(user)
	user_input = koreanhandler.splitWord()
	min_value = 10000
	key = ''
	same_key = []
	chosung_input = ''
	if len(user) == 1:
		chosung_input = hgtk.letter.decompose(user)[0]
	else:
		chosung_d1 = hgtk.letter.decompose(user[0])[0]
		chosung_d2 = hgtk.letter.decompose(user[1])[0]
		chosung_input = chosung_d1 + chosung_d2
	for voca in chosung_json[chosung_input]:
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
	return same_key


def makeAnswer(spacingList, answerList, konlpyList2, _josa):
	_str = "".join(konlpyList2)
	for i, target in enumerate(spacingList):
		if _str+_josa == target:
			changeStr = changeWord(_str)
			if not _str in changeStr:
				if i ==0 :
					answerList = konlpyList2 + spacingList[1:len(spacingList)+1]
				else:
					answerList = spacingList[0:i] + konlpyList2 + spacingList[i+1:len(spacingList)+1]
		elif _josa != "" and _str+_josa == target[0:len(_str+_josa)] and len(_str+_josa) != len(target):
			print(_str+_josa)
			print(target)
			tempList = target.split(_josa)
			if tempList != None:
				if i==0:
					answerList =  [tempList[0]+_josa, tempList[1]] + spacingList[1:len(spacingList)+1]
				else:
					answerList = spacingList[0:i] + [tempList[0]+_josa, tempList[1]] + spacingList[i+1:len(spacingList)+1]

	return answerList


with open('chosung_dict_v2.json') as f:
  chosung_json = json.load(f)

if __name__ == '__main__':
	while(True):
		inputStr = input("input: ")
		if inputStr == 'q':
			print("End")
			break

		print('-------------------------------')
		print('Before Jellypy : %s' % inputStr)
		# KoreanHandler class를 이용해 자음모음을 다 제거
		koreanhandler = koreanHandler(inputStr)
		space_string, punctuation = koreanhandler.deleteJM()

		# pykospacing 을 이용해 1차적으로 띄어쓰기를 함
		spacingStr = spacing(space_string)
		print('First preprocess : %s' % spacingStr)
		spacingList = spacingStr.split(" ")

		# konlpy Twitter를 이용해서 형태소 분석
		konlpyList = twitter.pos(spacingStr)
		changeThing = ""
		print('형태소 분석 결과 : %s' % konlpyList)
		konlpyList2 = list()

		answerList = list()
		answerList = spacingList

		_iter = 0	
		while _iter < len(konlpyList):
			if konlpyList[_iter][1] == 'Noun' or konlpyList[_iter][1]=='Adjective' or konlpyList[_iter][1]=='Exclamation':
				_josa = ""
				if _iter != len(konlpyList)-1 and konlpyList[_iter+1][1] == 'Josa':
					_josa = konlpyList[_iter+1][0]
				konlpyList2.append(konlpyList[_iter][0])
				answerList = makeAnswer(spacingList, answerList, konlpyList2, _josa)
			else:
				konlpyList2 = list()
			_iter = _iter + 1

		for _list in answerList:
			changeThing = changeThing + _list
			changeThing = changeThing + " "

		changeThing = changeThing + punctuation
		print('After Jellypy : %s' % changeThing)
		print('-------------------------------')

