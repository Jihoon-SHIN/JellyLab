from korean_spacing import koreanSpacing


while(True):
	inputStr = input("input: ")
	if inputStr == 'q':
		print("End")
		break

	print('-------------------------------')
	print('Before Jellypy : %s' % inputStr)

	afterStr = koreanSpacing(inputStr, permitChange=False).makeOutput()

	print('After Jellypy : %s' % afterStr)
	print('-------------------------------')



