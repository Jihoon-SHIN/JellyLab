# -*- coding: utf-8 -*-
from time import sleep

# f = open('2010_2018.txt','r',encoding='utf-8')
# f_w = open('2010_2018_1.txt','a',encoding='utf-8')
# f1 = open('2010_2018_1.txt', 'r', encoding='utf-8')
# f_w1 = open('2010_2018_2.txt','a',encoding='utf-8')
# f2 = open('2010_2018_2.txt', 'r', encoding='utf-8')
# f_w2 = open('2010_2018_3.txt','a',encoding='utf-8')

terminate = True
enter_count = 0
max_enter_count = 0

count = 0
count1 = 0
def delete_custom(f, f_w):
  lines = f.readlines()
  preLine = ""
  for line in lines:
    if line == preLine:
      continue
    if len(line)<6:
      continue
    f_w.write(line)
    preLine = line

  f.close()
  f_w.close()

def delete_custom2(f, f_w):
  lines = f.readlines()
  deleteString = '사진'
  for line in lines:
    if deleteString in line:
      continue
    else:
      f_w.write(line)
  f.close()
  f_w.close()

def infoFunc(f):
  lines = f.readlines()
  count = 0
  for line in lines:
      count = count+1
      print(line)
      sleep(3)
      if count ==10:
        break
  print(len(lines))
  f.close()



def delete_thing(f, f_w):
  enter_count = 0
  max_enter_count = 0
  terminate = True
  count = 0
  lines = f.readlines()
  for line in lines:
     count += 1
     if line == '':
         terminate = False
     if line == '\n':
         enter_count += 1
         if enter_count > max_enter_count:
             max_enter_count = enter_count
     else:
         erase_it = [] # 지울단어 넣기
         line = line.replace('           ','')
         line = line.replace('       ','')
         line = line.replace('   ','')
         line = line.replace('  ','')
  #         line = line.replace(' ','')
         line = line.replace('<','')
         line = line.replace('>','')
         line = line.replace('[','')
         line = line.replace(']','')
         line = line.replace('■','')
         line = line.replace('연합뉴스','')
         line = line.replace('경향신문','')
         line = line.replace('경향이 찍은 오늘','')
         for word in erase_it:
             line = line.replace(word,'')
         word_list = line.split(' ')
         word_list_v2 = []
         for i in range(0,len(word_list)):
             if '@' in word_list[i]:
                 word_list_v2 = word_list[:i]+word_list[i+1:]
                 line = ' '.join(word_list_v2)
         word_list_v3 = []
         for j in range(0,len(word_list_v2)):
             if '기자' in word_list_v2[j]:
                 word_list_v3 = word_list_v2[:j-1] + word_list_v2[j+1:]
                 line = ' '.join(word_list_v3)
         if line != '\n':
             f_w.write(line)
         enter_count = 0
  f.close()
  f_w.close()

# delete_thing->delete_custom->delete_custom2

# delete_thing(f, f_w)
# delete_custom(f1,f_w1)
# # infoFunc(f_namu)
# delete_custom2(f2,f_w2)

f = open('2010_2018_3_v2.txt','r')
f_1 = open('sum_news.txt','r')
file = open('sum_all_news.txt','a')

lines = f.readlines()
lines_ = f_1.readlines()


for line in lines:
  file.write(line)

for line in lines_:
  file.write(line)

f.close()
f_1.close()
file.close()

