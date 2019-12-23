# import re
#
# str='rnn[clus_num=6,pp=55]'
# in_string=re.findall(r'[[](.*?)[]]', str)
# str2=in_string[0].split(',')
# print(str2)
#
# for line in str2:
#     loc= line.index('=')
#     name=line[0:loc]
#     num=line[loc+1:]
#     print(loc,name,num)



import re

i='rnn'

algo = re.match(r'(.*?)\[(.*?)](.*?)',i)
if(algo==None):
    algo=i
    str=None
else:
    algo=algo.group(1)
    str= re.match(r'(.*?)\[(.*?)](.*?)',i).group(2)
print(algo)
print(str)
str=str.split(',')
# # print(str)
# # for line in str:
# #     loc= line.index('=')
# #     name=line[0:loc]
# #     num=int(line[loc+1:])
# #     print(num==6)
# #     print(loc,name,num)


