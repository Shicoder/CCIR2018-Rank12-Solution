

# infile = open('../zhihu/test_result_3.txt','r')
# outfile = open('../zhihu/result.csv','w')
# j=0
# for line in infile:
#     outfile.write(line)
#     items = line.strip('\n').split(',')
#     if len(items)!=100:
#         print("Error")
# print(j)
# infile.close()
# outfile.close()
# outfile = open('../zhihu/result.csv','w')
# item_set = set()
# i=0
# for line in infile:
#     i=i+1
#     item_set.add(line.strip('\n').split('\t')[1][0:4]+
#                  line.strip('\n').split('\t')[1][-4:])
#     if i==3:
#
#       print(item_set)
#       break
# infile.close()
# print(len(item_set))

#
# infile = open('../zhihu/test_result_2.txt','r')
# for line in infile:
#     items = line.strip('\n').split(',')
#     for item in items:
#         if item  in item_set:
#             print(item)
# import time
# timestamp = 1525853718
# ttt = time.localtime(1525966602)
# now_date = time.strftime("%Y-%m-%d %H:%M:%S", ttt)
# print(now_date)

# import pandas as pd
# data = pd.read_csv('/Users/shixiangfu/Documents/face1/data/zhihu/yoochoose-clicks.dat',sep=',', header=None,nrows=10)
# # data = pd.DataFrame([1,2,3])
# # print(data.columns.values)
# print(data.head())
a = set(['a','b'])
c = set(['a','c'])
print(a.issubset(c))
