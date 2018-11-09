import pandas as pd
# ##########################get last id from test set##########
# infile = open('../zhihu/testing_set_135089.txt','r')
# outfile = open('../zhihu/testing_set_135089_last_id.txt','w')
# j=0
# k=0
# for line in infile:
#     j=j+1
#     # print(j)
#     items = line.strip('\n').split('\t')
#     # print(items[1])
#     items_oneline = items[2].split(',')
#     if int(items[1])==0:
#         outfile.write(last_click_id + "\n")
#         k=k+1
#         print('miss:',k)
#         continue
#
#     last_id ='-1'
#     last_click_id ='-1'
#     for i in range(0,len(items_oneline)):
#         if i==0:
#             last_id=items_oneline[i].split('|')[0]
#         if items_oneline[i].split('|')[2]!=str(0):
#             last_click_id=items_oneline[i].split('|')[0]
#             break
#     # print(last_click_id)
#     if last_click_id !='-1':
#         outfile.write(last_click_id + "\n")
#     else:
#         outfile.write(last_id + "\n")
# print(j)
# infile.close()
# outfile.close()
##############################get top300 result #####################
# import pickle
# import operator
# dict = open('../zhihu/knn_dict2.pkl','rb')
# item_dict = pickle.load(dict)
# answer_dict = open('../zhihu/answer_dict.pkl','rb')
# answer = pickle.load(answer_dict)
# infile = open('../zhihu/testing_set_135089_last_id.txt','r')
# outfile = open('../zhihu/test_result.txt','w')
# print('loading finish!')
#
#
# cand = open('../zhihu/candidate.txt','r')
# # outfile = open('../zhihu/result.csv','w')
# item_set = set()
# for line in cand:
#     item_set.add(line.strip('\n').split('\t')[1])
#
#
# cand.close()
#
#
# for line in infile:
#     items = line.strip('\n')
#     if items in item_dict:
#         get_items = item_dict[items]
#     else:
#         tmp = ['-1']*100
#         outfile.write(",".join(tmp) + "\n")
#         continue
#     sorted_items = sorted(get_items.items(), key=operator.itemgetter(1),reverse=True)
#
#     print("sort_finish",sorted_items[0:5])
#     can_list=[]
#     count = 0
#     for id_and_num in sorted_items:
#
#         if count>300:
#             print(id_and_num[0] + '->' + str(id_and_num[1]))
#             break
#         if id_and_num[0] in answer and  answer[id_and_num[0]] in item_set:
#             # print("test")
#             can_list.append(id_and_num[0])
#             count = count + 1
#     outfile.write(",".join(can_list) + "\n")
#     # print(items=='A193942753')
# infile.close()
# outfile.close()
###########################use true id name and cut id length##########

import pickle
question = open('../zhihu/question_dict.pkl','rb')
question_dict = pickle.load(question)
answer = open('../zhihu/answer_dict.pkl','rb')
answer_dict = pickle.load(answer)

infile = open('../zhihu/test_result.txt','r')
testfile = open('../zhihu/testing_set_135089.txt','r')
outfile = open('../zhihu/test_result_0712_16.txt','w')
l=0
miss= 0
for line,testline in zip(infile,testfile):
    l=l+1
    # print(l)
    items = line.strip('\n').split(',')
    test_items = testline.strip('\n').split('\t')
    test_set = set()
    if int(test_items[1])>=1:
        test_items_sp = test_items[2].split(',')
        test_list = [x.split('|')[2] for x in test_items_sp]
        test_set = set(test_list)
    result = []
    num=0
    # print("items", items)
    if len(items) == 0 or items[0] == '':
        # print("items",len(items))
        tmp = ['-1'] * 100
        outfile.write(",".join(tmp) + "\n")
        continue
    for item in items:

        # print(num)
        # print(item[0])

        if item[0]=='A':
            if item in answer_dict and (item not in test_set):
                num = num + 1
                result.append(answer_dict[item][:4]+answer_dict[item][-4:])
            else:
                continue
        elif item[0] == 'Q':
            if item in question_dict:
                continue
                # num = num + 1
                # result.append(question_dict[item][:4] + question_dict[item][-4:])
            else:
                continue
        else:
            continue
        if num >= 100:
            break
    if num < 100:
        miss = miss+1
        tmp = ['-1'] * (100 - num)
        print("miss:",miss)
        result.extend(tmp)
    print(len(result))
    outfile.write(",".join(result) + "\n")

infile.close()
outfile.close()

######################get_ sub
infile = open('../zhihu/test_result_0712_16.txt','r')
outfile = open('../zhihu/result.csv','w')
j=0
for line in infile:
    j=j+1
    outfile.write(line)
    items = line.strip('\n').split(',')
    if len(items)!=100:
        print("Error")
print("135089===>",j)

infile.close()
outfile.close()

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
