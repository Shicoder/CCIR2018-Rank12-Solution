import pandas as pd
import pickle
import gc
infile = open('../zhihu/training_set_clean.txt','r')
# outfile = open('../zhihu/training_set_clean.csv','w')
j=0
my_dict = {}

# cand = open('../zhihu/candidate.txt','r')
# # outfile = open('../zhihu/result.csv','w')
# item_set = set()
# for line in cand:
#     item_set.add(line.strip('\n').split('\t')[1])
# cand.close()

for line in infile:
    j = j + 1
    # print(i)

    # print(line)
    items = line.strip('\n').split('\t')
    # print(items)
    for i in range(0,len(items)-2):
        if items[i+1] in my_dict:
            if items[i+2] in my_dict[items[i+1]]:
                my_dict[items[i+1]][items[i+2]] = my_dict[items[i+1]][items[i+2]]+1
            else:
                my_dict[items[i+1]][items[i + 2]]=1
            if items[i] in my_dict[items[i+1]]:
                my_dict[items[i+1]][items[i]] = my_dict[items[i+1]][items[i]]+1
            else:
                my_dict[items[i+1]][items[i]]=1
        else:
            sub_dict1 = {}
            sub_dict1[items[i+2]]=1
            my_dict[items[i+1]]=sub_dict1

            sub_dict2 = {}
            sub_dict2[items[i]]=1
            my_dict[items[i+1]]=sub_dict2
    # print(my_dict)
    if j%10000==0:
        print(j)
    # if i==100:
    #     df = pd.DataFrame(my_dict).T.fillna(0)
    #     break
print(j)
infile.close()
for items in my_dict:
    for key in my_dict[items]:
        if my_dict[items][key]<=2:
            del my_dict[items][key]
try:
    output = open('../zhihu/knn_dict2.pkl','wb')
    pickle.dump(my_dict,output)
except IOError as err:
    print('File error: ' + str(err))
except pickle.PickleError as perr:
    print('Pickling error: ' + str(perr))

# input = open('../zhihu/knn_dict2.pkl','rb')
# my_dict = pickle.load(input)
# # print(len(my_dict))
# print(my_dict[''])
#################################################
# infile = open('../zhihu/answer_id.dict','r')
# answer_dict={}
# j=0
# for line in infile:
#     j = j + 1
#     # print(i)
#
#     # print(line)
#     items = line.strip('\n').split('\t')
#     if 'A'+items[0] in answer_dict:
#         continue
#     else:
#         answer_dict['A'+items[0]]=items[1]
#     if j%10000==0:
#         print(j)
# # print(answer_dict)
# infile.close()
# try:
#     output = open('../zhihu/answer_dict.pkl','wb')
#     pickle.dump(answer_dict,output)
# except IOError as err:
#     print('File error: ' + str(err))
# except pickle.PickleError as perr:
#     print('Pickling error: ' + str(perr))
# output.close()
# print("answer finsh")
# gc.collect()
# ######################
# infile = open('../zhihu/question_id.dict','r')
# question_dict={}
# j=0
# for line in infile:
#     j = j + 1
#     # print(i)
#
#     # print(line)
#     items = line.strip('\n').split('\t')
#     if 'Q'+items[0] in question_dict:
#         continue
#     else:
#         question_dict['Q'+items[0]]=items[1]
#     if j%10000==0:
#         print(j)
# # print(answer_dict)
# infile.close()
# try:
#     output = open('../zhihu/question_dict.pkl','wb')
#     pickle.dump(question_dict,output)
# except IOError as err:
#     print('File error: ' + str(err))
# except pickle.PickleError as perr:
#     print('Pickling error: ' + str(perr))
# output.close()
# print("question finsh")