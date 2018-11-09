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
# for items in my_dict:
#     for key in my_dict[items]:
#         if my_dict[items][key]<=2:
#             del my_dict[items][key]

item_dict = my_dict.copy()
import  gc
gc.collect()


import pickle
import operator
# dict = open('../zhihu/knn_dict2.pkl','rb')
# item_dict = pickle.load(dict)
answer_dict = open('../zhihu/answer_dict.pkl','rb')
answer = pickle.load(answer_dict)
infile = open('../zhihu/testing_set_135089_last_id.txt','r')
outfile = open('../zhihu/test_result.txt','w')
print('loading finish!')


cand = open('../zhihu/candidate.txt','r')
# outfile = open('../zhihu/result.csv','w')
item_set = set()
for line in cand:
    item_set.add(line.strip('\n').split('\t')[1])


cand.close()


for line in infile:
    items = line.strip('\n')
    if items in item_dict:
        get_items = item_dict[items]
    else:
        tmp = ['-1']*100
        outfile.write(",".join(tmp) + "\n")
        continue
    sorted_items = sorted(get_items.items(), key=operator.itemgetter(1),reverse=True)

    print("sort_finish",sorted_items[0:5])
    can_list=[]
    count = 0
    for id_and_num in sorted_items:

        if count>300:
            print(id_and_num[0] + '->' + str(id_and_num[1]))
            break
        if id_and_num[0] in answer and  answer[id_and_num[0]] in item_set:
            # print("test")
            can_list.append(id_and_num[0])
            count = count + 1
    outfile.write(",".join(can_list) + "\n")
    # print(items=='A193942753')
infile.close()
outfile.close()