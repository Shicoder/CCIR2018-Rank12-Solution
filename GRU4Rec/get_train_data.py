import pickle
infile = open('../zhihu/training_set.txt','r')
outfile = open('../zhihu/training_set_GRU4Rec2.csv','w')
i=0
wi = 0
section_num=1
question = open('../zhihu/question_dict.pkl','rb')
question_dict = pickle.load(question)
answer = open('../zhihu/answer_dict.pkl','rb')
answer_dict = pickle.load(answer)
cand = open('../zhihu/candidate.txt','r')
# outfile = open('../zhihu/result.csv','w')
cand_item_set = set()
for line in cand:
    cand_item_set.add(line.strip('\n').split('\t')[1])
cand.close()
print("can_len:",len(cand_item_set))
import pandas as pd

testfile = open('../zhihu/testing_set_135089.txt','r')
# outtestfile = open('../zhihu/testing_set_135089_clean.csv','w')
i=0
for line in infile:
    i = i + 1

    items = line.strip('\n').split('\t')
    if int(items[1])<=1:
        continue
    items_oneline = items[2].split(',')
    for items in items_oneline:
        items = items.split('|')
        try:
            if items[2]!='0':
                if items[0] in answer_dict:
                    cand_item_set.add(answer_dict[items[0]])
                elif items[0] in question_dict:
                    cand_item_set.add(question_dict[items[0]])
        except (IOError, ZeroDivisionError), e:
            print e
testfile.close()
print("can_test_set:",len(cand_item_set))

def get_dict(s):
    if s in answer_dict:
        return answer_dict[s]
    else:
        return None
for line in infile:
    i = i + 1

    items = line.strip('\n').split('\t')
    # items_set = set(items)
    if int(items[1])>= 2 and i%20==0:
        items_oneline = items[2].split(',')
        # print(items_oneline)
        seq= []
        time_str = []
        for items in items_oneline:
            items = items.split('|')
            try:
                if items[2]!='0':
                    seq.append(items[0])
                    time_str.append(items[2])
            except (IOError, ZeroDivisionError), e:
                print e
        saved_seq=[]
        for s in seq:
            if s in answer_dict:
                saved_seq.append(answer_dict[s])
            elif s in question_dict:
                saved_seq.append(question_dict[s])
        saved_seq_2 = [s for s in saved_seq if s in cand_item_set]
        # if len(saved_seq_2) != len(seq):
        #     continue
        if len(saved_seq_2)>=2  and set(saved_seq).issubset(cand_item_set):
            for s,t in zip(saved_seq,time_str):
                line_list = [str(section_num),s,t]
                outfile.write(",".join(line_list) + "\n")
                wi =wi+1
            section_num=section_num+1
        if i%10000==0:
            print("total:",i)
            print("write",wi)

infile.close()
outfile.close()




