import pandas as pd

infile = open('../zhihu/testing_set_135089.txt','r')
outfile = open('../zhihu/testing_set_135089_clean.csv','w')
i=0
for line in infile:
    i = i + 1
    print(i)
    items = line.strip('\n').split('\t')
    items_oneline = items[2].split(',')
    # print(items_oneline)
    seq= []
    for items in items_oneline:
        items = items.split('|')
        try:
            if items[2]!='0':
                seq.append(items[0])
        except (IOError, ZeroDivisionError), e:
            print e
        outfile.write(",".join(seq) + "\n")
    if i%10000==0:
        print(i)
        # if len(seq)>=4:
        #     for i in range(0,len(seq)-3):
        #         if seq[i] in my_dict:
        #             for j in range(1,4):
        #                 if seq[i+j] in my_dict[seq[i]]:
        #                     my_dict[seq[i]][seq[i+j]] += 1
        #                 else:
        #                     my_dict[seq[i]][seq[i + j]] = 1
        #         else:
        #             for j in range(1,4):
        #                     my_dict[seq[i]][seq[i + j]] = 1
        # print(my_dict)
        # break
#


infile.close()
outfile.close()



# can_dict = {}
#
# can = open('../zhihu/candidate.txt','r')
# i=1
# for line in can:
#     items = line.strip('\n').split('\t')
#     items = items[1]
#     print(items)
#     can_dict[items]=str(i)
#     i=i+1
#     if i==5:
#         break
