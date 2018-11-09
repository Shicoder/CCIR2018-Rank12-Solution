

# infile = open('../zhihu/answer_infos.txt','r')
# # outfile = open('../zhihu/training_set_clean.txt','w')
# # i=0
# for line in infile:
#     print(line)
#     break
import pandas as pd

an = pd.read_table('../zhihu/answer_infos.txt',nrows=10,header=None)
print(an)