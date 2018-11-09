import sys

infile = open(sys.argv[1], 'r')
outfile = open(sys.argv[2], 'w')

for line in infile.readlines():
    items = line.strip('\n').split('\t')
    outfile.write("\t".join(items[0:-3]) + "\n")
    
infile.close()
outfile.close()
