import pickle
import operator


# a = [(1,2),(4,5),(3,6)]
# print(dict(a[0:2]))
dict = open('../zhihu/knn_dict.pkl', 'rb')
item_dict = pickle.load(dict)
print('loading finish!')

for key in item_dict:
    get_items = item_dict[key]
    sorted_items = sorted(get_items.items(), key=operator.itemgetter(1), reverse=True)
    new_item = sorted_items[0:300]
    item_dict[key] = dict(new_item)