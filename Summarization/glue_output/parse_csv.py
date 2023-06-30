import csv
import pprint

pp = pprint.PrettyPrinter(indent=4)
with open('/home/zwhe/fourier_transformer/custom_code/output/glue_fp150k.csv', 'r') as f:
    content = f.readlines()

# print(content)
result = dict()
for i in content:
    i = i.split(',')
    if i[0] not in result:
        result[i[0]] = i[1][:-1]
    if i[0] in result:
        if ' ' in result[i[0]]:
            acc = result[i[0]].split(' ')[1][:-1]
        else:
            acc = result[i[0]].split('accuracy')[1][:-1]
        if acc < i[1]: 
            result[i[0]] = i[1][:-1]

pp.pprint(result)