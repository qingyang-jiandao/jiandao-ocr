with open('xxx.list', 'r', encoding='utf-8') as f:
    lines = [line.strip('\n') for line in f.readlines()]

word_dict = {}
for i, line in enumerate(lines):
    splits = line.split('|')
    word = splits[1]
    #print(word)
    word_dict[word] = word

with open('dict_6862_extend.list', 'w', encoding='utf-8') as out_list:
    for index, word in enumerate(word_dict):
        label_info = '%s\n' % (word)
        out_list.write(label_info)

