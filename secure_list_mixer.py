secure_lists_for_use = ['secure_list_final_for_use.txt', 'secure_list_final_final_for_use.txt']

mixed_file_name = 'secure_list_final_mixed.txt'

dic = {}

mixed_list = []


for i in secure_lists_for_use:
    with open(i, 'r') as f:
        dic[secure_lists_for_use.index(i)] = f.read().split(', ')


for i in dic.values():
    for symbol in i:
        if symbol not in mixed_list:
            mixed_list.append(symbol)


with open(mixed_file_name, 'w') as f:
    f.write('')

for symbol in mixed_list:
    with open(mixed_file_name, 'a') as f:
        f.write(f"{symbol}, ")

with open(mixed_file_name, 'r') as f:
    text = f.read().replace("'", "")

with open(mixed_file_name, 'w') as f:
    f.write(text)


print("Mixing Done")