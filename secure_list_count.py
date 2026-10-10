files = ['secure_list_18_FXIFY.txt']


for i in files:
    with open(i, 'r') as f:
        count = f.read().count(',')
    file_name = i[12:-4]
    print(f"{file_name} --> {count}")