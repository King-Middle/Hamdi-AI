files = ['secure_list_18_FXIFY.txt']

for i in files:

    with open(i, 'r') as f:
        text = f.read()

        if text[-2:] == ", ":
            text = text[:-2]

    with open(f'{i[:-4]}_for_use{i[-4:]}', 'w') as f:

        f.write(f"'{text.replace(", ", "', '")}'")


print('Done')
