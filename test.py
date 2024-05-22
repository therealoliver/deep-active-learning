for i in range(10):
    with open('test.txt', 'a') as f:
        f.write(str(i) + '\t')
with open('test.txt', 'a') as f:
    f.write('\n')