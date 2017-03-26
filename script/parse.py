with open('test.csv') as f:
    content  = f.readlines()

j = 1
for i in content:
    f = open(str(j) + ".pgm", 'w')
    j += 1
    a = i.split(' ')
    f.write('P2\n28 28\n255\n')
    for k in range(0, 28):
        for l in range(0, 28):
            f.write(a[k*28+l])
            f.write(' ')
        f.write('\n')


