import sys

with open(sys.argv[1]) as f:
    content = f.readlines()

with open(sys.argv[2]) as f:
    content2 = f.readlines()

correct = 0.0
total = 0.0

for i in range(0,len(content)):
    if content2[i] == content[i]:
        correct += 1.0
    total += 1.0

print correct
print "Accuracy = ", correct / total
