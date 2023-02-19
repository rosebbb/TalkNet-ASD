x = [1, 2, 3,4,5,6,7,8]
l = len(x)
i = 0
while True:
    if i >= len(x):
        break
    num = x[i]
    print(i, num)
    if num == 3:
        x.remove(num)
    elif num == 4:
        x.remove(num)
    else:
        i+=1

print(x)