yourdict = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}

for k in list(yourdict.keys()):
    if yourdict [k] == 4:
        yourdict.pop(k)
        # del yourdict[k]

print(yourdict)