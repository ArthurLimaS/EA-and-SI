def dblp_loader():
    dblp = open("dblp.txt")
    dblp.readline() # pular primeira linha

    for line in dblp:
        if line != '':
            prefix = line[0:2]
        else:
            prefix = ''

        if prefix == '#*': # paperTitle
            a = 0
        elif prefix == '#@': # Authors
            a = 0
        elif prefix == '#t': # Year
            a = 0
        elif prefix == '#c': # publication venue
            a = 0
        elif prefix == '#index': # index id of this paper
            a = 0
        elif prefix == '#%': # the id of references of this paper (there are multiple lines, with each indicating a reference)
            a = 0
        elif prefix == '#!': # abstract
            a = 0

    """
    count = 0
    for x in file:
        print(x[0:2], end="")
        count += 1

        if count == 10:
            break
    """