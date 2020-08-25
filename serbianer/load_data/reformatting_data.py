def load_file(data_selected):
    if data_selected == "reldi":
        data_file = open("../../datasets/original/set.sr.conll", "r", encoding="utf-8")
    elif data_selected == "hr500k":
        data_file = open("../../datasets/original/hr500k.conll", "r", encoding="utf-8")
    elif data_selected == "test_ner":
        data_file = open("../../datasets/original/test_ner.conllu", "r", encoding="utf-8")
    else:
        print("Unknown file to work with")
        exit(-1)
    return data_file


def process(data_selected, name='', filepath='../datasets/'):
    """

    :param data_selected: 'reldi, hr500k, test_ner"
    :param filepath: where it will be saved
    :param name: name which it will be saved
    :return: processed in csv file
    """
    data_file = load_file(data_selected)
    data_list = []
    write_next = True
    for line in data_file:
        if line.strip().__len__() < 1:
            data_list.append(("", "", "", ""))
            continue
        elif line[0] == '#':
            continue
        try:
            split = line.strip().replace('\t', " ").split(" ")
            wordsplit = split[1]
            lemmasplit = split[2]

            if data_selected == "reldi":
                possplit = split[4]  # reldi
                tagsplit = split[10]  # reldi
            elif data_selected == "test_ner":
                possplit = split[3]  # test
                tagsplit = split[9]  # test
            elif data_selected == "hr500k":  # :
                possplit = split[4]  # hr500k
                tagsplit = split[10]  # hr500k

            t = (wordsplit.rsplit(), lemmasplit.rsplit(), possplit.rsplit(), tagsplit.rsplit())
            data_list.append(t)
        except:
            print(line.strip())  # if this comes to show, it means there was a error

    i = 1
    new_file_name = ''
    if name != '':
        new_file_name = filepath + name
    else:
        if data_selected == "reldi":
            new_file_name = filepath + "reldi.csv"
        elif data_selected == "hr500k":
            new_file_name = filepath + "hr500k.csv"
        elif data_selected == "test_ner":
            new_file_name = filepath + "test.csv"

    try:
        new_file = open(new_file_name, "w+", encoding="utf-8")
        new_file.write("Sentence #\tWord\tPos\tTag\n")
        for a, b, c, d in data_list:
            if a == "":
                write_next = True
                i = i + 1
            else:
                if write_next:
                    write_next = False
                    new_file.write("Sentence: {3}\t{0}\t{1}\t{2}\n".format(str(a[0]), str(c[0]), str(d[0]), i))
                else:
                    new_file.write("\t{0}\t{1}\t{2}\n".format(str(a[0]), str(c[0]), str(d[0])))
        new_file.close()
    except:
        print("Error while writing file")


if __name__ == '__main__':
    data_selected_to_process = "reldi"  # available reldi, hr500k and test_ner
    process(data_selected_to_process)
    print("Done")
