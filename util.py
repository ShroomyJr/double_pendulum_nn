import csv
from os import walk

def unpack_csv(path, rotate=False):
    with open(path, newline='') as csvfile:
        position_reader = csv.reader(csvfile)
        x_1, x_2, x_3 = [], [], []
        y_1, y_2, y_3 = [], [], []
        for row in position_reader:
            if rotate:
                y_1.append(-int(row[0]))
                x_1.append(int(row[1]))
                y_2.append(-int(row[2]))
                x_2.append(int(row[3]))
                y_3.append(-int(row[4]))
                x_3.append(int(row[5]))
            else:
                x_1.append(int(row[0]))
                y_1.append(int(row[1]))
                x_2.append(int(row[2]))
                y_2.append(int(row[3]))
                x_3.append(int(row[4]))
                y_3.append(int(row[5]))
        
        return x_1, y_1, x_2, y_2, x_3, y_3

def write_csv(data, path):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data[0])):
            row = [col[i] for col in data]
            writer.writerow(row)

def retrieve_csvs(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break
    return f