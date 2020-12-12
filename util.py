import csv
from os import walk

def unpack_csv(path, rotate=False):
    with open(path, newline='') as csvfile:
        position_reader = csv.reader(csvfile)
        x_1, x_2, x_3 = [], [], []
        y_1, y_2, y_3 = [], [], []
        for row in position_reader:
            if rotate:
                y_1.append(-float(row[0]))
                x_1.append(float(row[1]))
                y_2.append(-float(row[2]))
                x_2.append(float(row[3]))
                y_3.append(-float(row[4]))
                x_3.append(float(row[5]))
            else:
                x_1.append(float(row[0]))
                y_1.append(float(row[1]))
                x_2.append(float(row[2]))
                y_2.append(float(row[3]))
                x_3.append(float(row[4]))
                y_3.append(float(row[5]))
        
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

def scale(value,min_r, max_r, min_t, max_t):
    return (value-min_r)/(max_r-min_r)*(max_t-min_t)+min_t

def scale_data(data):
    r_min = min([min(data[0]), min(data[1]), \
             min(data[2]), min(data[3]), \
             min(data[4]), min(data[5])])
    r_max = max([max(data[0]), max(data[1]), \
                max(data[2]), max(data[3]), \
                max(data[4]), max(data[5])])
    t_min = -2
    t_max = 2

    scaled_data = [[scale(value, r_min, r_max, t_min, t_max) for value in col] for col in data]
    avg_x = sum(scaled_data[0])/len(scaled_data[0])
    avg_y = sum(scaled_data[1])/len(scaled_data[1])

    x1 = [value - avg_x for value in scaled_data[0]]
    y1 = [value + avg_x for value in scaled_data[1]]
    x2 = [value - avg_x for value in scaled_data[2]]
    y2 = [value + avg_x for value in scaled_data[3]]
    x3 = [value - avg_x for value in scaled_data[4]]
    y3 = [value + avg_x for value in scaled_data[5]]
    scaled_data=[x1, y1, x2, y2, x3, y3]
    return scaled_data

def read_csv_for_network(path):
     with open(path, newline='') as csvfile:
        position_reader = csv.reader(csvfile)
        return [[float(value) for value in row] for row in position_reader]