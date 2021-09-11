import pandas as pd
import numpy as np
import json
def read_csv(file_path):
    df = pd.read_csv(file_path)
    sentence1_list = []
    sentence2_list = []
    len_sentence1_list = []
    len_sentence2_list = []
    sentence2_list = []
    label_list = []
    temp = []
    for line_idx in range(len(df)):
        sentence1 = str(df.loc[line_idx, "sentence1"]).strip()
        sentence2 = str(df.loc[line_idx, "sentence2"]).strip()
        label = str(df.loc[line_idx, "label"]).strip()
        sentence1_list.append(sentence1)
        sentence2_list.append(sentence2)
        len_sentence1_list.append(len(sentence1))
        if len(sentence1) > 100:
            temp.append(sentence1)
        len_sentence2_list.append(len(sentence2))
        a = int(label)
        if a != 0 and a != 1:
            print(a)
        label_list.append(int(label))
    label_list = np.array(label_list)
    len_sentence1_list = np.array(len_sentence1_list)
    len_sentence2_list = np.array(len_sentence2_list)

    print("bupt")

def read_json(file_path):
    with open(file_path, 'r') as r:
        a = json.load(r)
    return a
def main():
    #read_csv('./data/dsdata2N/dsdata_train.csv')
    a = read_json('/home1/CM2021/zwh/QQdata/1-1/train_data.json')
    print("buot")
if __name__ == '__main__':
    main()