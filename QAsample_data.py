import pandas as pd
import numpy as np
import math
import json

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def create_positive(train_seed, df):
    train_list = []
    for item in train_seed:
        temp_dict = {}
        temp_seed = np.arange(9)
        np.random.shuffle(temp_seed)
        sentence1 = str(df.iloc[item, temp_seed[0]]).strip()
        sentence2 = str(df.iloc[item, 9]).strip()
        label = str(1).strip()
        temp_dict['sent_1'] = sentence1
        temp_dict['sent_2'] = sentence2
        temp_dict['label'] = label
        train_list.append(temp_dict)
    return train_list

def create_negative(train_seed_neg_1, train_seed_neg_2, df):
    train_list = []
    for item1, item2 in zip(train_seed_neg_1, train_seed_neg_2):
        temp_dict = {}
        temp_seed_1 = np.arange(9)
        np.random.shuffle(temp_seed_1)
        sentence1 = str(df.iloc[item1, temp_seed_1[0]]).strip()
        sentence2 = str(df.iloc[item2, 9]).strip()
        label = str(0).strip()
        temp_dict['sent_1'] = sentence1
        temp_dict['sent_2'] = sentence2
        temp_dict['label'] = label
        train_list.append(temp_dict)
    return train_list

def main():
    np.random.seed(42)
    df = read_csv('/home1/CM2021/zwh/QQdata/QA_smple.csv')
    total_seed = np.arange(len(df))
    total_seed_neg_1 = np.arange(len(df))
    total_seed_neg_2 = np.arange(len(df))

    np.random.shuffle(total_seed)
    np.random.shuffle(total_seed_neg_1)
    np.random.shuffle(total_seed_neg_2)

    train_seed = total_seed[0: math.floor(0.8 * len(df))]
    dev_seed = total_seed[math.floor(0.8 * len(df)): math.floor(0.9 * len(df))]
    test_seed = total_seed[math.floor(0.9 * len(df)):]
    train_seed_neg_1 = total_seed_neg_1[0: math.floor(0.8 * len(df))]
    dev_seed_neg_1 = total_seed_neg_1[math.floor(0.8 * len(df)): math.floor(0.9 * len(df))]
    test_seed_neg_1 = total_seed_neg_1[math.floor(0.9 * len(df)):]
    train_seed_neg_2 = total_seed_neg_2[0: math.floor(0.8 * len(df))]
    dev_seed_neg_2 = total_seed_neg_2[math.floor(0.8 * len(df)): math.floor(0.9 * len(df))]
    test_seed_neg_2 = total_seed_neg_2[math.floor(0.9 * len(df)):]

    train_list = create_positive(train_seed, df)
    dev_list = create_positive(dev_seed, df)
    test_list = create_positive(test_seed, df)

    train_list_neg = create_negative(train_seed_neg_1, train_seed_neg_2, df)
    dev_list_neg = create_negative(dev_seed_neg_1, dev_seed_neg_2, df)
    test_list_neg = create_negative(test_seed_neg_1, test_seed_neg_2, df)

    train_list.extend(train_list_neg)
    dev_list.extend(dev_list_neg)
    test_list.extend(test_list_neg)
    seed1 = np.arange(len(train_list))
    np.random.shuffle(seed1)
    train_list_save = []
    for item in seed1:
        train_list_save.append(train_list[item])
    seed2 = np.arange(len(dev_list))
    np.random.shuffle(seed2)
    dev_list_save = []
    for item in seed2:
        dev_list_save.append(dev_list[item])
    seed3 = np.arange(len(test_list))
    np.random.shuffle(seed3)
    test_list_save = []
    for item in seed3:
        test_list_save.append(test_list[item])

    with open('/home1/CM2021/zwh/QAdata/1-1/train_data.json', 'w') as w:
        json.dump(train_list_save, w)
    with open('/home1/CM2021/zwh/QAdata/1-1/dev_data.json', 'w') as w:
        json.dump(dev_list_save, w)
    with open('/home1/CM2021/zwh/QAdata/1-1/test_data.json', 'w') as w:
        json.dump(test_list_save, w)


    print("bupt")

if __name__ == '__main__':
    main()