import json


def main():
    with open('/home1/CM2021/zwh/qaData-wzc/qaTrainData.json', 'r') as r:
        a = json.load(r)
    with open('/home1/CM2021/zwh/QAdata/1-1/train_data.json', 'r') as r:
        b = json.load(r)
    print("bupt")

if __name__ == '__main__':
    main()