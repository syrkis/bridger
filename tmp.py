import matplotlib.pyplot as plt

files = ['no_dropout.csv','dropout_01.csv','dropout_025.csv','dropout_04.csv']
al = []
for file_name in files:
    with open(f"data/{file_name}","r") as f:
        lines = f.readlines()
        vals = list(map(float,lines[1].split(',')))
        al.append(vals)

labels = ['No Dropout','0.1 dropout','0.25 dropout','0.4 dropout']
x = [i for i in range(len(al[0]))]
for data,lab in zip(al,labels):
    plt.plot(x, data, label = lab)

plt.title("Development set loss")
plt.legend()

plt.show()
