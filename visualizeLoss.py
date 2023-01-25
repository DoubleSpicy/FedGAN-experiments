import matplotlib.pyplot as plt
import numpy as np
import re
paths = ['''../results/jan6/job14d2022-11-14-21_29_17.txt''', '''../results/jan6/job14c2022-11-14-20_19_20.txt''']
# paths = ['''../results/jan6/job14d2022-11-14-07_48_44.txt''', '''../results/jan6/job14c2022-11-14-07_14_41.txt''']
plt.yticks(np.arange(-300, 300, 10))
f, axs = plt.subplots(2, 2)
f.suptitle("WGAN-GP, celebA, 2 out of 5 +eyeglass")
client = 0
client_cnt = 5
n_critic = 0
row = 0
def parse(path):
    global client, client_cnt, n_critic, row
    g_loss, d_loss = [[] for x in range(client_cnt)], [[[], []] for x in range(client_cnt)]
    f = open(path, 'r')
    for line in f:
        if 'loss' not in line:
            continue
        match = re.search('loss.*: .+(,|$)', line)
        match = match.group().split(', ')
        for idx, text in enumerate(match):
            match[idx] = float(text[text.find(' ')+1:])
        if len(match) == 1: # is Generator
            g_loss[client].append(match[0])
            client = (client + 1) % client_cnt
        elif len(match) == 2:
            if n_critic == 10:
                d_loss[client][0].append(match[0])
                d_loss[client][1].append(match[1])
                n_critic = 0
            n_critic += 1

    draw(axs=axs, row=row, g_loss=g_loss, d_loss=d_loss)
    row += 1

def draw(axs, row, g_loss, d_loss):
    if row == 0:
        title = 'not share D'
    else:
        title = 'share D'
    axs[row, 0].plot(g_loss[0], label='g_loss')
    for i in range(client_cnt):
        # plt.plot(d_loss[i][1], label=f'd_real_loss[{i}]')
        axs[row, 0].plot(d_loss[i][0], label=f'd_fake_loss[{i}]')
    leg = axs[row, 0].legend(loc='best')
    axs[row, 0].title.set_text(title)
    axs[row, 1].plot(g_loss[0], label='g_loss')
    for i in range(client_cnt):
        # plt.plot(d_loss[i][1], label=f'd_real_loss[{i}]')
        axs[row, 1].plot(d_loss[i][1], label=f'd_real_loss[{i}]')
    leg = axs[row, 1].legend(loc='best')
    axs[row, 1].title.set_text(title)

for i in paths:
    parse(i)


# plt.plot(g_loss)
plt.show()