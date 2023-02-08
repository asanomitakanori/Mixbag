import numpy as np 
import matplotlib.pyplot as plt

dataset = 'svhn'

test_data = np.load('data/%s/test_data.npy'%(dataset))
test_label = np.load('data/%s/test_label.npy'%(dataset))
bags = np.load('data/%s/bags.npy'%(dataset))
labels = np.load('data/%s/labels.npy'%(dataset))
lps = np.load('data/%s/origin_lps.npy'%(dataset))

idx = 0

bag = np.zeros((32*8, 32*8, 3))
for i in range(8):
    for j in range(8):
        bag[32*i: 32*(i+1), 32*j: 32*(j+1)] = bags[idx][i*8+j]

plt.imshow(bag/255)
plt.title('%s'%(lps[idx]))
plt.axis('off')
plt.savefig('result/svhn.png')