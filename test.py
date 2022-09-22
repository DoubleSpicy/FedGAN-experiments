from time import sleep
from tqdm import tqdm
lis = ['a', 'b' ,'cdefg']
for a, b in tqdm(enumerate(lis)):
    sleep(1)
    print(a, b)