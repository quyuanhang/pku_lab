import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('edges.csv').sample(frac=0.05).values

following = dict()
friendship = dict()
friend_pool = set()
friendship_list = list()
following_list = list()

for row in tqdm.tqdm(data):
    u, v = row
    u, v = int(u), int(v)
    following.setdefault(u, set())
    following[u].add(v)
    if v in following and u in following[v]:
        friendship.setdefault(u, set())
        friendship[u].add(v)
        friendship.setdefault(v, set())
        friendship[v].add(u)
        friend_pool.update([u, v])

old = 0
for u, v_set in tqdm.tqdm(friendship.items()):
    if len(v_set) > 5:
        old += 1
        friendship_list.extend([[u, v, 2] for v in v_set])
        following_list.extend([u, v, 1] for v in following[u] if (v in friend_pool))

friend_train, friend_test = train_test_split(pd.DataFrame(friendship_list), test_size=0.5)
following_train, following_test = train_test_split(pd.DataFrame(following_list), test_size=0.5)

train = pd.concat([friend_train, following_train])
test = pd.concat([friend_test, following_test])

train.to_csv('train.csv', index=False, header=False)
test.to_csv('test.csv', index=False, header=False)