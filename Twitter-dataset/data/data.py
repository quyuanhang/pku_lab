import tqdm

following = dict()
friendship = dict()
friend_pool = set()
friend = 0

with open('edges.csv') as file:
    for row in tqdm.tqdm(file):
        u, v = row.strip().split(',')
        u, v = int(u), int(v)
        following.setdefault(u, set())
        following[u].add(v)
        if v in following and u in following[v]:
            friend += 1
            friendship.setdefault(u, set())
            friendship[u].add(v)
            friendship.setdefault(v, set())
            friendship[v].add(u)
            friend_pool.update([u, v])


print(friend) # 21776096  85331845-2*21776096 = 41779653
print(len(friend_pool)) # 3408836
