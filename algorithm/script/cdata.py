import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def run():

    # 性别字典
    with open('../data/gender.dat') as f:
        gender_dict = dict()
        for row in tqdm(f):
            user_id, gender = row.strip().split(',')
            gender_dict[user_id] = gender

    # 读取评分文件 构造字典
    with open('../data/ratings.dat') as f:
        male_rating = 0
        female_rating = 0
        male_rating_dict = {}
        famale_rating_dict = dict()
        for row in tqdm(f):
            user_id, item_id, rate = row.strip().split(',')
            rate = int(rate)
            if rate > 5:
                user = gender_dict[user_id] + str(user_id)
                item = gender_dict[item_id] + str(item_id)
                if user[0] == 'M' and item[0] == 'F':
                    male_rating += 1
                    if user not in male_rating_dict:
                        male_rating_dict[user] = dict()
                    male_rating_dict[user][item] = rate
                elif user[0] == 'F' and item[0] == 'M':
                    female_rating += 1
                    if item not in famale_rating_dict:
                        famale_rating_dict[item] = dict()
                    famale_rating_dict[item][user] = rate



    # 从评分中构造匹配字典 匹配列表
    match_dict = dict()
    match_list = list()
    for user, item_rate_dict in tqdm(male_rating_dict.items()):
        for item, rate in item_rate_dict.items():
            if rate > 5:
                if user in famale_rating_dict:
                    if item in famale_rating_dict[user]:
                        rate_ = famale_rating_dict[user][item]
                        if rate_ > 5:
                            if user not in match_dict:
                                match_dict[user] = dict()
                            match_dict[user][item] = rate + rate_
                            match_list.append([user, item, rate + rate_])


    # 构造匹配记录表
    match_frame = pd.DataFrame(match_list, columns=['male', 'female', 'rate'])
    print('users with matches', len(set(match_frame.iloc[:, 0])))


    def count_degree(frame, col):
        user = frame.columns[col]
        user_degree_series = frame.iloc[:, col]
        user_degree_frame = pd.DataFrame(user_degree_series.value_counts())
        user_degree_frame.columns = ['degree']
        user_degree_frame = pd.merge(frame, user_degree_frame,
                                    left_on=user, right_index=True)
        return user_degree_frame


    def filter_old(frame, N=0, M=100000):
        # 筛选老用户
        frame = count_degree(frame, 0)
        frame = count_degree(frame, 1)
        old_frame = frame[(frame['degree_x'] >= N) & (frame['degree_y'] >= N)]
        # print('rest users', len(set(old_frame.iloc[:, 0])))
        # print('rest items', len(set(old_frame.iloc[:, 1])))
        # print('rest matches', len(old_frame))
        return old_frame.iloc[:, :3]


    def iter_filter_old(frame, N=0, M=100000, step=100):
        for i in range(step):
            frame = filter_old(frame.iloc[:, :3], N, M)
            frame = count_degree(frame, 0)
            frame = count_degree(frame, 1)        
            if (frame['degree_x'].min() >= N and frame['degree_y'].min() >= N):
                break
        return frame.iloc[:, :3]


    # 输出迭代消去的数据损失量
    for i in [3]:
        print('least match for old user', i)
        old_match_frame = iter_filter_old(match_frame, i)
        old_male_set = set(old_match_frame['male'])
        old_female_set = set(old_match_frame['female'])


    def build_pos_data(old_male_set, old_female_set, male_rating_dict, male_match_dict, col):
        # 组合match和positive
        positive_data = list()
        for user in old_male_set:
            items = set(male_rating_dict[user].keys()) & old_female_set - set(male_match_dict[user].keys())
            positive_data.extend([[user, item, 1] for item in items])
            # for item in items:
                # if item in male_match_dict[user]:
                #     continue
                # else:
                #     positive_data.append([user, item, 1])

        return pd.DataFrame(positive_data, columns=col)

    male_posi_data = build_pos_data(
        old_male_set, old_female_set, male_rating_dict, match_dict, col=['male', 'female', 'rate'])
    female_posi_data = build_pos_data(
        old_male_set, old_female_set, famale_rating_dict, match_dict, col=['male', 'female', 'rate'])
    print('rest users', len(old_male_set))
    print('rest items', len(old_female_set))
    print('rest matches', len(old_match_frame))
    print('male positive num', len(male_posi_data))
    print('female positive num', len(female_posi_data))
    print('sparseness:', len(old_match_frame) / len(old_male_set) / len(old_female_set),
                         len(male_posi_data) / len(old_male_set) / len(old_female_set),
                         len(female_posi_data) / len(old_male_set) / len(old_female_set))
    

    # 划分数据
    old_match_frame['rate'] = 2

    print('male bsr', len(old_match_frame) / len(male_posi_data))
    print('female bsr', len(old_match_frame) / len(female_posi_data))


    match_train, match_test = train_test_split(old_match_frame, test_size=0.5)

    male_posi_train, male_posi_test = train_test_split(male_posi_data, test_size=0.5)
    female_posi_train, female_posi_test = train_test_split(female_posi_data, test_size=0.5)
    male_train = pd.concat([match_train, male_posi_train])
    male_test = pd.concat([match_test, male_posi_test])
    female_train = pd.concat([match_train, female_posi_train])
    female_test = pd.concat([match_test, female_posi_test])

    male_train.to_csv('../data/male_train.csv', index=False)
    male_test.to_csv('../data/male_test.csv', index=False)
    female_train.to_csv('../data/female_train.csv', index=False)
    female_test.to_csv('../data/female_test.csv', index=False)

    return True

if __name__ == '__main__':
    run()
