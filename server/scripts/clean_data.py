import pandas as pd
import json

with open('../data/nobel_.csv') as f:
    frame = pd.read_csv(f, error_bad_lines=False)
    frame = frame.drop_duplicates(subset=['Year', 'Birth Date', 'Full Name'])
    frame = frame.loc[:, ['Year', 'Category', 'Prize Share', 'Birth Country']]
    frame.columns = ('year', 'category', 'share', 'country')
    frame = frame.dropna(subset=['year']).fillna(value=0)
    frame['year'] = frame['year'].map(int)
    frame['share'] = frame['share'].map(lambda x: 1 / int(x[-1]))
    frame['country'] = frame['country'].map(
            lambda x: x[x.find('(')+1: -1] if type(x)==str and x.find('(') != -1 else x)

nodes = list()
links = list()

        
frame = frame.sort_values(by=['year'])
frame.index = frame['year']
min_year = int(frame.year.min())
max_year = int(frame.year.max())

categorys = ['Physics', 'Medicine', 'Chemistry', 'Peace', 'Literature']
node_keys = ('year', 'category', 'share', 'country')

def save_node(node, nodes_dict, nodes_list, node_index):
    for key, _node in nodes_dict.items():
        if node['country'] == _node['country']:
            _node['category'].setdefault(list(node['category'].keys())[0], 0)
            _node['category'][list(node['category'].keys())[0]] += node['share']
            nodes_dict[key] = _node
            nodes_list[_node['index']] = _node
            return 0
    index = node_index.generate_index()
    node['index'] = index
    nodes_dict[index] = node
    nodes_list.append(node)
    return 0    


class NodeIndex():
    def __init__(self):
        self.node_num = -1
    def generate_index(self):
        self.node_num += 1
        return self.node_num

node_index = NodeIndex()
last = dict()
for year in range(min_year, max_year):
    if year == 1901:
        categorys.append('Economics')
    try:
        current_raw = frame.loc[year]
    except:
        current_raw = pd.DataFrame(index=[year]*len(categorys), columns=node_keys)
        current_raw['year'] = year
        current_raw['category'] = categorys
        current_raw['share'] = 1
        current_raw['country'] = 'no'
    if type(current_raw) == pd.Series:
        current_raw = pd.DataFrame(current_raw).T
    tmp = dict()
    for category in categorys:
        cat_data = current_raw[current_raw['category']==category]
        if cat_data.empty:
            node = dict(zip(node_keys, (year, {category: 1}, 1, 'no')))
            save_node(node, tmp, nodes, node_index)
        else:
            for i, row in cat_data.iterrows():
                node = dict(row)
                node['year'] = year
                node['category'] = {node['category']: node['share']}
                save_node(node, tmp, nodes, node_index)
    if not last:
        last = tmp
        continue
    for last_node in last.values():
        for cur_node in tmp.values():
            same_categorys = set(last_node['category'].keys()) & set(cur_node['category'].keys())
            if same_categorys:
                for category in same_categorys:
                    links.append({
                        'source': last_node['index'],
                        'target': cur_node['index'],
                        'value': last_node['category'][category] * cur_node['category'][category],
                        'category': category
                    })
    last = tmp


data = {'nodes': nodes, 'links': links}
with open('../data/nobel_.json', 'w+') as f:
    json.dump(obj=data, fp=f, indent=4)
