import tensorflow as tf

class Matrix_BPR(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=15,
                 master_learning_rate=0.01,
                 ):
        
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.master_learning_rate = master_learning_rate
        self.user_positive_items_pairs = tf.placeholder(tf.int64, [None, 2])
        self.negative_samples = tf.placeholder(tf.int64, [None, None])
        self.score_user_ids = tf.placeholder(tf.int64, [None])
        self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32,seed=1))     
        self.item_embeddings = tf.Variable(tf.random_normal([n_items, embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32,seed=2))   


        self.user_select = tf.nn.embedding_lookup(self.user_embeddings,self.user_positive_items_pairs[:,0])
        self.pos_select = tf.nn.embedding_lookup(self.item_embeddings,self.user_positive_items_pairs[:,1])
        self.neg_select = tf.nn.embedding_lookup(self.item_embeddings,self.negative_samples[:,0])        
        self.pos_item = tf.reduce_sum(tf.multiply(self.user_select,self.pos_select),axis=1)
        self.neg_item = tf.reduce_sum(tf.multiply(self.user_select,self.neg_select),axis=1)
        self.l2 = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)                             
        self.loss =   -tf.reduce_sum(tf.log(tf.sigmoid(self.pos_item- self.neg_item))) +0.075*self.l2
        self.optimize = tf.train.AdagradOptimizer(self.master_learning_rate).minimize(self.loss) 
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def partial_fit(self, X,Y):
        self.sess.run( (self.optimize), feed_dict = {self.user_positive_items_pairs: X,self.negative_samples: Y})


def load_from_sparse_matrix(self, data):
    user_set = set(data[data[:, 2]==2, 0]) & set(data[data[:, 2]==1, 0])
    item_set = set(data[:, 1])

    self._n_users = len(user_set)
    self._users = list(range(self._n_users))
    self._n_items = len(item_set)
    self._items = list(range(self._n_items))
    
    self._user_index = dict(zip(user_set, list(range(len(user_set)))))
    self._index_user = dict(zip(list(range(len(user_set))), user_set))
    self._item_index = dict(zip(item_set, list(range(len(item_set)))))
    self._index_item = dict(zip(list(range(len(item_set))), item_set))

    pos_dict = dict()
    match_dict = dict()
    for (user, item, rate) in data:
        if not (user in self._user_index and item in self._item_index):
            continue
        user = self._user_index[user]
        item = self._item_index[item]
        if rate == 2:
            if user not in match_dict:
                match_dict[user] = list()
            match_dict[user].append(item)
        if rate == 1:
            if user not in pos_dict:
                pos_dict[user] = list()
            pos_dict[user].append(item)
    # for u in set(match_dict.keys()) - set(pos_dict.keys()):
    #     match_dict.pop(u)
    # for u in set(pos_dict.keys()) - set(match_dict.keys()):
    #     pos_dict.pop(u)
    return match_dict, pos_dict