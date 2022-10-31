
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

import pdb
# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22
class SimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        # training_set,test_set  [[user_id, item_id, float(weight)]...]

        super(SimGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.temp = float(args['-temp'])
        self.pool='mean'   # "[concat, mean, sum, final]"
        self.decay = 1e-4
        self.cl_weight=float(args['-cl_weight'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb ,rec_user_emb_all_layer,rec_item_emb_all_layer= model()

                user_emb, pos_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx]
                user_emb_all_layer, pos_item_emb_all_layer = rec_user_emb_all_layer[user_idx], rec_item_emb_all_layer[pos_idx]
                # <-----------------------------------------------------------------------

                neg_gcn_embs_mixgcf = []
                neg_gcn_embs_mixgcf.append(self.negative_sampling(rec_user_emb_all_layer, rec_item_emb_all_layer,
                                                           user_idx, neg_idx,
                                                           pos_idx))
                neg_gcn_embs = torch.stack(neg_gcn_embs_mixgcf, dim=1)

                pdb.set_trace()
                rec_loss, _, _= self.create_bpr_loss(user_emb_all_layer, pos_item_emb_all_layer, neg_gcn_embs)    # MixGCF的BPR loss计算方式

                neg_item_emb = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3]))  # 这里是Mix的方法
                # neg_item_emb = rec_item_emb[neg_idx[:,:1]]  # 这里是SimGCL原本的方法
                user_emb = self.pooling(user_emb_all_layer)
                pos_item_emb = self.pooling(pos_item_emb_all_layer)
                # <-----------------------------------------------------------------------

                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) #人家原本的BPR loss计算方式
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx]) * self.cl_weight
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss
                # neg_item_emb.shape = torch.Size([2048, 64])

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # if n % 100==0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb,_,_ = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 ,_,_= self.model(perturbed=True)
        user_view_2, item_view_2 ,_,_= self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb ,_,_= self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        """user : 指的是uid
            neg_candidates : neg_item_id
            pos_item : pos_item_id
        """
        batch_size = len(user)
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing 往里面插入一个维度
        # n_e.shape = torch.Size([1024, 64, 4, 64])
        # p_e.shape = torch.Size([1024, 4, 64])
        # seed.shape = torch.Size([1024, 1, 4, 1])

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
                              range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel] = torch.Size([1024, 4, 64])
        # pos_gcn_embs: [batch_size, n_hops+1, channel] = torch.Size([1024, 4, 64])
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]  = torch.Size([1024, 1, 4, 64]) 这里变成4维了，和其他两个tensor维度对不上了

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb) # u_e.shape = torch.Size([1024, 256])
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, 1, -1)
        # 这里把 neg_gcn_embs([2048, 1, 4, 64]) view成为([2048, 4, 64])，再 pooling ，再恢复为([2048, 1, 256])

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1) # pos_scores.shape = torch.Size([1024])
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss





class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        embs=[] # 作者说的，要跳过最原始的E_0

        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            embs.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        embs = torch.stack(embs,dim=1)
        user_all_embeddings_all_layer, item_all_embeddings_all_layer = torch.split(embs, [self.data.user_num, self.data.item_num])



        return user_all_embeddings, item_all_embeddings,user_all_embeddings_all_layer,item_all_embeddings_all_layer







