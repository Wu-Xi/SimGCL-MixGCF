
<h2>Requirements</h2>
	
```
numba==0.53.1
numpy==1.20.3
scipy==1.6.2
tensorflow==1.14.0
torch>=1.7.0
```

<h2>Usage</h2>
<ol>
<li>Configure the xx.conf file in the directory named conf. (xx is the name of the model you want to run)</li>
<li>Run main.py and choose the model you want to run.</li>
</ol>

<h2>Implemented Models</h2>

<table class="table table-hover table-bordered">
  <tr>
		<th>Model</th> 		<th>Paper</th>      <th>Type</th>   <th>Code</th>
   </tr>
   <tr>
    <td scope="row">XSimGCL</td>
        <td>Yu et al. <a href="https://arxiv.org/abs/2209.02544" target="_blank">XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation</a>, Submitted to TKDE.
         </td> <td>Graph + CL</d> <td>PyTorch</d> 
      </tr>
   <tr>
    <td scope="row">SimGCL</td>
        <td>Yu et al. <a href="https://arxiv.org/abs/2112.08679" target="_blank">Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation</a>, SIGIR'22.
         </td> <td>Graph + CL</d> <td>PyTorch</d> 
      </tr>
   <tr>
    <td scope="row">DirectAU</td>
        <td>Wang et al. <a href="https://arxiv.org/abs/2206.12811" target="_blank">Towards Representation Alignment and Uniformity in Collaborative Filtering</a>, KDD'22.
         </td> <td>Graph</d> <td>PyTorch</d> 
      </tr>   
<tr>
    <td scope="row">NCL</td>
        <td>Lin et al. <a href="https://arxiv.org/abs/2202.06200" target="_blank">Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning</a>, WWW'22.
         </td> <td>Graph + CL</d> <td>PyTorch</d> 
      </tr>
   <tr>
    <td scope="row">MixGCF</td>
        <td>Huang et al. <a href="https://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf" target="_blank">MixGCF: An Improved Training Method for Graph Neural
Network-based Recommender Systems</a>, KDD'21.
         </td> <td>Graph + DA</d> <td>PyTorch</d> 
      </tr>
     <tr>
    <td scope="row">MHCN</td>
        <td>Yu et al. <a href="https://dl.acm.org/doi/abs/10.1145/3442381.3449844" target="_blank">Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation</a>, WWW'21.
         </td> <td>Graph + CL</d> <td>TensorFlow</d>
      </tr>
     <tr>	
    <td scope="row">SGL</td>
        <td>Wu et al. <a href="https://dl.acm.org/doi/10.1145/3404835.3462862" target="_blank">Self-supervised Graph Learning for Recommendation</a>, SIGIR'21.
         </td> <td>Graph + CL</d> <td>TensorFlow & Torch</d> 
      </tr>
    <tr>
    <td scope="row">SEPT</td>
        <td>Yu et al. <a href="https://arxiv.org/abs/2106.03569" target="_blank">Socially-Aware Self-supervised Tri-Training for Recommendation</a>, KDD'21.
         </td> <td>Graph + CL</d> <td>TensorFlow</d> 
      </tr>
          <tr>
    <td scope="row">BUIR</td>
        <td>Lee et al. <a href="https://arxiv.org/abs/2105.06323" target="_blank">Bootstrapping User and Item Representations for One-Class Collaborative Filtering</a>, SIGIR'21.
         </td> <td>Graph + DA</d> <td>PyTorch</d>
      </tr>
        <tr>
    <td scope="row">SSL4Rec</td>
        <td>Yao et al. <a href="https://dl.acm.org/doi/abs/10.1145/3459637.3481952" target="_blank">Self-supervised Learning for Large-scale Item Recommendations</a>, CIKM'21.
	     </td> <td>Graph + CL</d>  <td>PyTorch</d>
      </tr>
    <tr>
    <td scope="row">SelfCF</td>
        <td>Zhou et al. <a href="https://arxiv.org/abs/2107.03019" target="_blank">SelfCF: A Simple Framework for Self-supervised Collaborative Filtering</a>, arXiv'21.
         </td> <td>Graph + DA</d> <td>PyTorch</d>
      </tr>
    <tr>
    <td scope="row">LightGCN</td>
        <td>He et al. <a href="https://dl.acm.org/doi/10.1145/3397271.3401063" target="_blank">LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation</a>, SIGIR'20.
	     </td> <td>Graph</d>  <td>PyTorch</d>
      </tr>
         <tr>
    <td scope="row">MF</td>
        <td>Yehuda et al. <a href="https://ieeexplore.ieee.org/abstract/document/5197422" target="_blank">Matrix Factorization Techniques for Recommender Systems</a>, IEEE Computer'09.
	     </td> <td>Graph</d>  <td>PyTorch</d> 
      </tr>
  </table>  
* CL is short for contrastive learning (including data augmentation); DA is short for data augmentation only

<h2>Leaderboard</h2>
The results are obtained on the dataset of <b>Yelp2018</b>. We performed grid search for the best hyperparameters. <br>
General hyperparameter settings are: batch_size: 2048, emb_size: 64, learning rate: 0.001, L2 reg: 0.0001. <br><br>


|  Model   |      Recall@20      | NDCG@20 | Hyperparameter settings                                                                             |
|:--------:|:-------------------:|:-------:|:----------------------------------------------------------------------------------------------------|
|   MF    |       0.0543        | 0.0445  |          |
|   LightGCN    |       0.0639        | 0.0525  |     layer=3     |
|   NCL    |       0.0670        | 0.0562  | layer=3, ssl_reg=1e-6, proto_reg=1e-7, tau=0.05, hyper_layers=1, alpha=1.5, num_clusters=2000 |
|   SGL    |       0.0675        | 0.0555  |     λ=0.1, ρ=0.1, tau=0.2 layer=3     |
|  MixGCF  |       0.0691        | 0.0577  |      layer=3, n_nes=64, layer=3       |
| DirectAU |       0.0695        | 0.0583  |             𝛾=2, layer=3             |
|  SimGCL  |       0.0721        | 0.0601  |   λ=0.5, eps=0.1, tau=0.2, layer=3    |
| XSimGCL  |       0.0723        | 0.0604  | λ=0.2, eps=0.2, l∗=1 tau=0.15 layer=3 |

<h2>Implement Your Model</h2>
 
1. Create a **.conf** file for your model in the directory named conf.
2. Make your model **inherit** the proper base class.
3. **Reimplement** the following functions.
	+ *build*(), *train*(), *save*(), *predict*()
4. Register your model in **main.py**.



<h2>Related Datasets</h2>
<div>
 <table class="table table-hover table-bordered">
  <tr>
    <th rowspan="2" scope="col">Data Set</th>
    <th colspan="5" scope="col" class="text-center">Basic Meta</th>
    <th colspan="3" scope="col" class="text-center">User Context</th> 
    </tr>
  <tr>
    <th class="text-center">Users</th>
    <th class="text-center">Items</th>
    <th colspan="2" class="text-center">Ratings (Scale)</th>
    <th class="text-center">Density</th>
    <th class="text-center">Users</th>
    <th colspan="2" class="text-center">Links (Type)</th>
    </tr>   
   <tr>
    <td><a href="https://pan.baidu.com/s/1hrJP6rq" target="_blank"><b>Douban</b></a> </td>
    <td>2,848</td>
    <td>39,586</td>
    <td width="6%">894,887</td>
    <td width="10%">[1, 5]</td>
    <td>0.794%</td>
    <td width="4%">2,848</td>
    <td width="5%">35,770</td>
    <td>Trust</td>
    </tr> 
	 <tr>
    <td><a href="http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip" target="_blank"><b>LastFM</b></a> </td>
    <td>1,892</td>
    <td>17,632</td>
    <td width="6%">92,834</td>
    <td width="10%">implicit</td>
    <td>0.27%</td>
    <td width="4%">1,892</td>
    <td width="5%">25,434</td>
    <td>Trust</td>
    </tr> 
    <tr>
    <td><a href="https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=0" target="_blank"><b>Yelp</b></a> </td>
    <td>19,539</td>
    <td>21,266</td>
    <td width="6%">450,884</td>
    <td width="10%">implicit</td>
    <td>0.11%</td>
    <td width="4%">19,539</td>
    <td width="5%">864,157</td>
    <td>Trust</td>
    </tr>
    <tr>
    <td><a href="https://www.dropbox.com/sh/20l0xdjuw0b3lo8/AABBZbRg9hHiN42EHqBSvLpta?dl=0" target="_blank"><b>Amazon-Book</b></a> </td>
    <td>52,463</td>
    <td>91,599</td>
    <td width="6%">2,984,108</td>
    <td width="10%">implicit</td>
    <td>0.11%</td>
    <td width="4%">-</td>
    <td width="5%">-</td>
    <td>-</td>
    </tr>  
  </table>
</div>



