# Compound-protein interactions

#### Description
Hypergraph-based dual-channel improved variational autoencoder with cross-attention for compound-protein interactions identification

#### Dependencies
Matlab >= 2024b was used to construct model for identifying compound-protein interactions.
Python 
dhg (https://github.com/Accenture/AmpliGraph) was employed to perform embedded learning for knowledge graph.
torch (https://github.com/benedekrozemberczki/karateclub) was utilized to perform embedding learning for complex network.

#### Instructions
(1) Collect compound-protein interaction information, chemical SMILEs and protein sequence data from databases DrugBank and UniprotKB, and calculate molecular fingerprint descriptors and protein primary structure features.
(2) Construct a hypergraph with compounds as vertices and proteins as hyperedges, and a hypergraph with compounds as hyperedges and proteins as vertices, based on collected compound-protein interaction data. 
(3) Two hypergraph structures and corresponding vertex feature matrices are input into the improved hypergraph variational autoencoder, and the obtained embedded features are used to represent compounds and proteins, respectively.
(4) Multi-head cross-attention operations are performed on the embedded features of proteins and compounds to obtain fusion features that capture their interaction information. Then, the fusion features are fed into deep neural network model to identify potential compound-protein interactions.  

#### Methods
AMCF_RDP('D019829','ALBU_HUMAN')

#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
