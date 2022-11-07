


2022_11_7 LightGCN_with_full_cl_loss.py


在原来的基础上加入了全套的对比损失。（both positive pairs and negative pairs）

we use "mixup" as data augmentation to synthetize hard negative item , we also put hard negative item into BPR loss.

我们认为既然hard negative item是由pos_item and randon_negative_item来线性插值合成的，那它在这两者之间肯定有一个bias，随着随机采样的weight不同，bias的方向也不同。我需要它远离randon_negative_item ，同时在语义上尽可能的靠近pos_item。由此产生的伪views，带入cl_loss。

consider pos_item and hard_negative_item as positive pairs.

consider origin_negative_item and hard_negative_item as negative pairs.



