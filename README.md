# PMSFF: Protein Binding Residues Prediction through Multi-scale Feature Fusion


# Abstract
Accurate prediction of protein binding residues (PBRs) is significant for the understanding of cellular activity and helpful for the design of novel drugs. In recent years, a lot of computational predictors based on machine learning and deep learning models are proposed to accelerate the recognition of PBRs from sequence. However, those methods are still weak in the characterization of residue feature causing low predictive accuracy and limited application range. Specifically, most methods utilize the sliding window approach, which simply concatenate several continuous residue feature vector together. This way does not explore the importance of key residues and only focuses on the local sequential environment. Moreover, an appropriate sliding window length is undetermined when using different predictive models or predicting different PBRs types which results in low predictive accuracy and limited application range. In this study, we propose a sequence-based framework that can be applied to multiple types of PBRs **P**rediciton through **M**ulti-**S**cale **F**eature **F**usion (PMSFF). We adopt a pre-trained language model ProtT5 for encoding each amino acid residue in protein sequence. To improve the sliding window approach, we construct multi-scale residue feature by utilizing multi-size windows for mining more important nerghboring residues and multi-size kernels for lerning information on more scales. Besides, we take a protein sequence as one sentence and use bidirectional GRU for learning global semantic information. We collect and reorganize a series of benchmark datasets including various kinds of PBRs, and PMSFF is trained and evaluated on them. Compared with existing state-of-the-art methods, PMSFF achieves superior performance on most PBRs prediction task.


## 1. Datasets and trainded models
The datasets used for training MsPBRsP and the trained models mentioned in our manuscrpit can be downloaded from https://pan.baidu.com/s/1R1d3ixNpBgTuCY0WvRMftQ （Password: PBRS）

## 2. Requirement
* Python = 3.9.10  
* Pytorch = 1.10.2  
* Scikit-learn = 1.0.2

## 3. Usage
develop_mspbrsp.py provides the code to reproduce the MsPBRsP (hyperparameters can be reset in configs.py).

get_preds_single.py shows an example how to generate binding residue predictions.

ProtT5 embeddings can be generated using bio_embeddings (https://github.com/sacdallago/bio_embeddings).

We provide an example in ./test_data and the ProtT5 embedding of testing protein is saved in a csv file.

## 4. Citation
If you are using MsPBRsP and find it helpful for PBRs prediction, we would appreciate if you could cite the following publication:

[1] Shuai Lu, Yuguang Li, Xiaofei Nan*, Shoutao Zhang*. Attention-based Convolutional Neural Networks for Protein-Protein Interaction Site Prediction[C]. The 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM2021), 2021, 141-144. DOI:10.1109/BIBM52615.2021.9669435.

[2] Yuguang Li, Shuai Lu*, Qiang Ma, Xiaofei Nan, Shoutao Zhang. Protein-Protein Interaction Site Prediction Based on Attention Mechanism and Convolutional Neural Networks[J]. Doi: 10.1109/TCBB.2023.3323493.


## 5. References
[1] Min Zeng, Fuhao Zhang, Fang-Xiang Wu, Yaohang Li, Jianxin Wang, Min Li*. Protein-protein interaction site prediction through combining local and global features with deep neural networks[J]. Bioinformatics, 36(4), 2020, 1114–1120. DOI:10.1093/bioinformaticsz699.  

[2] Bas Stringer*, Hans de Ferrante, Sanne Abeln, Jaap Heringa, K. Anton Feenstra and Reza Haydarlou* (2022). PIPENN: Protein Interface Prediction from sequence with an Ensemble of Neural Nets[J]. Bioinformatics, 38(8), 2022, 2111–2118. DOI:10.1093/bioinformatics/btac071.

[3] Dallago C, Schütze K, Heinzinger M, Olenyi T, Littmann M, Lu AX, Yang KK, Min S, Yoon S, Morton JT, & Rost B (2021). Learned embeddings from deep learning to visualize and predict protein sets[J]. Current Protocols, 1, e113. DOI: 10.1002/cpz1.113

## 6. Contact
For questions and comments, feel free to contact: ieslu@zzu.edu.cn.


