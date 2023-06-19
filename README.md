# MsPBRsP: Multi-scale Protein Binding Residues Prediction Using Language Model
This repository is the implementation of a bioRXiv paper:" MsPBRsP: Multi-scale Protein Binding Residues Prediction Using Language Model" (https://www.biorxiv.org/content/10.1101/2023.02.26.528265v1)

# Abstract
Accurate prediction of protein binding residues (PBRs) from sequence is important for the understanding of cellular activity and helpful for the design of novel drug. However, experimental methods are time-consuming and expensive. In recent years, a lot of computational predictors based on machine learning and deep learning models are proposed to reduce such consumption. Those methods often use Multiple Sequence Alignment (MSA) tools such as PSI-BLAST or MMseqs2 to generate evolutionary features as well as NetSurfP or PSIPRED to obtain predicted features. And then, those features will be entered into predictive models as necessary supplementary input. The input generation process normally takes long time, and there is no standard to specify which and how many statistic results should be provided to a prediction model. In addition, prediction of PBRs relies on residue local context, but the most appropriate scale is undetermined. Most works pre-selected certain residue features as input and a scale size based on expertise for certain type of PBRs. In this study, we propose a general tool-free end-to-end framework that can be applied to all types of PBRs, **M**ulti-**s**cale **P**rotein **B**inding **R**esidue**s** **P**rediction using language model (MsPBRsP). We adopt a pre-trained language model ProtT5 to save the large consumption caused by MSA tools, and use protein sequence alone as input to our model. To ease scale size uncertainty, we construct multi-size windows in attention layer and multi-size kernels in convolutional layer. We test our framework on various benchmark datasets including PBRs from protein-protein, protein-nucleotide, protein-small ligand, heterodimer, homodimer and antibody-antigen interactions. Compared with existing state-of-the-art methods, MsPBRsP achieves superior performance with less running time and higher prediction rates on most PBRs prediction task. Specifically, MsPBRsP surpasses the second best method by as much as 44.0\% and 52.8\% at AUPRC and MCC, respectively. And, it decrease running time greatly. The source code and datasets are available at https://github.com/biolushuai/MsPBRsP-for-multiple-PBRs-prediction.


# 1. Datasets
Baidu Netdisk：https://pan.baidu.com/s/1GUO57KPy27t7UUBn8XTt1Q （Password: PBRS）

# 2. Requirement
* Python = 3.9.10  
* Pytorch = 1.10.2  
* Scikit-learn = 1.0.2

# 6. Contact
For questions and comments, feel free to contact : ielushuai@126.com.


