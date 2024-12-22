# Dual-Branch-Neural-Network_Cas13a
A dual-branch neural network model for predicting crRNA activity in CRISPR/Cas13a tasks.
# 1 Data Statement
The original CRISPR/LwaCas13a dataset and the CRISPR/LbuCas13a dataset used in this code are both derived from the paper by Hayden C. Metsky et al.

If you use this data in your work, please cite:

  Metsky, H. C.; Welch, N. L.; Pillai, P. P.; Haradhvala, N. J.; Rumker, L.; Mantena, S.; Zhang, Y. B.; Yang, D. K.; Ackerman, C. M.; Weller, J.; et al. Designing 
  Sensitive Viral Diagnostics with Machine Learning. Nat. Biotechnol. 2022, 40 (7), 1123-1131. DOI: 10.1038/s41587-022-01213-5.
  
The datasets in the 'data/' folder of this project are data that have been encoded with one-hot (2D) encoding and descriptive feature encoding.
# 2 Model Training And Predicting
Here, we take training the LwaCas13a classification model as an example：

  python LwaCas13a_Classification_Model.py

