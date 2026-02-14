\# Bacterial Genus Classification from 16S rRNA Sequences



Machine learning pipeline for classifying bacterial genera using k-mer features and XGBoost/CNN models on the RDP Trainset 18 dataset.



\## Project Overview



This project develops high-accuracy machine learning classifiers for genus-level bacterial identification from 16S rRNA gene sequences. Achieved \*\*96.91-97.22% accuracy\*\* using XGBoost and CNN models, outperforming traditional alignment-based methods.



\## Key Results



\- \*\*Top-100 Model\*\*: 96.91% accuracy (XGBoost), 97.22% accuracy (CNN)

\- \*\*Full Model\*\*: 82.84% accuracy across 1,712 genera

\- \*\*Dataset\*\*: 19,927 sequences from RDP Trainset 18

\- Addressed class imbalance using SMOTE

\- GPU-accelerated training with feature importance analysis



\## Technical Highlights



\- \*\*Feature Engineering\*\*: K-mer extraction (4-5 bp) with binary/count encoding

\- \*\*Models\*\*: XGBoost (gradient boosting), CNN (deep learning)

\- \*\*Data Processing\*\*: BioPython for FASTA parsing, scikit-learn for vectorization

\- \*\*Performance\*\*: 5-fold cross-validation, 94.48% ± 0.58% robustness



\## Files



\- `Final-16S rRNA Sequence Classification.ipynb` - Main classification pipeline

\- `Final-CNN.ipynb` - CNN model implementation

\- `Final-Classifying\_Bacterial\_Genera\_from\_16S\_rRNA\_Sequences\_Using\_K\_mer\_Features\_and\_XGBoost\_with\_the\_RDP\_Trainset\_18\_Dataset.pdf` - Full research paper



\## Technologies Used



\- Python (pandas, NumPy, scikit-learn, XGBoost, BioPython)

\- Machine Learning (classification, SMOTE, cross-validation)

\- Deep Learning (CNN with TensorFlow/Keras)

\- GPU acceleration (NVIDIA RTX 4080)



\## Author



Gregory Luna  

Long Island University - Computer Science  

January 2025



\## Citation



If you use this work, please cite:

```

Luna, G. (2025). Classifying Bacterial Genera from 16S rRNA Sequences Using K-mer Features 

and XGBoost with the RDP Trainset 18 Dataset. Long Island University.

```

