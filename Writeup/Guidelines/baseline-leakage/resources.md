Foundational papers

- Ning, Selesnick & Duval — "Chromatogram baseline estimation and denoising using sparsity", Chemometrics and Intelligent Laboratory Systems, 2014 — the original BEADS paper
- Eilers, P.H.C. — "A perfect smoother", Analytical Chemistry, 2003 — foundation for AsLS family
- Baek et al. — "Baseline correction using asymmetrically reweighted penalized least squares smoothing (arPLS)", Analyst, 2015
- Zhang et al. — "Baseline correction using adaptive iteratively reweighted penalized least squares (airPLS)", Analyst, 2010
- Ryan et al. — "SNIP: Statistics-sensitive nonlinear iterative peak-clipping algorithm", 1988
- Grushka, E. — "Characterization of exponentially modified Gaussian peaks in chromatography", Analytical Chemistry, 1972 — gold standard EMG peak model
- Naish & Hartwell — "Exponentially modified Gaussian functions — a good model for chromatographic peaks?", Chromatographia, 1988


Algorithm unrolling

- Gharbi, Chouzenoux, Pesquet & Duval — "Unrolled deep networks for sparse signal restoration in analytical - chemistry", MLSP / Inria, 2024 — hal.science/hal-04707472v1 — the most directly relevant published work
- Gharbi et al. — "UNROLLED DEEP NETWORKS FOR SPARSE SIGNAL RESTORATION", Inria, hal-03988686v2
- Diamond & Sitzmann — "Unrolled Optimization with Deep Priors", arXiv:1705.08041, 2017 — arxiv.org/pdf/1705.08041
- Zhang & Ghanem — "ISTA-Net+: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing", - CVPR 2018
- Liu et al. — "ALISTA: Analytic weights are as good as learned weights in LISTA", ICLR 2019
- You et al. — "ISTA-Net++: Flexible Deep Unfolding Network for Compressive Sensing", IEEE TIP 2021
- Zheng et al. — "Hybrid ISTA: Unfolding ISTA with convergence guarantees", IEEE TPAMI 2022 — GitHub: ZhengZY-EE/- Hybrid_ISTA
- Yang et al. — "ADMM-Net: A deep learning approach for compressive sensing MRI", NeurIPS 2016
- Monga, Li & Eldar — "Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image - Processing", IEEE Trans. Signal Processing, 2021 — ResearchGate
- DeMUN paper — "Comprehensive Examination of Unrolled Networks for Solving Linear Inverse Problems", Entropy / - MDPI, 2025 — mdpi.com/1099-4300/27/9/929
- ADMM+DEQ survey — "Deep Unrolling with ADMM and LASSO in NLOS, CT, and MRI Inverse Problems" — coder-nova.com


Deep Equilibrium Models

- Bai, Kolter & Koltun — "Deep Equilibrium Models", NeurIPS 2019
- Yu & Dansereau — "MsDC-DEQ-Net: Deep Equilibrium Model with Multi-scale Dilated Convolution", arXiv:2401.02884, 2024 — arxiv.org/abs/2401.02884
- GUDL — "Unsupervised Deep Equilibrium Model Learning for Large-Scale Channel Estimation with Performance Guarantees", arXiv, 2025 — arxiv.org/html/2508.10546


Neural baseline correction (chromatography & spectroscopy)

- Kensert et al. — "Deep convolutional autoencoder for the simultaneous removal of baseline noise and baseline drift in chromatograms", J. Chromatography A, 2021 — ScienceDirect — trained on 190K synthetic chromatograms
- Chen et al. — "Baseline correction using a deep-learning model combining ResNet and UNet", Analyst / RSC, 2022 — pubs.rsc.org
- Han et al. — "Denoising and Baseline Correction Methods for Raman Spectroscopy Based on Convolutional Autoencoder: A Unified Solution", Sensors / MDPI, 2024 — mdpi.com/1424-8220/24/10/3161 / PubMed
- Zhao et al. — "Estimating baselines of Raman spectra based on transformer and manually annotated data", Spectrochimica Acta Part A, 2025 — ScienceDirect
- Hu et al. — "RSPSSL: A novel high-fidelity Raman spectral preprocessing scheme", Light: Science & Applications, 2024 — nature.com
- DIRAS+ — "Adaptive Physics-Aware Raman Baseline Correction with Machine Learning Predicted Parameters", Analytical Chemistry, 2025 — pubs.acs.org
- OP-airPLS — "Beyond Traditional airPLS: Improved Baseline Removal in SERS with Parameter-Focused Optimization and Prediction", Analytical Chemistry, 2025 — pubs.acs.org
- arPLS GUI paper — "A graphical user interface for arPLS baseline correction", ScienceDirect, 2023 — ScienceDirect


Architecture references

- Gulati et al. — "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech 2020
- Luo et al. — "DF-Conformer: Integrated architecture of Conv-TasNet and Conformer using linear complexity self-attention for speech enhancement", Interspeech 2022 — ResearchGate
- Kim & Lee — "Hybrid dual-path network: Singing voice separation combining Conformer and Transformer", Speech Communication, 2024 — ScienceDirect
- Stoller et al. — "Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation", ISMIR 2018
- van den Oord et al. — "WaveNet: A Generative Model for Raw Audio", arXiv:1609.03499, 2016 — arxiv.org/pdf/1609.03499
- Nguyen et al. — "A Novel Approach to WaveNet Architecture for RF Signal Separation with Learnable Dilation", ICASSP 2024 — arxiv.org/html/2402.09461v1
- Gu & Dao — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023 — AI Intuition primer


Orthogonality / NMF references

- Ding et al. — "Orthogonal Nonnegative Matrix Tri-Factorizations for Clustering", KDD 2006
- Pompili et al. — "Two algorithms for orthogonal nonnegative matrix factorization with application to clustering", IEEE TKDE 2014
- Selesnick & Chen — "Total Variation Denoising with Overlapping Group Sparsity", ICASSP 2013


Curriculum learning

- Hacohen & Weinshall — "On the Power of Curriculum Learning in Training Deep Networks", ICML 2019


Code / libraries

- pybaselines — github.com/derb12/pybaselines — 50+ algorithms, unified API, BSD-3
- torchdeq — PyTorch DEQ library for deep equilibrium models
- Conformer PyTorch — github.com/sooftware/conformer
- Kensert autoencoder — github.com/akensert/autoencoder-chromatogram-enhancement

