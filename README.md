# CS6780 - Advanced Machine Learning

course website: https://www.cs.cornell.edu/courses/cs6780/2019sp/

## Overview

This course gives a graduate-level introduction to machine learning and in-depth coverage of new and advanced methods in machine learning, as well as their underlying theory. It emphasizes approaches with practical relevance and discusses a number of recent applications of machine learning in areas like information retrieval, recommender systems, data mining, computer vision, natural language processing and robotics. An open research project is a major part of the course.

In particular, the course will cover the following topics:

- **Supervised Batch Learning**: model, decision theoretic foundation, model selection, model assessment, empirical risk minimization
- **Decision Trees**: TDIDT, attribute selection, pruning and overfitting
- **Statistical Learning Theory**: generalization error bounds, VC dimension
- **Large-Margin Methods and Kernels**: linear Rules, margin, Perceptron, SVMs, duality, non-linear rules through kernels
- **Deep Learning**: multi-layer perceptrons, deep networks, stochastic gradient
- **Probabilistic Models**: generative vs. discriminative, maximum likelihood, Bayesian inference
- **Structured Output Prediction**: undirected graphical models, structural SVMs, conditional random fields
- **Latent Variable Models**: k-means clustering, mixture of Gaussians, expectation-maximization algorithm, matrix factorization, embeddings
- **Online Learning**: experts, bandits, online convex optimization
- **Causal Inference**: interventional vs. observational data, treatment effect estimation

---

## Schedule

Introduction [slides](./01-intro.pdf)
- What is learning?
- What is machine learning used for?
- Overview of course, course policies, and contact info.

Supervised Batch Learning [slides](./02-supervised.pdf)
- Supervised learning for binary classification, multi-class classification, regression, and stuctured output prediction
- Instance space, target function, training examples
- Hypothesis space, consistency, and version space
- Classifying with a decision tree
- Representational power of decision trees
- TDIDT decision tree learning algorithm

Empirical Risk Minimization [slides](./03-erm.pdf)
- Independently identically distributed (iid) data
- Risk, empirical risk, and Empirical Risk Minimization (ERM)
- Bayes decision rule and Bayes risk
- Model selection and overfitting

Statistical Learning Theory 1: Generalization Error Bounds [slides](./04-statlearntheory1.pdf)
- Model assessment (see slides from last lecture)
- Generalization error bound for finite H and zero error

Statistical Learning Theory 2: Generalization Error Bounds [slides](./04-statlearntheory1.pdf)
- Generalization error bound for finite H and non-zero error
- Probably Approximately Correct (PAC) learning
- Theoretical characterization of overfitting

Linear Rules and Perceptron [slides](./05-perceptron.pdf)
- Linear classification rules
- Batch Perceptron learning algorithm
- Online Perceptron learning algorithm
- Margin of linear classifier
- Perceptron mistake bound

Optimal Hyperplanes and SVMs [slides](./06-svm_opthyp.pdf)
- Optimal hyperplanes and margins
- Hard-margin Support Vector Machine
- Vapnik-Chervonenkis Dimension
- VC dimension of hyperplanes
- Symmetrization and error bound

Soft-Margin and Duality [slides](./07-svm_duality.pdf)
- Soft-margin Support Vector Machine
- Dual Perceptron
- Primal and dual SVM optimization problem
- Connection between primal and dual

Kernels [slides](./08-kernels.pdf)
- Bounds on the leave-one-out error of an SVM (see slides from last lecture)
- Input space and feature space
- Kernels as a method for non-linear learning
- Kernels for learning with non-vectorial data

Regularized Linear Models [slides](./09-reglin.pdf)
- Conditional maximum likelihood
- Maximum a posteriori estimates
- Logistic regression
- Ridge regression

Deep Network Models [slides](./10-deep.pdf)
- Multi-layer perceptrons
- Relationship to kernels
- Stochastic gradient descent
- Convolution and pooling

Generative Models for Classification [slides](./11-genclass.pdf)
- Modeling the data generating process
- Multivariate naive Bayes classifier
- Linear discriminant analysis
- Multinomial naive Bayes classifier

Generative Models for Sequence Prediction [slides](./12-genstruct.pdf)
- Hidden Markov Model (HMM)
- Estimation of HMM parameters
- Viterbi algorithm

Discriminative Training for Structured Output Prediction [slides](./13-discstruct.pdf)
- Structural SVMs
- Conditional Random Fields
- Encoder/Decoder Networks

Online Learning: Expert Setting [slides](./14-expert.pdf)
- Expert online learning model
- Halving algorithm and optimal mistake bound
- Regret
- Weighted Majority algorithm

Online Learning: Bandit Setting [slides](./15-bandit.pdf)
- Exponentiated Gradient algorithm (see slides from last lecture)
- Adversarial Bandit model
- EXP3 algorithm
- Stochastic Bandit model
- UCB1 algorithm

Clustering [slides](./16-clustering.pdf)
- Unsupervised learning tasks
- Hierarchical agglomerative clustering
- k-Means

Latent Variable Models [slides](./17-latent.pdf)
- Mixture of Gausssians
- Expectation-maximization (EM) algorithm
- Derivation of EM
- Latent variable problems beyond mixtures

Recommender Systems [slides](./18-recommender.pdf)
- Uses of recommender systems
- Content vs. collaborative recommendation
- Low-rank matrix completion
- Selection biases in preference data

Counterfactual Evaluation and Learning [slides](./19-counterfactualml1.pdf)
- Contextual-bandit feedback
- Policy and utility
- Potential outcomes model
- On-policy vs. off-policy estimators

Counterfactual Evaluation and Learning [slides](./20-counterfactualml2.pdf)
- Counterfactual Risk Minimization
- POEM
- BanditNet

Learning to Rank [slides](./21-ltr.pdf)
- Behavior and ranking metrics
- LTR with expert labels
- LTR with implicit feedback
- Connection to counterfactual evaluation and learning

Wrap-Up [slides](./22-wrapup.pdf)
- Main themes of class
- What we did not cover
- Follow-up courses