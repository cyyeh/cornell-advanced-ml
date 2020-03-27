# CS4780 - Machine Learning for Intelligent Systems

course website: https://www.cs.cornell.edu/courses/cs4780/2019fa/

## Overview

Machine learning is concerned with the question of how to make computers learn from experience. The ability to learn is not only central to most aspects of intelligent behavior, but machine learning techniques have become key components of many software systems. For examples, machine learning techniques are used to build search engines, to recommend movies, to understand natural language and images, and to build autonomous robots. This course will introduce the fundamental set of techniques and algorithms that constitute supervised machine learning as of today. The course will not only discuss individual algorithms and methods, but also tie principles and approaches together from a theoretical perspective. In particular, the course will cover the following topics:

- **Supervised Batch Learning**: model, decision theoretic foundation, model selection, model assessment, empirical risk minimization
- **Instance-based Learning**: K-Nearest Neighbors, collaborative filtering
- **Decision Trees**: TDIDT, attribute selection, pruning and overfitting
- **Linear Rules**: Perceptron, logistic regression, linear regression, duality
- **Support Vector Machines**: Optimal hyperplane, margin, kernels, stability
- **Deep Learning**: multi-layer perceptrons, deep networks, stochastic gradient
- **Generative Models**: generative vs. discriminative, naive Bayes, linear discriminant analysis
- **Structured Output Prediction**: predicting sequences, hidden markov model, rankings
- **Statistical Learning Theory**: generalization error bounds, VC dimension
- **Online Learning**: experts, bandits, online mistake bounds

## Schedule

Introduction [slides](./01-intro.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/56169d176dda4e7d84611d0d838708301d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 1
- What is learning?
- What is machine learning used for?
- Overview of course, course policies, and contact info.

Instance-Based Learning [slides](./02-knn.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/833d9df3e0c042039c0b11dd99a2d7601d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 19.1, 19.3
- Definition of binary classification, instance space, target function, training examples.
- Unweighted k-nearest neighbor (kNN) rule.
- Weighted kNN.
- Effect of selecting k.
- Supervised learning for binary classification, multi-class classification, regression, and stuctured output prediction.
- kNN for regression and collaborative filtering.

Supervised Learning and Decision Trees [slides](./03-DTs.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/92e2b307ebac4040aab096f89b8bf2c01d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Hypothesis space, consistency, and version space
- List-then-eliminate algorithm
- Classifying with a decision tree
- Representational power of decision trees
- TDIDT decision tree learning algorithm
- Splitting criteria for TDIDT learning

Prediction and Overfitting [slides](./04-overfit.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/e6abe1476d3f46418b464ef3f68ca5921d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 2.1-2.2, 18.2
- Training error, Test error, prediction error
- Independently identically distributed (i.i.d) data
- Overfitting
- Occam's Razor

Model Selection and Assessment [slides](./05-modsel.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/f08f496428b14f05a69e856fc83fc6fa1d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 11 (w/o 11.1) and McNemar's Test ([ref1](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)) and ([ref2](https://en.wikipedia.org/wiki/McNemar%27s_test))
- Model selection
- Controlling overfitting in decision trees
- Train/validate/test split and k-fold cross-validation
- Statistical tests for assessing learning results

Linear Classifiers and Perceptrons [slides](./06-perceptron.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/152fcb9ca824477e969e4133bf4011951d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 9-9.1 (w/o 9.1.3)
- Linear classification rules
- Linear programming for linear classification
- (Batch) Perceptron learning algorithm

Convergence of Perceptron [slides](./07-mistakebound.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/c60227b1210f47bdb33eb2251b2c6e0f1d?
- Reading: UML 9.1.2
- Margin of linear classifiers
- Convergence of Perceptron
- Online Mistake Bound Learning

Optimal Hyperplanes and SVMs [slides](./08-svm_opthyp.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/d7fff623c4404390a51974edf5cbf5941d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 15.1-15.2
- Optimal hyperplanes and margins
- Hard-margin Support Vector Machine
- Soft-margin Support Vector Machine
- Primal SVM optimization problem

Margin, Duality and Stability [slides](./09-svm_duality.pdf)/[video]((https://cornell.mediasite.com/Mediasite/Play/b0afc1ce44a948138a579585ec3070921d?catalog=467d80413f01480ab0be1b1722e68bf221))
- Reading: UML 15.3-15.4
- Dual Perceptron
- Dual SVM optimization problem
- Connections between dual and primal
- Bounds on the leave-one-out error of SVMs

Kernels [slides](./10-svm_kernels.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/5ed6b4946bd2409080b6d53f086eec551d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 16.1-16.2
- Input space and feature space
- Kernels as a method for non-linear learning
- Kernels for learning with non-vectorial data

Regularized Linear Models [slides](./11-reglin.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/ac2944ff38b448e2965d8e82df9fe0f01d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 13.1, 9.2, 9.3
- Conditional maximum likelihood
- Logistic regression
- Ridge regression

Stochastic Gradient Descent [slides](./12-sgd.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/2899233f70ed4b1d8bb9e39a6baec6fa1d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 14
- Overview of convexity and gradients
- Gradient Descent
- Stochastic Gradient Descent

Deep Neural Networks [slides](./13-nn.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/b8dbeb72ed9d484f814dece8a3c103101d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 20-20.3
- Demos: [Step sizes](http://fa.bianp.net/teaching/2018/eecs227at/) and [Tensorflow Playground](http://playground.tensorflow.org/)
- AdaGrad and Momentum for SGD
- Feedforward Neural Networks
- Activation Functions
- Expressive power of neural networks

Backpropagation in Neural Networks [slides](./14-backprop.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/8e71228a7c1f4617b184415f804aeb841d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 20.6
- SGD in Multi-layer Networks
- Backpropagations

Review Session [video](https://cornell.mediasite.com/Mediasite/Play/22c323d93ef744a388adfe2d2145dc0f1d?catalog=467d80413f01480ab0be1b1722e68bf221)
Review of material covered so far

Statistical Learning Theory I [slides](./17-slt1.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/bfe63b38ed044e4090088067a41190041d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 3-4
- Psychic Game
- Generalization error bound for finite H and zero training error
- Generalization error bound for finite H and non-zero training error
- Sample Complexity for finite H.

Statistical Learning Theory II [slides](./18-slt2.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/fb17d38b765545c386702dcc6db14dd01d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 4 and 6
- Generalization error bound for finite H and non-zero training error
- VC Dimension and Growth Function.
- Sample Complexity for infinite H.

Statistical Learning Theory III [slides](./19-slt3.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/2c80ec11019c43bdafb9571f9c9010591d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 6
- VC dimension examples
- PAC Learning.

Intelligibility in Machine Learning [video](https://cornell.mediasite.com/Mediasite/Play/96a6fc27c683406e8785f182101ef1161d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Guest lecture by Rich Caruana

Generative Models for Classification [slides](./20-genclass.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/f32800a399b94b9d84ef902699af0b481d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 24.2, 24.3
- Generative modeling vs. discriminative training
- Multivariate Naive Bayes
- Multinominal Naive Bayes
- Linear Discriminant Analysis

Structured Output Prediction: Generative Models [slides](./21-genstruct.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/a28be6056d8944c49339e4bb98c683151d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: Murphy 17.3, 17.4.4, 17.5.1
- Hidden Markov Model (HMM)
- Estimation of HMM parameters
- Viterbi algorithm

Structured Output Prediction: Discriminative Training [slides](./22-discstruct.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/f890982ce53c43cb8db6d7f9897ce73b1d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: Murphy 19.7
- Structural SVMs
- Conditional Random Fields
- Encoder/Decoder Networks

Online Learning [slides](./23-online.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/b13297401c7b435a867e84f5bed279e51d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 21-21.2.1 and optional reading
- Online Learning Model
- Halving Algorithm
- (Deterministic) Weighted Majority Algorithm

Boosting [slides](./24-boosting.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/34096a868dfb47b2a9c21ac4be14438f1d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: UML 10-10.3
- Optional readings: [Schapire 2001 Survey](./boosting-survey.pdf) and [Schapire's Tutorial](./boosting-tutorial.pdf)
- Strong and Weak Learning
- Adaboost
- Boosting and Regret Minimization
- Ensemble Methods and Bagging.

Learning to Act and Causality [slides](./25-causalml.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/8b7c70efc4354881b283a685a5d859c41d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: Imbens/Rubin Chapter 1
- Contextual-Bandit Feedback
- Policy Learning
- Potential Outcome Model
- Model-the-World vs. Model-the-Bias

Privacy and Fairness [slides](./26-privacy-fairness.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/01da5e08e32445319d8f9e3b0498db351d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: Dwork and Roth Chapters 1-2
- Differential Privacy
- Fairness Tradeoffs

Fairness in Ranking and Wrap-up [slides](./27-fairrank.pdf)/[video](https://cornell.mediasite.com/Mediasite/Play/2ae9a6c47f9343e89a7ae1543b69ba441d?catalog=467d80413f01480ab0be1b1722e68bf221)
- Reading: none
- Fairness of exposure
- Endogenous vs. exogenous bias
- Review of main class themes
- Pointers to followup classes


## Reading Materials

- [Understanding Machine Learning - From Theory to Algorithms](./understanding-machine-learning-theory-algorithms.pdf)
- [The Algorithmic Foundations of Differential Privacy](./privacybook.pdf)