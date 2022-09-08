---
layout: page
permalink: /talks/
title: talks
description: All confirmed talks for BFO-2022. Click on a title to expand the abstract.
nav: true
nav_order: 2
---

<details>
  <summary> <b>Amir Gholami</b>: <i>A Fast, Fisher Based Pruning of Transformers without Retraining.</i> </summary>
  
Pruning is an effective way to reduce the huge inference cost of large Transformer models. However, prior work on model pruning requires retraining the model. This can add high cost and complexity to model deployment, making it difficult to use in many practical situations. To address this, we propose a fast post- training pruning framework for Transformers that does not require any retraining. Given a resource constraint and a sample dataset, our framework automatically prunes the Transformer model using structured sparsity methods. To retain high accuracy without retraining, we introduce three novel techniques: (i) a lightweight mask search algorithm that finds which heads and filters to prune based on the Fisher information; (ii) mask rearrangement that complements the search algorithm; and (iii) mask tuning that reconstructs the output activations for each layer. We apply our method to BERT-BASE and DistilBERT, and we evaluate its effectiveness on GLUE and SQuAD benchmarks. Our framework achieves up to 2.0x reduction in FLOPs and 1.56x speedup in inference latency, while maintaining < 1\% loss in accuracy. Importantly, our framework prunes Transformers in less than 3 minutes on a single GPU, which is over two orders of magnitude faster than existing pruning approaches that retrain. 
</details>


<details>
  <summary> <b>Coralia Cartis</b>: <i>Tensor Methods for Nonconvex Optimization.</i> </summary>
  
We consider the advantages of having and incorporating higher- (than second-) order derivative information inside regularization frameworks, generating higher-order regularization algorithms that have better complexity, universal properties and can certify higher-order criticality of candidate solutions. Time permitting, we also discuss inexact settings where problem information and smoothness assumptions are weakened, without affecting the algorithms’ complexity. Efficient solution of some higher-order polynomial subproblems will also be discussed. 
</details>

<details>
  <summary> <b>Frank E. Curtis</b>: <i>Deterministically Constrained Stochastic Optimization.</i> </summary>
  
This talk highlights the recent work by my research group on the design, analysis, and implementation of algorithms for solving continuous nonlinear optimization problems that involve a stochastic objective function and deterministic constraints.  We will focus on our sequential quadratic optimization (commonly known as SQP) methods for cases when the constraints are defined by nonlinear systems of equations and inequalities.  Our methods are applicable for solving various types of problems, such as for training machine learning (e.g., deep learning) models with constraints.  Our work focuses on the "fully stochastic" regime in which only stochastic gradient estimates are employed, for which we have derived convergence-in-expectation results and worst-case iteration complexity bounds that are on par with stochastic gradient methods for the unconstrained setting. We will also discuss the various extensions that my group is exploring.
</details>

<details>
  <summary> <b>Madeleine Udell</b>: <i>Low Rank Approximation for Faster Convex Optimization.</i> </summary>
  
Low rank structure is pervasive in real-world datasets. This talk shows how to accelerate the solution of fundamental computational problems, including eigenvalue decomposition, linear system solves, and composite convex optimization,by exploiting this low rank structure. We present a simple and efficient method for approximate top eigendecomposition based on randomized numerical linear algebra. Armed with this primitive, we design a new randomized preconditioner for the conjugate gradient method, and a method called NysADMM, based on the inexact alternating directions method of multipliers, for composite convex optimization. These methods come with strong theoretical and numerical support. Indeed, a simple implementation of NysADMM solves important large-scale statistical problems like lasso, logistic regression, and support vector machines 2--58x faster than standard solvers.
</details>
  
## Schedule
| Allocated Time | Agenda Item                                           |
|----------------|-------------------------------------------------------|
| 08.30 - 08.45  | Welcome and opening remarks                           |
| 08.45 - 09.30  | Plenary Talk from  Donald Goldfarb                    |
| 09.30 - 10.15  | Plenary Talk from Coralia Cartis                      |
| 10.15 - 11.00  | Spotlight Talks #1                                    |
| 11.00 - 11.30  | Coffee Break                                          |
| 11.00 - 12.00  | Poster Session  #1                                    |
| 12.00 - 12.45  | Spotlight talks #2                                    |
| 12.45 - 14.15  | Lunch Break                                           |
| 14.15 -  15.00 | Plenary Talk from Frank E. Curtis                     |
| 15.00 - 15.45  | Plenary Talk from Madeleine Udell                     |
| 15.45 - 16.15  | Coffee Break                                          |
| 15.45 - 16.30  | Poster Session #2                                     |
| 16.30 - 17.15  | Plenary talk from  Amir Gholami                       |
| 17.15 - 18.00  | Junior researcher poster session                      |
| 18.00 - 18.15  | Plenary ending discussion and closing remarks         |
