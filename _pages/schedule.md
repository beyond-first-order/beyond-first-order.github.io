---
layout: page
permalink: /talks/
title: schedule
description: All confirmed talks for HOO-2022. Click on a title to expand the abstract.
nav: true
nav_order: 2
---
  
## **Workshop Schedule**

Note: All times are in <b>Central Standard Time</b>
  
| **Allocated Time** | **Agenda Item**                               |
|--------------------|-----------------------------------------------|
| 08.15 - 08.30      | Welcome and opening remarks                   |
| 08.30 - 09.15      | Plenary Talk from  Donald Goldfarb            |
| 09.15 - 10.00      | Plenary Talk from Coralia Cartis              |
| 10.00 - 10.30      | Coffee Break                                  |
| 10.00 - 10.30      | Poster Session  \#1                           |
| 10.30 - 10.45      | Spotlight Talk from Wenqi Zhu                 |
| 10.45 - 11.00      | Spotlight Talk from  Farshed Abdukhakimov     |
| 11.00 - 11.15      | Spotlight Talk from Chuwen Zhang              |
| 11.15 - 11.30      | Spotlight Talk from Zachary Novack            |
| 11.30 - 11.45      | Spotlight Talk from Klea Ziu                  |
| 11.45 - 12.00      | Spotlight Talk from Kabir A Chandrasekher     |
| 12.00 - 13.30      | Lunch Break                                   |
| 13.30 - 14.15      | Plenary Talk from Frank E. Curtis             |
| 14.15 - 15.00      | Plenary Talk from Madeleine Udell             |
| 15.00 - 15.30      | Coffee Break                                  |
| 15.00 - 16.00      | Poster Session \#2                            |
| 16.00 - 16.45      | Plenary talk from  Amir Gholami               |
| 16.45 - 17.00      | Plenary ending discussion and closing remarks |

<br>
## Plenary Talks



<details>
  <summary> <b>Coralia Cartis</b>: <i>Tensor Methods for Nonconvex Optimization.</i> </summary>
  
We consider the advantages of having and incorporating higher- (than second-) order derivative information inside regularization frameworks, generating higher-order regularization algorithms that have better complexity, universal properties and can certify higher-order criticality of candidate solutions. Time permitting, we also discuss inexact settings where problem information and smoothness assumptions are weakened, without affecting the algorithms’ complexity. Efficient solution of some higher-order polynomial subproblems will also be discussed. 
</details>


<details>
  <summary> <b>Frank E. Curtis</b>: <i>Deterministically Constrained Stochastic Optimization.</i> </summary>
  
This talk highlights the recent work by my research group on the design, analysis, and implementation of algorithms for solving continuous nonlinear optimization problems that involve a stochastic objective function and deterministic constraints.  We will focus on our sequential quadratic optimization (commonly known as SQP) methods for cases when the constraints are defined by nonlinear systems of equations and inequalities.  Our methods are applicable for solving various types of problems, such as for training machine learning (e.g., deep learning) models with constraints.  Our work focuses on the "fully stochastic" regime in which only stochastic gradient estimates are employed, for which we have derived convergence-in-expectation results and worst-case iteration complexity bounds that are on par with stochastic gradient methods for the unconstrained setting. We will also discuss the various extensions that my group is exploring.
</details>



<details>
  <summary> <b>Amir Gholami</b>: <i>A Fast, Fisher Based Pruning of Transformers without Retraining.</i> </summary>
  
Pruning is an effective way to reduce the huge inference cost of large Transformer models. However, prior work on model pruning requires retraining the model. This can add high cost and complexity to model deployment, making it difficult to use in many practical situations. To address this, we propose a fast post- training pruning framework for Transformers that does not require any retraining. Given a resource constraint and a sample dataset, our framework automatically prunes the Transformer model using structured sparsity methods. To retain high accuracy without retraining, we introduce three novel techniques: (i) a lightweight mask search algorithm that finds which heads and filters to prune based on the Fisher information; (ii) mask rearrangement that complements the search algorithm; and (iii) mask tuning that reconstructs the output activations for each layer. We apply our method to BERT-BASE and DistilBERT, and we evaluate its effectiveness on GLUE and SQuAD benchmarks. Our framework achieves up to 2.0x reduction in FLOPs and 1.56x speedup in inference latency, while maintaining < 1\% loss in accuracy. Importantly, our framework prunes Transformers in less than 3 minutes on a single GPU, which is over two orders of magnitude faster than existing pruning approaches that retrain. 
</details>





  
<details>
  <summary> <b>Donald Goldfarb</b>: <i>Efficient Second-Order Stochastic Methods for Machine Learning.</i> </summary>
  
Our talk focuses on training Deep Neural Networks (DNNs), which due to the enormous number of parameters current DNNs have, using the Hessian matrix or a full approximation to it in a second-order method is prohibitive, both in terms of memory requirements and computational cost per iteration. Hence, to be practical, layer-wise block-diagonal approximations to these matrices are usually used.  Here we describe second-order quasi-Newton (QN), natural gradient (NG), and generalized Gauss-Newton (GGN) methods of this type that are competitive with and often outperform first-order methods. These methods include those that use layer-wise (i) Kronecker-factored BFGS and L-BFGS QN approximations, (ii) tensor normal covariance and (iii) mini-block Fisher matrix approximations, and (iv) Sherman-Morrison-Woodbury based variants of NG and GGN methods.
</details>


<details>
  <summary> <b>Madeleine Udell</b>: <i>Low Rank Approximation for Faster Convex Optimization.</i> </summary>
  
Low rank structure is pervasive in real-world datasets. This talk shows how to accelerate the solution of fundamental computational problems, including eigenvalue decomposition, linear system solves, and composite convex optimization,by exploiting this low rank structure. We present a simple and efficient method for approximate top eigendecomposition based on randomized numerical linear algebra. Armed with this primitive, we design a new randomized preconditioner for the conjugate gradient method, and a method called NysADMM, based on the inexact alternating directions method of multipliers, for composite convex optimization. These methods come with strong theoretical and numerical support. Indeed, a simple implementation of NysADMM solves important large-scale statistical problems like lasso, logistic regression, and support vector machines 2--58x faster than standard solvers.
</details>
  
<br>
## Spotlight Talks
  
<details>
	<summary> <b> Farshed Abdukhakimov, Chulu Xiang, Dmitry Kamzolov, Robert M Gower, Martin Takac </b>: <i> PSPS: Preconditioned Stochastic Polyak Step-size method for badly scaled data </i> </summary>

<br>
The family of Stochastic Gradient Methods with Polyak Step-size offers an update rule that alleviates the need of fine-tuning the learning rate of an optimizer.  Recent work (Robert M Gower, Mathieu Blondel, Nidham Gazagnadou, and Fabian Pedregosa: Cutting some slack for SGD with adaptive polyak stepsizes) has been proposed to introduce a slack variable, which makes these methods applicable outside of the interpolation regime. In this paper, we combine  preconditioning and slack in an updated optimization algorithm to show its performance on badly scaled and/or ill-conditioned datasets. We use Hutchinson's method to obtain an estimate of a Hessian which is used as the preconditioner.
<br>
</details>  
  
<details>
	<summary> <b> Dmitry Kamzolov, Klea Ziu, Artem Agafonov, Martin Takac </b>: <i> Cubic Regularized Quasi-Newton Methods </i> </summary>

<br>
In this paper, we propose a Cubic Regularized L-BFGS. Cubic Regularized Newton outperforms the classical Newton method in terms of global performance. In classics, L-BFGS approximation is applied for the Newton method. We propose a new variant of inexact Cubic Regularized Newton. Then, we use L-BFGS approximation as an inexact Hessian for Cubic Regularized Newton. It allows us to get better theoretical convergence rates and good practical performance, especially from the points where classical Newton is diverging. 
<br>
</details>
  
<details>
	<summary> <b> Mengqui Lou, Kabir A Chandrasekher, Ashwin Pananjady </b>: <i> Alternating minimization for generalized rank one matrix sensing: Sharp predictions from a random initialization </i> </summary>

<br>
We consider the problem of estimating the factors of a rank-1 matrix with i.i.d. Gaussian, rank-1 measurements that are nonlinearly transformed and corrupted by noise. Considering two prototypical choices for the nonlinearity, we study the convergence properties of a natural alternating update rule for this nonconvex optimization problem starting from a random initialization. We show sharp linear convergence guarantees for a sample-split version of the algorithm by deriving a deterministic recursion that is accurate even in high-dimensional problems. Our sharp, non-asymptotic analysis also exposes several other fine-grained properties of this problem, including how the nonlinearity, sample size, and noise level affect convergence behavior. Our results are enabled by showing that the empirical error recursion can be predicted by our deterministic sequence within fluctuations of the order n^{-1/2} when each iteration is run with n observations. Our technique leverages leave-one-out tools and provides an avenue for sharply analyzing higher-order iterative algorithms from a random initialization in other optimization problems with random data.
<br>
</details>  
 
<details>
	<summary> <b> Zachary Novack, Simran Kaur, Tanya Marwah, Saurabh Garg, Zachary Lipton </b>: <i> Disentangling the Mechanisms Behind Implicit Regularization in SGD </i> </summary>

<br>
A number of competing hypotheses have been proposed to explain why small-batch Stochastic Gradient Descent (SGD) leads to improved generalization over the full-batch regime, with recent work crediting the implicit regularization of various quantities throughout training. However, to date, empirical evidence assessing the explanatory power of these hypotheses is lacking. In this paper, we conduct an extensive empirical evaluation, focusing on the ability of various theorized mechanisms to close the small-to-large batch generalization gap. Additionally, we characterize how the quantities that SGD has been claimed to (implicitly) regularize change over the course of training. By using micro-batches, i.e. disjoint smaller subsets of each mini-batch, we empirically show that explicitly penalizing the gradient norm or the Fisher Information Matrix trace, averaged over micro-batches, in the large-batch regime recovers small-batch SGD generalization, whereas Jacobian-based regularizations fail to do so. This generalization performance is shown to often be correlated with how well the regularized model’s gradient norms resemble those of small-batch SGD. We additionally show that this behavior breaks down as the micro-batch size approaches the batch size. Finally, we note that in this line of inquiry, positive experimental findings on CIFAR10 are often reversed on other datasets like CIFAR100, highlighting the need to test hypotheses on a wider collection of datasets.
<br>
</details>  
  
  
  
<details>
	<summary> <b> Chuwen Zhang, Jiang Bo, Chang He, Yuntian Jiang, Dongdong Ge, Yinyu Ye </b>: <i> DRSOM: A Dimension Reduced Second-Order Method </i> </summary>

<br>
In this paper, we propose a Dimension-Reduced Second-Order Method (DRSOM) for convex and nonconvex (unconstrained) optimization. Under a trust-region-like framework, our method preserves the convergence of the second-order method while using only Hessianvector products in a few directions, which enables the computational overhead of our method remain comparable to the first-order such as the gradient descent method. Theoretically, we show that the method has a local quadratic convergence and a global convergence rate of O(ϵ −3/2 ) to satisfy the first-order and second-order conditions under a commonly adopted approximated Hessian assumption. We further show that this assumption can be removed if we perform a step of the Lanczos method periodically at the end-stage of the algorithm. The applicability and performance of DRSOM are exhibited by various computational experiments, particularly in machine learning and deep learning. For neural networks, our preliminary implementation seems to gain computational advantages in terms of training accuracy and iteration complexity over state-of-the-art first-order methods such as SGD and ADAM.
<br>
</details>
  
<details>
	<summary> <b> Wenqi Zhu, Coralia Cartis </b>: <i> Quartic Polynomial Sub-problem Solutions in Tensor Methods for Nonconvex Optimization </i> </summary>

<br>
There has been growing interest in high-order tensor methods for nonconvex optimization in machine learning as these methods provide better/optimal worst-case evaluation complexity, stability to parameter tuning, and robustness to problem conditioning. The well-known $p$th-order adaptive regularization (AR$p$) method relies crucially on repeatedly minimising a nonconvex multivariate Taylor-based polynomial sub-problem. It remains an open question to find efficient techniques to minimise such a sub-problem for $p\ge3$.  In this paper, we propose a second-order method (SQO) for the AR$3$ (AR$p$ with $p=3$) sub-problem. SQO approximates the special-structure quartic polynomial sub-problem from above and below by using second-order models that can be minimised efficiently and globally. We prove that SQO finds a local minimiser of a quartic polynomial, but in practice, due to its construction, it can find a much lower minimum than cubic regularization approaches. This encourages us to continue our quest for algorithmic techniques that find approximately global solutions for such polynomials.
<br>
</details>
<br>

  
