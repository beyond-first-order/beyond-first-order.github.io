---
layout: page
permalink: /papers/
title: accepted papers
description: All accepted papers for HOO-2022. Click on a title to expand the abstract. 
nav: true
nav_order: 3
---
<details>
	<summary> <b> Farshed Abdukhakimov, Chulu Xiang, Dmitry Kamzolov, Robert M Gower, Martin Takac </b>: <i> PSPS: Preconditioned Stochastic Polyak Step-size method for badly scaled data </i> </summary>

<br>
The family of Stochastic Gradient Methods with Polyak Step-size offers an update rule that alleviates the need of fine-tuning the learning rate of an optimizer.  Recent work (Robert M Gower, Mathieu Blondel, Nidham Gazagnadou, and Fabian Pedregosa: Cutting some slack for SGD with adaptive polyak stepsizes) has been proposed to introduce a slack variable, which makes these methods applicable outside of the interpolation regime. In this paper, we combine  preconditioning and slack in an updated optimization algorithm to show its performance on badly scaled and/or ill-conditioned datasets. We use Hutchinson's method to obtain an estimate of a Hessian which is used as the preconditioner.
<br>
</details>
<br>
<details>
	<summary> <b> Artem Agafonov, Brahim Erraji, Martin Takac </b>: <i> FLECS-CGD: A Federated Learning Second-Order Framework via Compression and Sketching with Compressed Gradient Differences </i> </summary>

<br>
In the recent paper FLECS (Agafonov et al,  FLECS: A Federated Learning Second-Order Framework via Compression and Sketching), the second-order framework FLECS was proposed for the Federated Learning problem. This method utilize compression of sketched Hessians to make communication costs low. However, the main bottleneck of FLECS is gradient communication without compression. In this paper, we propose the modification of FLECS with compressed gradient differences, which we call FLECS-CGD (FLECS with Compressed Gradient Differences) and make it applicable for stochastic optimization. Convergence guarantees are provided in strongly convex and nonconvex cases. Experiments show the practical benefit of proposed approach.
<br>
</details>
<br>
<details>
	<summary> <b> Coralia Cartis, Zhen Shao </b>: <i> Random-subspace adaptive cubic regularisation method for nonconvex optimisation </i> </summary>

<br>
We investigate second-order methods for nonconvex optimisation, and propose a Random Subspace Adaptive Cubic Regularisation (R-ARC) method, which we analyse under various assumptions on the objective function and the sketching matrices that generate the random subspaces. We show that, when the sketching matrix achieves a subspace embedding of the augmented matrix of the gradient and the Hessian with sufficiently high probability, then the R-ARC method satisfies, with high probability, a complexity bound of order O(ε^{-3/2}) to drive the (full) gradient norm below ε; matching in the accuracy order its deterministic counterpart (ARC). As an illustration, we particularise our results to the special case of a scaled Gaussian ensemble. 
<br>
</details>
<br>
<details>
	<summary> <b> Mohamed Elsayed, Rupam Mahmood </b>: <i> HesScale: Scalable Computation of Hessian Diagonals </i> </summary>

<br>
Second-order optimization uses curvature information about the objective function, which can help in faster convergence. However, such methods typically require expensive computation of the Hessian matrix, preventing their usage in a scalable way. The absence of efficient ways of computation drove the most widely used methods to focus on first-order approximations that do not capture the curvature information. In this paper, we develop HesScale, a scalable approach to approximating the diagonal of the Hessian matrix, to incorporate second-order information in a computationally efficient manner. We show that HesScale has the same computational complexity as backpropagation. Our results on supervised classification show that HesScale achieves high approximation accuracy, allowing for scalable and efficient second-order optimization.
<br>
</details>
<br>




<details>
	<summary> <b> Yuchen Fang, Sen Na, Mladen Kolar </b>: <i> Trust-Region Sequential Quadratic Programming for Stochastic Optimization with Random Models: First-Order Stationarity </i> </summary>

<br>
We consider optimization problems with a stochastic objective and deterministic constraints, and design a trust-region sequential quadratic programming (TR-SQP) method to solve them. We name our method TR-SQP for STochastic Optimization with Random Models (TR-SQP-STORM). In each iteration, our algorithm constructs a random model for the objective that satisfies suitable accuracy conditions with a high but fixed probability. The algorithm decides whether a trial step is successful or not based on two ratios: the ratio between the estimated actual reduction and predicted reduction on the $\ell_2$ merit function, and the ratio between the estimated KKT residual and trust-region radius. For each successful step, the algorithm increases the trust-region radius, and further decides whether the step is reliable or not based on the amount of the predicted reduction. If the step is reliable, then the algorithm relaxes the accuracy conditions for the next iteration. To resolve the infeasibility issue of trust-region methods for constrained problems, we employ an adaptive relaxation technique proposed by a companion paper. Under reasonable assumptions, we establish the global first-order convergence guarantee: the KKT residual converges to zero almost surely. We apply our method on a subset of problems in CUTEst set to demonstrate its empirical performance.
<br>
</details>
<br>

<details>
	<summary> <b> Yuchen Fang, Sen Na, Mladen Kolar </b>: <i> Fully Stochastic Trust-Region Sequential Quadratic Programming for Equality-Constrained Optimization Problems </i> </summary>

<br>
We propose a fully stochastic trust-region sequential quadratic programming (TR-StoSQP) algorithm to solve nonlinear optimization problems. The problems involve a stochastic objective and deterministic equality constraints. Under the fully stochastic setup, we suppose that only a single sample is generated in each iteration to estimate the objective gradient. Compared to the existing line-search StoSQP schemes, our algorithm allows one to employ indefinite Hessian matrices for SQP subproblems. The algorithm adaptively selects the radius of the trust region based on an input sequence $\{\beta_k\}$, the estimated KKT residual, and the estimated Lipschitz constants of the objective gradients and constraint Jacobians. To address the infeasibility issue of trust-region methods that arises in constrained optimization, we propose an adaptive relaxation technique to compute the trial step. In particular, we decompose the trial step into a normal step and a tangential step. Based on the ratios of the feasibility and optimality residuals to the full KKT residual, we decompose the full trust-region radius into two segments that are used to control the size of the normal and tangential steps, respectively. The normal step has a closed form, while the tangential step is solved from a trust-region subproblem, of which the Cauchy point is sufficient for our study. We establish the global almost sure convergence guarantee of TR-StoSQP, and demonstrate its empirical performance on a subset of problems in CUTEst test set.
<br>
</details>
<br>


 



<details>
	<summary> <b> Rustem Islamov, Xun Qian, Slavomír Hanzely, Mher Safaryan, Peter Richtarik </b>: <i> Distributed Newton-Type Methods with Communication Compression and Bernoulli Aggregation </i> </summary>

<br>
Despite their high computation and communication costs, Newton-type methods remain an appealing option for distributed training due to their robustness against ill-conditioned convex problems. In this work, we study {\em communication compression} and {\em aggregation mechanisms} for curvature information in order to reduce these costs while preserving theoretically superior local convergence guarantees. We prove that the recently developed class of {\em three point compressors (3PC)} of Richtárik et al. [2022] for gradient communication can be generalized to Hessian communication as well. This result opens up a wide variety of communication strategies, such as {\em contractive compression} and {\em lazy aggregation}, available to our disposal to compress prohibitively costly curvature information. Moreover, we discovered several new 3PC mechanisms, such as {\em adaptive thresholding} and {\em Bernoulli aggregation}, which require reduced communication and occasional Hessian computations. Furthermore, we extend and analyze our approach to bidirectional communication compression and partial device participation setups to cater to the practical considerations of applications in federated learning. For all our methods, we derive fast {\em condition-number-independent} local linear and/or superlinear convergence rates. Finally, with extensive numerical evaluations on convex optimization problems, we illustrate that our designed schemes achieve state-of-the-art communication complexity compared to several key baselines using second-order information.
<br>
</details>
<br>
<details>
	<summary> <b> Qiujiang Jin, Aryan Mokhtari, Nhat Ho, Tongzheng Ren </b>: <i> Statistical and Computational Complexities of BFGS Quasi-Newton Method for Generalized Linear Models </i> </summary>

<br>
The gradient descent (GD) method has been used widely to solve parameter estimation in generalized linear models (GLMs), a generalization of linear models when the link function can be non-linear. While GD has optimal statistical and computational complexities for estimating the true parameter under the high signal-to-noise ratio (SNR) regime of the GLMs, it has sub-optimal complexities when the SNR is low, namely, the iterates of GD require polynomial number of iterations to reach the final statistical radius. The slow convergence of GD for the low SNR case is mainly due to the local convexity of the least-square loss functions of the GLMs. To address the shortcomings of GD, we propose to use the BFGS quasi-Newton method to solve parameter estimation of the GLMs. On the optimization side, when the SNR is low, we demonstrate that iterates of BFGS converge linearly to the optimal solution of the population least-square loss function. On the statistical side, we prove that the iterates of BFGS reach the final statistical radius of the low SNR GLMs after a logarithmic number of iterations, which is much lower than the polynomial number of iterations of GD. We also present numerical experiments that match our theoretical findings.
<br>
</details>
<br>
<details>
	<summary> <b> Dmitry Kamzolov, Klea Ziu, Artem Agafonov, Martin Takac </b>: <i> Cubic Regularized Quasi-Newton Methods </i> </summary>

<br>
In this paper, we propose a Cubic Regularized L-BFGS. Cubic Regularized Newton outperforms the classical Newton method in terms of global performance. In classics, L-BFGS approximation is applied for the Newton method. We propose a new variant of inexact Cubic Regularized Newton. Then, we use L-BFGS approximation as an inexact Hessian for Cubic Regularized Newton. It allows us to get better theoretical convergence rates and good practical performance, especially from the points where classical Newton is diverging. 
<br>
</details>
<br>
<details>
	<summary> <b> Xilin Li </b>: <i> Black Box Lie Group Preconditioners for SGD </i> </summary>

<br>
A matrix free and a low rank approximation preconditioner are proposed to accelerate the convergence of stochastic gradient descent (SGD) by exploiting curvature information sampled from Hessian-vector products or finite differences of parameters and gradients similar to the BFGS algorithm.     Both preconditioners are fitted with an online updating manner minimizing a criterion that is free of line search and robust to stochastic gradient noise, and further constrained to be on certain connected  Lie groups to preserve their corresponding symmetry or invariance, e.g., orientation of coordinates by the connected general linear group with positive determinants.   The Lie group's equivariance property facilitates preconditioner fitting, and its invariance property saves any need of damping, which is common in second order optimizers, but difficult to tune.  The learning rate for parameter updating and step size for preconditioner fitting  are naturally normalized, and their default values work well in most situations.   
<br>
</details>
<br>
<details>
	<summary> <b> Shuang Li, William J Swartworth, Martin Takac, Deanna Needell, Robert M Gower </b>: <i> Using quadratic equations for overparametrized models </i> </summary>

<br>
Recently the SP (Stochastic Polyak step size) method has emerged as a competitive adaptive method for setting the step sizes of SGD.  SP can be interpreted as a method specialized to interpolated models, since it solves the \emph{interpolation equations}. SP solves these equation by using local linearizations of the model.  We take a step further and develop a method for solving the interpolation equations that uses the local second-order approximation of the model. Our resulting method SP2 uses Hessian-vector products to speed-up the convergence of SP. Furthermore, and rather uniquely among second-order methods, the design of SP2 in no way relies on positive definite Hessian matrices or convexity of the objective function. We show SP2 is very competitive on matrix completion, non-convex test problems and logistic regression. We also provide a convergence theory on sums-of-quadratics. 
<br>
</details>
<br>
<details>
	<summary> <b> Tianyi Lin, Michael Jordan </b>: <i> Perseus: A Simple and Optimal High-Order Method for Variational Inequalities </i> </summary>

<br>
This paper settles an open and challenging question pertaining to the design of simple high-order regularization methods for solving smooth and monotone variational inequalities (VIs). A VI involves finding $x^\star \in \XCal$ such that $\langle F(x), x - x^\star\rangle \geq 0$ for all $x \in \XCal$ and we consider the setting where $F: \br^d \mapsto \br^d$ is smooth with up to $(p-1)^{\textnormal{th}}$-order derivatives. High-order methods based on similar binary search procedures have been further developed and shown to achieve a rate of $O(\epsilon^{-2/(p+1)}\log(1/\epsilon))$~\citep{Bullins-2020-Higher,Lin-2021-Monotone,Jiang-2022-Generalized}. However, such search procedure can be computationally prohibitive in practice~\citep{Nesterov-2018-Lectures} and the problem of finding a simple high-order regularization methods remains as an open and challenging question in the optimization theory. We propose a $p^{\textnormal{th}}$-order method that does \textit{not} require any binary search procedure and prove that it can converge to a weak solution at a global rate of $O(\epsilon^{-2/(p+1)})$. A lower bound of $\Omega(\epsilon^{-2/(p+1)})$ is also established under a linear span assumption to show that our $p^{\textnormal{th}}$-order method is optimal in the monotone setting. A version with restarting attains a global linear and local superlinear convergence rate for smooth and strongly monotone VIs. Our method can achieve a global rate of $O(\epsilon^{-2/p})$ for solving smooth and non-monotone VIs satisfying the Minty condition. The restarted version again attains a global linear and local superlinear convergence rate if the strong Minty condition holds. 
<br>
</details>
<br>
<details>
	<summary> <b> Mengqui Lou, Kabir A Chandrasekher, Ashwin Pananjady </b>: <i> Alternating minimization for generalized rank one matrix sensing: Sharp predictions from a random initialization </i> </summary>

<br>
We consider the problem of estimating the factors of a rank-1 matrix with i.i.d. Gaussian, rank-1 measurements that are nonlinearly transformed and corrupted by noise. Considering two prototypical choices for the nonlinearity, we study the convergence properties of a natural alternating update rule for this nonconvex optimization problem starting from a random initialization. We show sharp linear convergence guarantees for a sample-split version of the algorithm by deriving a deterministic recursion that is accurate even in high-dimensional problems. Our sharp, non-asymptotic analysis also exposes several other fine-grained properties of this problem, including how the nonlinearity, sample size, and noise level affect convergence behavior. Our results are enabled by showing that the empirical error recursion can be predicted by our deterministic sequence within fluctuations of the order n^{-1/2} when each iteration is run with n observations. Our technique leverages leave-one-out tools and provides an avenue for sharply analyzing higher-order iterative algorithms from a random initialization in other optimization problems with random data.
<br>
</details>
<br>
<details>
	<summary> <b> Zachary Novack, Simran Kaur, Tanya Marwah, Saurabh Garg, Zachary Lipton </b>: <i> Disentangling the Mechanisms Behind Implicit Regularization in SGD </i> </summary>

<br>
A number of competing hypotheses have been proposed to explain why small-batch Stochastic Gradient Descent (SGD) leads to improved generalization over the full-batch regime, with recent work crediting the implicit regularization of various quantities throughout training. However, to date, empirical evidence assessing the explanatory power of these hypotheses is lacking. In this paper, we conduct an extensive empirical evaluation, focusing on the ability of various theorized mechanisms to close the small-to-large batch generalization gap. Additionally, we characterize how the quantities that SGD has been claimed to (implicitly) regularize change over the course of training. By using micro-batches, i.e. disjoint smaller subsets of each mini-batch, we empirically show that explicitly penalizing the gradient norm or the Fisher Information Matrix trace, averaged over micro-batches, in the large-batch regime recovers small-batch SGD generalization, whereas Jacobian-based regularizations fail to do so. This generalization performance is shown to often be correlated with how well the regularized model’s gradient norms resemble those of small-batch SGD. We additionally show that this behavior breaks down as the micro-batch size approaches the batch size. Finally, we note that in this line of inquiry, positive experimental findings on CIFAR10 are often reversed on other datasets like CIFAR100, highlighting the need to test hypotheses on a wider collection of datasets.
<br>
</details>
<br>
<details>
	<summary> <b> Kazuki Osawa, Satoki Ishikawa, Rio Yokota, Shigang Li, Torsten Hoefler </b>: <i> ASDL: A Unified Interface for Gradient Preconditioning in PyTorch </i> </summary>

<br>
Gradient preconditioning is a key technique to integrate the second-order information into gradients for improving and extending gradient-based learning algorithms. In deep learning, stochasticity, nonconvexity, and high dimensionality lead to a wide variety of gradient preconditioning methods, with implementation complexity and inconsistent performance and feasibility. We propose the Automatic Second-order Differentiation Library (ASDL), an extension library for PyTorch, which offers various implementations and a plug-and-play unified interface for gradient preconditioning. ASDL enables the study and structured comparison of a range of gradient preconditioning methods.
<br>
</details>
<br>
<details>
	<summary> <b> Jean Pachebat, Sergei Ivanov </b>: <i> High-Order Optimization of Gradient Boosted Decision Trees </i> </summary>

<br>
Gradient Boosted Decision Trees (GBDTs) are dominant machine learning algorithms for modeling discrete or tabular data. Unlike neural networks with millions of trainable parameters, GBDTs optimize loss function in an additive manner and have a single trainable parameter per leaf, which makes it easy to apply high-order optimization of the loss function. In this paper, we introduce high-order optimization for GBDTs based on numerical optimization theory which allows us to construct trees based on high-order derivatives of a given loss function. In the experiments, we show that high-order optimization has faster per-iteration convergence that leads to reduced running time. Our solution can be easily parallelized and run on GPUs with little overhead on the code. Finally, we discuss future potential improvements such as automatic differentiation of arbitrary loss function and combination of GBDTs with neural networks.
<br>
</details>
<br>
<details>
	<summary> <b> Dmitry A. Pasechnyuk, Alexander Gasnikov, Martin Takac </b>: <i> Effects of momentum scaling for SGD </i> </summary>

<br>
The paper studies the properties of stochastic gradient methods with preconditioning. We focus on momentum updated preconditioners with momentum coefficient $\beta$. Seeking to explain practical efficiency of scaled methods, we provide convergence analysis in a norm associated with preconditioner, and demonstrate that scaling allows one to get rid of gradients Lipschitz constant in convergence rates. Along the way, we emphasize important role of $\beta$, undeservedly set to constant $0.99...9$ at the arbitrariness of various authors. Finally, we propose the explicit constructive formulas for adaptive $\beta$ and step size values.
<br>
</details>
<br>
<details>
	<summary> <b> Krishna Pillutla, Vincent Roulet, Sham Kakade, Zaid Harchaoui </b>: <i> The Trade-offs of Incremental Linearization Algorithms for Nonsmooth Composite Problems </i> </summary>

<br>
Gauss-Newton methods and their stochastic version have been widely used in machine learning. Their non-smooth counterparts, modified Gauss-Newton or prox-linear algorithms, can lead to contrasted outcomes when compared to gradient descent in large scale settings. We explore the contrasting performance of these two classes of algorithms in theory on a stylized statistical example, and experimentally on learning problems including structured prediction. 
<br>
</details>
<br>
<details>
	<summary> <b> Omead Pooladzandi, Yiming Zhou </b>: <i> Improving Levenberg-Marquardt Algorithm for Neural Networks </i> </summary>

<br>
We explore the usage of the Levenberg-Marquardt(LM) algorithm for regression (non-linear least squares) and classification (generalized Gauss-Newton methods) tasks in neural networks. We compare the performance of the LM method with other popular first-order algorithms such as SGD and Adam, as well as other second-order algorithms such as L-BFGS, Hessian-Free  and KFAC. We further speed up the LM method by using adaptive momentum, learning rate line search, and uphill step acceptance.
<br>
</details>
<br>
<details>
	<summary> <b> Vincent Roulet, Maryam Fazel, Siddhartha Srinivasa, Zaid Harchaoui </b>: <i> On the Global Convergence of the Regularized Generalized Gauss-Newton Algorithm </i> </summary>

<br>
We detail the global convergence rates of a regularized generalized Gauss-Newton algorithm applied to compositional problems with surjective inner Jacobian mappings. Our analysis uncovers several convergence phases for the algorithm and identifies the key condition numbers governing the complexity of the algorithm. We present an implementation with a line-search adaptive to the constants of the problem.
<br>
</details>
<br>
<details>
	<summary> <b> Di Zhang, Suvrajeet Sen </b>: <i> A Stochastic Conjugate Subgradient Algorithm for Kernelized Support Vector Machines: The Evidence </i> </summary>

<br>
Kernel Support Vector Machines (Kernel SVM) provide a powerful class of tools for classifying data whose classes are best identified via a nonlinear function. While a Kernel SVM is usually treated as a Quadratic Program (QP), its solution is usually obtained using stochastic gradient descent (SGD). In this paper we treat the Kernel SVM as a Stochastic Quadratic Linear Programming (SQLP) problem which motivates a decomposition-based algorithm that separates parameter choice from error estimation, with the latter being separable by data points. In order to take advantage of the quadratic structure due to the kernel matrix we introduce a conjugate subgradient approach. While convergence of the new method can be shown, the focus of this brief paper is on computational evidence which illustrates that our method maintains the scalability of SGD, while improving the accuracy of classification/optimization.
<br>
</details>
<br>
<details>
	<summary> <b> Chuwen Zhang, Jiang Bo, Chang He, Yuntian Jiang, Dongdong Ge, Yinyu Ye </b>: <i> DRSOM: A Dimension Reduced Second-Order Method </i> </summary>

<br>
In this paper, we propose a Dimension-Reduced Second-Order Method (DRSOM) for convex and nonconvex (unconstrained) optimization. Under a trust-region-like framework, our method preserves the convergence of the second-order method while using only Hessianvector products in a few directions, which enables the computational overhead of our method remain comparable to the first-order such as the gradient descent method. Theoretically, we show that the method has a local quadratic convergence and a global convergence rate of O(ϵ −3/2 ) to satisfy the first-order and second-order conditions under a commonly adopted approximated Hessian assumption. We further show that this assumption can be removed if we perform a step of the Lanczos method periodically at the end-stage of the algorithm. The applicability and performance of DRSOM are exhibited by various computational experiments, particularly in machine learning and deep learning. For neural networks, our preliminary implementation seems to gain computational advantages in terms of training accuracy and iteration complexity over state-of-the-art first-order methods such as SGD and ADAM.
<br>
</details>
<br>
<details>
	<summary> <b> Mingxi Zhu, Yinyu Ye </b>: <i> How Small Amount of Data Sharing Benefits Higher-Order Distributed Optimization and Learning </i> </summary>

<br>
Distributed optimization algorithms have been widely used in machine learning and statistical estimation. While distributed algorithms have the merits in parallel processing and protecting local data security, they often suffer from slow convergence compared with centralized optimization algorithms. This paper focuses on how small amount of data sharing could benefit distributed higher-order optimization algorithms with its application in learning problems. Specifically, we consider how data sharing could benefit distributed multi-block alternating direction method of multipliers (ADMM) and preconditioned conjugate gradient method (PCG) with application in machine learning tasks of linear and logistic regression. These algorithms are commonly known as algorithms between the first and the second order methods. Theoretically, we prove that a small amount of data share leads to improvements from near-worst to near-optimal convergence rate when applying ADMM and PCG methods to machine learning tasks. A side theory product is the tight worst-case bound of linear convergence rate for distributed ADMM applied in linear regression. We further propose a meta randomized data-sharing scheme and provide its tailored applications in multi-block ADMM and PCG methods in order to enjoy both the benefit from data-sharing and from the efficiency of distributed computing. From the numerical evidences, we are convinced that our algorithms provide good quality of estimators in both the least square and the logistic regressions within much fewer iterations by only sharing a small amount of pre-fixed data, while purely distributed optimization algorithms may take hundreds more times of iterations to converge. We hope that the discovery resulted from this paper would encourage even small amount of data sharing among different regions to combat difficult global learning problems.
<br>
</details>
<br>
<details>
	<summary> <b> Wenqi Zhu, Coralia Cartis </b>: <i> Quartic Polynomial Sub-problem Solutions in Tensor Methods for Nonconvex Optimization </i> </summary>

<br>
There has been growing interest in high-order tensor methods for nonconvex optimization in machine learning as these methods provide better/optimal worst-case evaluation complexity, stability to parameter tuning, and robustness to problem conditioning. The well-known $p$th-order adaptive regularization (AR$p$) method relies crucially on repeatedly minimising a nonconvex multivariate Taylor-based polynomial sub-problem. It remains an open question to find efficient techniques to minimise such a sub-problem for $p\ge3$.  In this paper, we propose a second-order method (SQO) for the AR$3$ (AR$p$ with $p=3$) sub-problem. SQO approximates the special-structure quartic polynomial sub-problem from above and below by using second-order models that can be minimised efficiently and globally. We prove that SQO finds a local minimiser of a quartic polynomial, but in practice, due to its construction, it can find a much lower minimum than cubic regularization approaches. This encourages us to continue our quest for algorithmic techniques that find approximately global solutions for such polynomials.
<br>
</details>
<br>
