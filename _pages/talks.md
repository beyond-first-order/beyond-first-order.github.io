---
layout: page
permalink: /talks/
title: talks
description: All confirmed talks for BFO-2022. Click on a title to expand the abstract.
nav: true
nav_order: 2
---

<details>
  <summary>**Coralia Cartis**: Tensor Methods for Nonconvex Optimization</summary>
  
We consider the advantages of having and incorporating higher- (than second-) order derivative information inside regularization frameworks, generating higher-order regularization algorithms that have better complexity, universal properties and can certify higher-order criticality of candidate solutions. Time permitting, we also discuss inexact settings where problem information and smoothness assumptions are weakened, without affecting the algorithms’ complexity. Efficient solution of some higher-order polynomial subproblems will also be discussed. 
</details>

<details>
  <summary>**Frank E. Curtis**: Deterministically Constrained Stochastic Optimization</summary>
  
This talk highlights the recent work by my research group on the design, analysis, and implementation of algorithms for solving continuous nonlinear optimization problems that involve a stochastic objective function and deterministic constraints.  We will focus on our sequential quadratic optimization (commonly known as SQP) methods for cases when the constraints are defined by nonlinear systems of equations and inequalities.  Our methods are applicable for solving various types of problems, such as for training machine learning (e.g., deep learning) models with constraints.  Our work focuses on the "fully stochastic" regime in which only stochastic gradient estimates are employed, for which we have derived convergence-in-expectation results and worst-case iteration complexity bounds that are on par with stochastic gradient methods for the unconstrained setting. We will also discuss the various extensions that my group is exploring.
</details>

<details>
  <summary>**Madeleine Udell**: Low Rank Approximation for Faster Convex Optimization</summary>
  
Low rank structure is pervasive in real-world datasets. This talk shows how to accelerate the solution of fundamental computational problems, including eigenvalue decomposition, linear system solves, and composite convex optimization,by exploiting this low rank structure. We present a simple and efficient method for approximate top eigendecomposition based on randomized numerical linear algebra. Armed with this primitive, we design a new randomized preconditioner for the conjugate gradient method, and a method called NysADMM, based on the inexact alternating directions method of multipliers, for composite convex optimization. These methods come with strong theoretical and numerical support. Indeed, a simple implementation of NysADMM solves important large-scale statistical problems like lasso, logistic regression, and support vector machines 2--58x faster than standard solvers.
</details>
