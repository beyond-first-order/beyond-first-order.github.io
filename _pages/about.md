---
layout: about
title: about
permalink: /
subtitle: <p style="color:#359B9C";>Order up! The Benefits of Higher-Order Optimization in Machine Learning</p>
news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: true  # includes social icons at the bottom of the page
---

<img style="float: right; " src="../assets/img/logo.png"
     width="200"
     height="200" />


Optimization is a cornerstone of nearly all modern machine learning (ML) and deep learning (DL). Simple first-order gradient-based methods dominate the field for convincing reasons: low computational cost, simplicity of implementation, and strong empirical results.

Yet second- or higher-order methods are rarely used in DL, despite also having many strengths: faster per-iteration convergence, frequent explicit regularization on step-size, and better parallelization than SGD. Additionally, many scientific fields use second-order optimization with great success.

A driving factor for this is the large difference in development effort. By the time higher-order methods were tractable for DL, first-order methods such as SGD and it's main varients (SGD + Momentum, Adam, ...) already had many years of maturity and mass adoption.

The purpose of this workshop is to address this gap, to create an environment where higher-order methods are fairly considered and compared against one-another, and to foster healthy discussion with the end goal of mainstream acceptance of higher-order methods in ML and DL.
