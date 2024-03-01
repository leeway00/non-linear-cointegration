# Non-Linear Cointegration in Pairs Trading

**Non-Linear Cointegration Models**:
- Threshold VECM (T-VECM)
- Functional Coefficient Cointegration method

**Research Target:**

1. The primary purpose is to analyze and model the non-linearity of cointegration from pairs trading. Please refer to `nonlinearcoint_pairs_trading.pdf` for the analysis result. I haven't run backtesting based on these models since the research is primarily focused on the robustness of the statistical arbitrage opportunity (e.g., whether the cointegrated relationship can be better captured/modeled at the out-of-sample).
For my research on trading signals at pairs trading, please take a look at [Copula Pairs Trading Strategy](https://github.com/leeway00/FINM_33150_Final_Project/blob/master/Notebook.ipynb), where the entire Section 2 (signal generation using Archimedean Mixture Copula with Non-Parametric Marginal) is my contribution.

2. The second target is to explore and implement the two non-linear cointegration models. For the Python implementation of the Seo test for T-VECM, I mainly referenced Sitgler's algorithm from R (tsDyn package). Functional Coefficient Cointegration (Xiao, 2009) is implemented with truncated Gaussian distribution as a kernel function.

**Data used**: 
- Market index-related ETFs
- Daily series

**References for implemented models**:

- Balke, N. S., & Fomby, T. B. (1997). Threshold cointegration. International economic review, 627-645.
- Seo, M. (2006). Bootstrap testing for the null of no cointegration in a threshold vector error correction model. Journal of Econometrics, 134(1), 129-150.
- Seo, M. H. (2008). Unit root test in a threshold autoregression: asymptotic theory and residual-based block bootstrap. Econometric Theory, 24(6), 1699-1716.
- Seo, M. H. (2011). Estimation of nonlinear error correction models. Econometric Theory, 27(2), 201-234.
- Stigler, M. (2010). Threshold cointegration: overview and implementation in R. R package version 0.7-2. URL
- Xiao, Z. (2009). Functional-coefficient cointegration models. Journal of Econometrics, 152(2), 81-92.

