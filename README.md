# Controllable Financial (conditional) Diffusion Model for Time Series Generation

A synthetic financial data generation model based on __conditional diffusion 
models__ that accepts __condition__ about synthetic data. Conditions are derived 
from price data into the generator via __cross attention__.

## Methodology abstraction:
CoFinDiff transforms stock price series into images using __Harr wavelet__ [Ramsey et al., 1995](https://www.worldscientific.com/doi/epdf/10.1142/S0218348X95000291)
___

### Objective 1.1:
Reproduce the results from the paper to satisfy the architecture built 
by the authors.
#### sub objectives: 
1. Dataset Collection 
2. Input Data Processing:
    - Logarithmic returns of price series. 
    - Multiply logarithmic returns by 100
3. Calculation for Conditions:
    - Trend
    - Volatility

### Objective 1.2: Building the input to the architecture 
1. Wavelet Transformation (For time-frequency representation of the financial signal)
2. Get the time series as Image data
3. Conditional Calculations 

### Objective 1.3: Diffusion Model 
1. Understanding and Explanation based on our implementation [Flow matching and Diffusion Models](https://diffusion.csail.mit.edu/2026/index.html)
   - asdkfj 
---

### Objective 2:
Make changes to the pipeline based on my financial data. 


---

### Objective 3:
Check the synthetic data results for both objective 1 and 2 against my evaluation
framework. 

---