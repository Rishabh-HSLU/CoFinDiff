## Section 1.1: Generative Modeling As Sampling

What does it mean to *generate* something?
 
Let's say we want to generate a financial time series for a stock ticker from NASDAQ.
In simple terms, a financial time series is a realization of a stochastic process.
To keep things concrete, we fix a horizon of $d$ time steps, so a realization is a vector
$x \in \mathbb{R}^d$ (e.g. $d$ daily closing prices).
 
There is no single best realization of this series. A generator could produce many different,
equally plausible paths for the same ticker, and which one is "best" depends on the downstream task 
(e.g. stress-testing a predictive model), some realizations will fit that purpose better than others,
but none is uniquely correct.
 
In ML, it is common to capture this diversity of plausible realizations as a probability distribution.
We call this the **data distribution**, denoted $p_{data}$, a probability density function $p_{data}:
\mathbb{R}^d \to \mathbb{R}_{\geq 0}$. The density $p_{data}(x)$ assigns a relative likelihood to each
possible realization $x$: realizations near regions of high density are more probable than those in
low-density regions (though, as with any continuous density, the probability of any single exact point
is technically zero, what matters is relative likelihood and probability mass over regions).
 
A generator, in this view, is something that, assuming it has truly captured the data-generating process
of the ticker can be sampled from to produce new realizations consistent with $p_{data}$.
Therefore, how "good" a series fits - a rather subjective statement - is replaced by how 
**likely** it is under the data distribution specfic to the downstream task. With this, 
we can mathematically express the task of generation as sampling from the (unknown) distribution
$p_{data}$. 

**Key Ideas:**
1. Object being generated as vectors
2. Generating an object $z$ is modeled as sampling from data distribution $z$ ~ $p_{data}$.
3. A dataset consists of finite number of samples $z_{1}, ... z_{n}$ sampled independently 
from $p_{data}$.

 
