# Machine Learning in C for the sake of learning

## Read these : 

* http://neuralnetworksanddeeplearning.com By Michael Nielsen / Dec 2019
* https://en.wikipedia.org/wiki/Backpropagation
* https://en.wikipedia.org/wiki/Linear_algebra

Keeping in mind that in my implementation use a different way of aligning the 
indices from Nielsen's book. I'll write it up bellow, when I get to it.

## Motivations

* I knew it would involve linear algebra, which I've wanted to get a refresher in. 
* Has real implications when it comes to memory management and algorithm speeds.
* Honestly just looked like a fun project

## Equations

### Cost function

$$
\begin{align}
C_x &= \frac{1}{2} ||y - a^L|| ^ 2 \\
\Leftrightarrow C_x &= \frac{1}{2} \sqrt{(y_0 - a_0^L)^ 2 + ... + (y_n - a_n^L)^2}^2\\
\Leftrightarrow C_x &= \frac{1}{2} \sum_{i=0}^n{(y_i - a_i^L)^2}
\end{align}
$$


$$C = \frac{1}{X}\sum_{x=0}^X C_x$$
