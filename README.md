# Machine Learning in C for the sake of learning

## Read these : 

* http://neuralnetworksanddeeplearning.com By Michael Nielsen / Dec 2019
* https://en.wikipedia.org/wiki/Backpropagation
* https://en.wikipedia.org/wiki/Linear_algebra

mnist data source :
http://yann.lecun.com/exdb/mnist/
(the ones provided in Nielsen's repo is a python pickle file)

Keeping in mind that in my implementation use a different way of aligning the 
indices from Nielsen's book. I'll write it up bellow, when I get to it.

## Motivations

* I knew it would involve linear algebra, which I've wanted to get a refresher in. 
* Has real implications when it comes to memory management and algorithm speeds.
* Honestly just looked like a fun project
* I'm now in too deep to go back D:

## Equations

### Cost function


$$
\begin{align}
C_x &= \frac{1}{2} ||y - a^L|| ^ 2 \\
\Leftrightarrow C_x &= \frac{1}{2} \sqrt{(y_0 - a_0^L)^ 2 + ... + (y_{N-1} - a_{N-1}^L)^2}^2\\
\Leftrightarrow C_x &= \frac{1}{2} \sum_{n=0}^{N-1}{(y_n - a_n^L)^2}
\end{align}
$$


* $y$ is the vector of expected_outputs for an input $x$ vector of inputs;
* $a^L$ is the activation of the last layer (ie output of network)
* $C_x$ the cost, which is an estimate of how wrong the network is, for a given input $x$
* $N-1$ is the number of nodes in the last layer

$$C = \frac{1}{X}\sum_{x=0}^{X-1} C_x$$

* $C$ is the average of the $C_x$ for X training inputs
