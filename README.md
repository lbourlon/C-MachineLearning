# Machine Learning in C for the sake of learning

## Usage

```bash

# You'll need to download the training data for yourself :
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Unzip them to the location the code expects
mkdir -p mnist/train && mkdir -p mnist/validate
gzip -d t10k-labels-idx1-ubyte.gz -c > mnist/validate/labels.idx1
gzip -d t10k-images-idx3-ubyte.gz -c > mnist/validate/images.idx3
gzip -d train-labels-idx1-ubyte.gz -c > mnist/train/labels.idx1
gzip -d train-images-idx3-ubyte.gz -c > mnist/train/images.idx3

# Compile and run the application
make release && ./release
```

## Read these : 

* http://neuralnetworksanddeeplearning.com By Michael Nielsen / Dec 2019
* https://en.wikipedia.org/wiki/Backpropagation
* https://en.wikipedia.org/wiki/Linear_algebra

mnist data source :
http://yann.lecun.com/exdb/mnist/
(the ones provided in Nielsen's repo is a python pickle file)

## Motivations

* I knew it would involve linear algebra, which I've wanted to get a refresher in. 
* Has real implications when it comes to memory management and algorithm speeds.
* Honestly just looked like a fun project
* I'm now in too deep to go back D:

## Structures and Indices alignment

| Network Input  | Hidden layer 1 | Hidden layer 2 | Network Output |
| -------------  | -------------  | -------------  | -------------  |
|      N/A       |    net->w[1]   |    net->w[2]   |    net->w[3]   |
|      N/A       |    net->b[1]   |    net->b[2]   |    net->b[3]   |
|    act->a[0]   |    act->a[1]   |    act->a[2]   |    act->a[3]   |
|      N/A       |    act->z[1]   |    act->z[2]   |    act->z[3]   |
|      N/A       |    act->e[1]   |    act->e[2]   |    act->e[3]   |


The struct network contains the information about a given network. That is, the number of
layers, the number of nodes in each layer, a list of weight matrixes and a list of bias
vectors.

The struct activations contains the activations in each layer, the weighted activations
in each layer and the errors for each layer. Where act->a[0] is the activation of the 0th
layer.

And so net->w[1] and net->[1] are the weight and biases applied to act->a[0] (network input)
The output of this operation is act->z[1] the 1st weighted activation. Which we then get
the net->a[1] by applying the sigmoid function.

act->e[1] is used by the backpropagation algorithm, it is a measure of how wrong act->z[1] is.

## Botlenecks
- nw_feed_forward which takes ~43% of total execution time.
- nw_gradient_descent takes ~55% of total execution time aswell.

Execution times percentages depend on configuration, number of batches and other things. These are for the current main.c file.

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
