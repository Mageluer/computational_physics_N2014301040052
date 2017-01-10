# Python Machine Learning

## Preface
We live in the midst of a data deluge. According to recent estimates, 2.5 Quintilian ($$10^{18}$$ ) bytes of data are generated on a daily basis. This is so much data that over 90 percent of the information that we store nowadays was generated in the past decade alone. Unfortunately, most of this information cannot be used by humans. Either the data is beyond the means of standard analytical methods, or it is simply too vast for our limited minds to even comprehend.

Through Machine Learning, we enable computers to process, learn from, and draw actionable insights out of the otherwise impenetrable walls of big data. From the massive supercomputers that support Google's search engines to the smartphones that we carry in our pockets, we rely on Machine Learning to power most of the world around us—often, without even knowing it.

Specific to the programming language level, Python is one of the best choices, because it is simple enough and practical. Even in the field of data science, Python can be said to be tight in the top spot. The reason is that we don't have to spend time on boring grammar details. Once you have the idea, you can quickly implement the algorithm and test it on a real data set.

## 1 Introduction
In my opinion, machine learning, the application and science of algorithms that makes sense of data, is the most exciting field of all the computer sciences! We are living in an age where data comes in abundance; using the self-learning algorithms from the field of machine learning, we can turn this data into knowledge. Thanks to the many powerful open source libraries that have been developed in recent years, there has probably never been a better time to break into the machine learning field and learn how to utilize powerful algorithms to spot patterns in data and make predictions about future events.

We will learn about the main concepts and different types of machine learning. Together with a basic introduction to the relevant terminology, we will lay the groundwork for successfully using machine learning techniques for practical problem solving.

we will cover the following topics:

> - The general concepts of machine learning 
> - The three types of learning and basic terminology
> - The building blocks for successfully designing machine learning systems
> - Installing and setting up Python for data analysis and machine learning

### 1.1 Building intelligent machines to transform data into knowledge
In this age of modern technology, there is one resource that we have in abundance: a large amount of structured and unstructured data. In the second half of the twentieth century, machine learning evolved as a subfield of *artificial intelligence* that involved the development of self-learning algorithms to gain knowledge from that data in order to make predictions. Instead of requiring humans to manually derive rules and build models from analyzing large amounts of data, machine learning offers a more efficient alternative for capturing the knowledge in data to gradually improve the performance of predictive models, and make data-driven decisions. Not only is machine learning becoming increasingly important in computer science research but it also plays an ever greater role in our everyday life. Thanks to machine learning, we enjoy robust e-mail spam filters, convenient text and voice recognition software, reliable Web search engines, challenging chess players, and, hopefully soon, safe and efficient self-driving cars.

### 1.2 The three different types of machine learning
In this section, we will take a look at the three types of machine learning: *supervised learning*, *unsupervised learning*, and *reinforcement learning*. We will learn about the fundamental differences between the three different learning types and, using conceptual examples, we will develop an intuition for the practical problem domains where these can be applied:

![three_types_of_ML](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/three_types_of_ML.png)

#### 1.2.1 Making predictions about the future with supervised learning
The main goal in supervised learning is to learn a model from labeled *training* data that allows us to make predictions about unseen or future data. Here, the term *supervised* refers to a set of samples where the desired output signals (labels) are already known.

Considering the example of e-mail spam filtering, we can train a model using a supervised machine learning algorithm on a corpus of labeled e-mail, e-mail that are correctly marked as spam or not-spam, to predict whether a new e-mail belongs to either of the two categories. A supervised learning task with discrete *class labels*, such as in the previous e-mail spam-filtering example, is also called a *classification* task. Another subcategory of supervised learning is *regression*, where the outcome signal is a continuous value:

![e-mail_spam_filtering](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/e-mail_spam_filtering.png)

##### 1.2.1.1 Classification for predicting class labels
Classification is a subcategory of supervised learning where the goal is to predict the categorical class labels of new instances based on past observations. Those class labels are discrete, in-ordered values that can be understood as the group memberships of the instances. The previously mentioned example of e-mail-spam detection represents a typical example of a binary classification task, where the machine learning algorithm learns a set of rules in order to distinguish between two possible classes: spam and non-spam e-mail.

However, the set of class labels does not have to be of a binary nature. The predictive model learned by a supervised learning algorithm can assign any class label that was presented in the training dataset to a new, unlabeled instance. A typical example of a *multi-class classification* task is handwritten character recognition. Here, we could collect a training data-set that consists of multiple handwritten examples of each letter in the alphabet. Now, if a user provides a new handwritten character via an input device, our predictive model will be able to predict the correct letter in the alphabet with certain accuracy. However, our machine learning system would be unable to correctly recognize any of the digits zero to nine, for example, if they were not part of our training data sets.

The following figure illustrates the concept of a binary classification task given 30 training samples: 15 training samples are labeled as *negative class* (circles) and 15 training samples are labeled as *positive class* (plus signs). In this scenario, our dataset is two-dimensional, which means that each sample has two values associated with it: $$x_1$$ and $$x_2$$ . Now, we can use a supervised machine learning algorithm to learn a rule—the decision boundary represented as a black dashed line—that can separate those two classes and classify new data into each of those two categories given its $$x_1$$ and $$x_2$$ values:
![binary classification](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/binary_classification.jpg)

##### 1.2.1.2 Regression for predicting continuous outcomes
We learned in the previous section that the task of classification is to assign categorical, unordered labels to instances. A second type of supervised learning is the prediction of continuous outcomes, which is also called *regression analysis*. In regression analysis, we are given a number of *predictor* (explanatory) variables and a continuous response variable (outcome), and we try to find a relationship between those variables that allows us to predict an outcome.

For example, let's assume that we are interested in predicting the Math SAT scores of our students. If there is a relationship between the time spent studying for the test and the final scores, we could use it as training data to learn a model that uses the study time to predict the test scores of future students who are planning to take this test.

> The term regression was devised by Francis Galton in his article Regression Towards Mediocrity in Hereditary Stature in 1886. Galton described the biological phenomenon that the variance of height in a population does not increase over time. He observed that the height of parents is not passed on to their children but the children's height is regressing towards the population mean.

The following figure illustrates the concept of *linear regression*. Given a predictor variable $$x$$ and a response variable $$y$$, we fit a straight line to this data that minimizes the distance—most commonly the average squared distance—between the sample points and the fitted line. We can now use the intercept and slope learned from this data to predict the outcome variable of new data:

![linear regression](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/linear_regression.png)

#### 1.2.2 Solving interactive problems with reinforcement learning
Another type of machine learning is reinforcement learning. In reinforcement learning, the goal is to develop a system (*agent*) that improves its performance based on interactions with the *environment*. Since the information about the current state of the environment typically also includes a so-called *reward* signal, we can think of reinforcement learning as a field related to *supervised* learning. However, in reinforcement learning this feedback is not the correct ground truth label or value, but a measure of how well the action was measured by a *reward* function. Through the interaction with the environment, an agent can then use reinforcement learning to learn a series of actions that maximizes this reward via an exploratory trial-and-error approach or deliberative planning.

A popular example of reinforcement learning is a chess engine. Here, the agent decides upon a series of moves depending on the state of the board (the environment), and the reward can be defined as *win* or *lose* at the end of the game:

![reinforcement learning](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/reinforcement_learning.png)

#### 1.2.3 Discovering hidden structures with unsupervised learning
In supervised learning, we know the *right answer* beforehand when we train our model, and in reinforcement learning, we define a measure of *reward* for particular actions by the agent. In unsupervised learning, however, we are dealing with unlabeled data or data of *unknown structure*. Using unsupervised learning techniques, we are able to explore the structure of our data to extract meaningful information without the guidance of a known outcome variable or reward function.

##### 1.2.3.1 Finding subgroups with clustering
*Clustering* is an exploratory data analysis technique that allows us to organize a pile of information into meaningful subgroups (*clusters*) without having any prior knowledge of their group memberships. Each cluster that may arise during the analysis defines a group of objects that share a certain degree of similarity but are more dissimilar to objects in other clusters, which is why clustering is also sometimes called "unsupervised classification." Clustering is a great technique for structuring information and deriving meaningful relationships among data, For example, it allows marketers to discover customer groups based on their interests in order to develop distinct marketing programs.

The figure below illustrates how clustering can be applied to organizing unlabeled data into three distinct groups based on the similarity of their features $$x_1$$ and $$x_2$$ :

![clustering](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/clustering.png)

##### 1.2.3.2 Dimensionality reduction for data compression
Another subfield of unsupervised learning is *dimensionality reduction*. Often we are working with data of high dimensionality—each observation comes with a high number of measurements—that can present a challenge for limited storage space and the computational performance of machine learning algorithms. Unsupervised dimensionality reduction is a commonly used approach in feature preprocessing to remove noise from data, which can also degrade the predictive performance of certain algorithms, and compress the data onto a smaller dimensional subspace while retaining most of the relevant information.

Sometimes, dimensionality reduction can also be useful for visualizing data—for example, a high-dimensional feature set can be projected onto one-, two-, or three-dimensional feature spaces in order to visualize it via 3D- or 2D-scatterplots or histograms. The figure below shows an example where non-linear dimensionality reduction was applied to compress a 3D *Swiss Roll* onto a new 2D feature subspace:

![dimensionality reduction](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/dimensionality_reduction.png)

### 1.3 An introduction to the basic terminology and notations
Now that we have discussed the three broad categories of machine learning—supervised, unsupervised, and reinforcement learning—let us have a look at the basic terminology that we will be using in the next sections. The following table depicts an excerpt of the *Iris* dataset, which is a classic example in the field of machine learning. The Iris dataset contains the measurements of 150 iris flowers from three different species: *Setosa*, *Versicolor*, and *Viriginica*. Here, each flower sample represents one row in our data set, and the flower measurements in centimeters are stored as columns, which we also call the features of the data-set:

![Iris](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/Iris.png)

To keep the notation and implementation simple yet efficient, we will make use of some of the basics of *linear algebra*. In the following sections, we will use a *matrix* and *vector* notation to refer to our data. We will follow the common convention to represent each sample as separate row in a feature matrix $$X$$ , where each feature is stored as a separate column.

The Iris dataset, consisting of 150 samples and 4 features, can then be written as a 150 × 4 matrix $$X\in\mathbb{R}^{150\times4}$$ :

![Iris dataset](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/Iris_dataset.png)

### 1.4 A road map for building machine learning systems
In the previous sections, we discussed the basic concepts of machine learning and the three different types of learning. In this section, we will discuss other important parts of a machine learning system accompanying the learning algorithm. The diagram below shows a typical workflow diagram for using machine learning in *predictive modeling*, which we will discuss in the following subsections:

![roadmap](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/roadmap.png)

#### 1.4.1 Preprocessing – getting data into shape
Raw data rarely comes in the form and shape that is necessary for the optimal performance of a learning algorithm. Thus, the *preprocessing* of the data is one of the most crucial steps in any machine learning application. If we take the Iris flower dataset from the previous section as an example, we could think of the raw data as a series of flower images from which we want to extract meaningful features. Useful features could be the color, the hue, the intensity of the flowers, the height, and the flower lengths and widths. Many machine learning algorithms also require that the selected features are on the same scale for optimal performance, which is often achieved by transforming the features in the range $$[0, 1]$$ or a standard normal distribution with zero mean and unit variance, as we will see in the later sections.

Some of the selected features may be highly correlated and therefore redundant to a certain degree. In those cases, dimensionality reduction techniques are useful for compressing the features onto a lower dimensional subspace. Reducing the dimensionality of our feature space has the advantage that less storage space is required, and the learning algorithm can run much faster.

To determine whether our machine learning algorithm not only performs well on the training set but also generalizes well to new data, we also want to randomly divide the dataset into a separate training and test set. We use the training set to train and optimize our machine learning model, while we keep the test set until the very end to evaluate the final model.

#### 1.4.2 Training and selecting a predictive model
As we will see in later sections, many different machine learning algorithms have been developed to solve different problem tasks. An important point that can be summarized from David Wolpert's famous *No Free Lunch Theorems* is that we can't get learning "for free" (*The Lack of A Priori Distinctions Between Learning Algorithms*, D.H. Wolpert 1996; *No Free Lunch Theorems for Optimization*, D.H. Wolpert and W.G. Macready, 1997). Intuitively, we can relate this concept to the popular saying, "*I suppose it is tempting, if the only tool you have is a hammer, to treat everything as if it were a nail*" (Abraham Maslow, 1966). For example, each classification algorithm has its inherent biases, and no single classification model enjoys superiority if we don't make any assumptions about the task. In practice, it is therefore essential to compare at least a handful of different algorithms in order to train and select the best performing model. But before we can compare different models, we first have to decide upon a metric to measure performance. One commonly used metric is classification accuracy, which is defined as the proportion of correctly classified instances.

One legitimate question to ask is: *how do we know which model performs well on the final test dataset and real-world data if we don't use this test set for the model selection but keep it for the final model evaluation*? In order to address the issue embedded in this question, different cross-validation techniques can be used where the training dataset is further divided into training and *validation subsets* in order to estimate the *generalization performance* of the model. Finally, we also cannot expect that the default parameters of the different learning algorithms provided by software libraries are optimal for our specific problem task. Therefore, we will make frequent use of *hyperparameter optimization* techniques that help us to fine-tune the performance of our model in later sections. Intuitively, we can think of those hyperparameters as parameters that are not learned from the data but represent the knobs of a model that we can turn to improve its performance, which will become much clearer in later sections when we see actual examples.

#### 1.4.3 Evaluating models and predicting unseen data instances
After we have selected a model that has been fitted on the training dataset, we can use the test dataset to estimate how well it performs on this unseen data to estimate the generalization error. If we are satisfied with its performance, we can now use this model to predict new, future data. It is important to note that the parameters for the previously mentioned procedures—such as feature scaling and dimensionality reduction—are solely obtained from the training dataset, and the same parameters are later re-applied to transform the test dataset, as well as any new data samples—the performance measured on the test data may be overoptimistic otherwise.

### 1.5 Using Python for machine learning
Python is one of the most popular programming languages for data science and therefore enjoys a large number of useful add-on libraries developed by its great community.

Although the performance of interpreted languages, such as Python, for computation-intensive tasks is inferior to lower-level programming languages, extension libraries such as *NumPy* and *SciPy* have been developed that build upon lower layer *Fortran* and *C* implementations for fast and vectorized operations on multidimensional arrays.

For machine learning programming tasks, we will mostly refer to the *scikit-learn* library, which is one of the most popular and accessible open source machine learning libraries as of today.

## 2 Training Machine Learning Algorithms for Classification
In this section, we will make use of one of the first algorithmically described machine learning algorithms for classification, the *perceptron* and *adaptive linear *neurons*. We will start by implementing a perceptron step by step in Python and training it to classify different flower species in the Iris dataset. This will help us to understand the concept of machine learning algorithms for classification and how they can be efficiently implemented in Python. Discussing the basics of optimization using adaptive linear neurons will then lay the groundwork for using more powerful classifiers via the scikit-learn machine-learning library in Section 3.

The topics that we will cover in this section are as follows:
> - Building an intuition for machine learning algorithms
> - Using pandas, NumPy, and matplotlib to read in, process, and visualize data
> - Implementing linear classification algorithms in Python

### 2.1 Artificial neurons – a brief glimpse into the early history of machine learning
Before we discuss the perceptron and related algorithms in more detail, let us take a brief tour through the early beginnings of machine learning. Trying to understand how the biological brain works to design artificial intelligence, Warren McCullock and Walter Pitts published the first concept of a simplified brain cell, the so-called *McCullock-Pitts (MCP) neuron*, in 1943 (W. S. McCulloch and W. Pitts. *A Logical Calculus of the Ideas Immanent in Nervous Activity.* The bulletin of mathematical biophysics, 5(4):115–133, 1943). Neurons are interconnected nerve cells in the brain that are involved in the processing and transmitting of chemical and electrical signals, which is illustrated in the following figure:

![Neurons](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/Neurons.png)

McCullock and Pitts described such a nerve cell as a simple logic gate with binary outputs; multiple signals arrive at the dendrites, are then integrated into the cell body, and, if the accumulated signal exceeds a certain threshold, an output signal is generated that will be passed on by the axon.

Only a few years later, Frank Rosenblatt published the first concept of the perceptron learning rule based on the MCP neuron model (F. Rosenblatt, *The Perceptron, a *Perceiving and Recognizing Automaton*. Cornell Aeronautical Laboratory, 1957). With his perceptron rule, Rosenblatt proposed an algorithm that would automatically learn the optimal weight coefficients that are then multiplied with the input features in order to make the decision of whether a neuron fires or not. In the context of supervised learning and classification, such an algorithm could then be used to predict if a sample belonged to one class or the other.

More formally, we can pose this problem as a binary classification task where we refer to our two classes as $$1$$ (positive class) and $$-1$$ (negative class) for simplicity. We can then define an *activation function* $$\phi(z)$$ that takes a linear combination of certain input values x and a corresponding weight vector $$w$$ , where $$z$$ is the so-called net input ($$z = w_1x_1 + \cdots + w_mx_m$$):

$$w=\begin{bmatrix}w_1\\ \vdots\\ w_m\end{bmatrix},\qquad x=\begin{bmatrix}x_1\\ \vdots\\ x_m\end{bmatrix}$$

Now, if the activation of a particular sample $$x$$ , that is, the output of $$\phi(z)$$, is greater than a defined threshold $$\theta$$ , we predict class $$1$$ and class $$-1$$, otherwise, in the perceptron algorithm, the activation function $$\phi(\cdot)$$ is a simple *unit step function*, which is sometimes also called the *Heaviside step function*:

$$\phi(z)=\left\{\begin{aligned}1&\quad if \quad z\ge0\\-1&\quad otherwise\end{aligned}\right.$$

For simplicity, we can bring the threshold $$\theta$$ to the left side of the equation and define a weight-zero as $$w_0 = − \theta$$ and $$x_0 = 1$$, so that we write $$z$$ in a more compact form $$z =  w_0x_0 + w_1x_1 + \cdots + w_mx_m = w^Tx$$ and $$\phi(z)=\left\{\begin{aligned}1&\quad if\quad z\ge\theta\\-1&\quad otherwise\end{aligned}\right.$$.

The following figure illustrates how the net input $$z =  w^Tx$$ is squashed into a binary output ($$-1$$ or $$1$$) by the activation function of the perceptron (left subfigure) and how it can be used to discriminate between two linearly separable classes (right subfigure):

![thresholded perceptron](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/thresholded_perceptron.png)

The whole idea behind the MCP neuron and Rosenblatt's *thresholded* perceptron model is to use a reductionist approach to mimic how a single neuron in the brain works: it either *fires* or it doesn't. Thus, Rosenblatt's initial perceptron rule is fairly simple and can be summarized by the following steps:

> 1. Initialize the weights to 0 or small random numbers.
> 2. For each training sample $$x^{( i )}$$ perform the following steps:
>> 1. Compute the output value $$\hat{y}$$ .
>> 2. Update the weights.

Here, the output value is the class label predicted by the unit step function that we defined earlier, and the simultaneous update of each weight $$w_j$$in the weight vector
$$w$$ can be more formally written as:

$$w_j : = w_j + \Delta w_j$$

The value of $$\Delta w_j$$ , which is used to update the weight $$w_j$$ , is calculated by the perceptron learning rule:

$$\Delta w_j = \eta( y ^{(i)} − \hat{y}^{(i)})x_j^{(i)}$$

Where $$\eta$$ is the learning rate (a constant between 0.0 and 1.0), $$y ^{(i)}$$ is the true class label of the $$i$$th training sample, and $$\hat{y}^{(i)}$$ is the predicted class label. It is important to note that all weights in the weight vector are being updated simultaneously, which means that we don't recompute the $$\hat{y}^{(i)}$$ before all of the weights $$\Delta w_j$$∆ w j were updated. Concretely, for a 2D dataset, we would write the update as follows:

$$\begin{aligned}
\Delta w_0 &= \eta( y ^{(i)} − output^{(i)}\\
\Delta w_1 &= \eta( y ^{(i)} − output^{(i)})x_1^{(i)}\\
\Delta w_2 &= \eta( y ^{(i)} − output^{(i)})x_2^{(i)}
\end{aligned}$$

Before we implement the perceptron rule in Python, let us make a simple thought experiment to illustrate how beautifully simple this learning rule really is. In the two scenarios where the perceptron predicts the class label correctly, the weights remain unchanged:

$$\begin{aligned}
\Delta w_j &= \eta( 1^{(i)} − 1^{(i)})x_j^{(i)}=0\\
\Delta w_j &= \eta( -1^{(i)} − -1^{(i)})x_j^{(i)}=0
\end{aligned}$$

However, in the case of a wrong prediction, the weights are being pushed towards the direction of the positive or negative target class, respectively:

$$\begin{aligned}
\Delta w_j &= \eta( 1^{(i)} − -1^{(i)})x_j^{(i)}=2\eta x_j^{(i)}\\
\Delta w_j &= \eta( -1^{(i)} − 1^{(i)})x_j^{(i)}=-2\eta x_j^{(i)}
\end{aligned}$$

To get a better intuition for the multiplicative factor $$x_j$$ , let us go through another simple example, where:

$$\hat{y}_j = + 1, y^{( i )} = − 1, \eta = 1$$

Let's assume that $$x_ j^{(i)} = 0.5$$, and we misclassify this sample as -1. In this case, we would increase the corresponding weight by 1 so that the activation $$x_j^{(i)} = w_j^{(i)}$$ will be more positive the next time we encounter this sample and thus will be more likely to be above the threshold of the unit step function to classify the sample as +1:

$$\Delta w_j = \eta( 1^{(i)} − -1^{(i)})0.5^{(i)}=(2)0.5^{(i)}=1$$ 

The weight update is proportional to the value of $$x_j^{(i)}$$ . For example, if we have another sample $$x_ j^{(i)} = 2$$ that is incorrectly classified as -1, we'd push the decision boundary by an even larger extend to classify this sample correctly the next time:

$$\Delta w_j = \eta( 1^{(i)} − -1^{(i)})2^{(i)}=(2)2^{(i)}=4$$

It is important to note that the convergence of the perceptron is only guaranteed if the two classes are linearly separable and the learning rate is sufficiently small. If the two classes can't be separated by a linear decision boundary, we can set a maximum  number of passes over the training dataset (*epochs*) and/or a threshold for the number of tolerated misclassifications—the perceptron would never stop updating the weights otherwise:

![linearly separable](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/linearly_separable.png)

Now, before we jump into the implementation in the next section, let us summarize what we just learned in a simple figure that illustrates the general concept of the perceptron:

![perceptron](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/perceptron.png)

The preceding figure illustrates how the perceptron receives the inputs of a sample $$x$$ and combines them with the weights $$w$$ to compute the net input. The net input is then passed on to the activation function (here: the unit step function), which generates a binary output -1 or +1—the predicted class label of the sample. During the learning phase, this output is used to calculate the error of the prediction and update the weights.

### 2.2 Implementing a perceptron learning algorithm in Python
In the previous section, we learned how Rosenblatt's perceptron rule works; let us now go ahead and implement it in Python and apply it to the Iris dataset. We will take an objected-oriented approach to define the perceptron interface as a Python Class , which allows us to initialize new perceptron objects that can learn from data via a `fit` method, and make predictions via a separate predict method. As a convention, we add an underscore to attributes that are not being created upon the initialization of the object but by calling the object's other methods—for example, `self.w_ `.

```py
import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta:float
    Learning rate (between 0.0 and 1.0)
    n_iter:int
    Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array
    Weights after fitting.
    errors_: list
    Numebr of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X: {array-like}, shape=[n_samples, n_features]
        Training vectors, where n_samples is the number of samples
        and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
        Target values.

        Returns
        ----------
        self: object
        """

        self.w_ = np.zeros(1 + X.shape[1]) # Add w_0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        """Calculate net input"""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0, 1, -1) #analoge ? : in C++
```

Using this perceptron implementation, we can now initialize new `Perceptron` objects with a given learning rate `eta` and `n_iter` , which is the number of epochs(passes over the training set). Via the `fit` method we initialize the weights in `self.w_` to a zero-vector $$\mathbb{R}^{m+1}$$ where $$m$$ stands for the number of dimensions(features) in the dataset where we add 1 for the zero-weight (that is, the threshold).

> NumPy indexing for one-dimensional arrays works similarly to Python lists using the square-bracket ([]) notation. For two-dimensional arrays, the first indexer refers to the row number, and the second indexer to the column number. For example, we would use X[2, 3] to select the third row and fourth column of a 2D array X.

After the weights have been initialized, the `fit` method loops over all individual samples in the training set and updates the weights according to the perceptron learning rule that we discussed in the previous section. The class labels are predicted by the `predict` method, which is also called in the `fit` method to predict the class label for the weight update, but predict can also be used to predict the class labels of new data after we have fitted our model. Furthermore, we also collect the number of misclassifications during each epoch in the list `self.errors_` so that we can later analyze how well our perceptron performed during the training. The `np.dot` function that is used in the `net_input` method simply calculates the vector dot product $$w^Tx$$.

#### 2.2.1 Training a perceptron model on the Iris dataset
To test our perceptron implementation, we will load the two flower classes *Setosa* and *Versicolor* from the Iris dataset. Although, the perceptron rule is not restricted to two dimensions, we will only consider the two features *sepal length* and *petal length* for visualization purposes. Also, we only chose the two flower classes *Setosa* and *Versicolor* for practical reasons. However, the perceptron algorithm can be extended to multi-class classification—for example, through the *[One-vs.-All](https://en.wikipedia.org/wiki/Multiclass_classification)* technique.

First, we will use the pandas library to load the Iris dataset directly from the *UCI* Machine Learning Repository into a DataFrame object and print the last five lines via the `tail` method to check that the data was loaded correctly:

```py
>>> import pandas as pd
>>> df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
>>> df.tail()
       0    1    2    3               4
145  6.7  3.0  5.2  2.3  Iris-virginica
146  6.3  2.5  5.0  1.9  Iris-virginica
147  6.5  3.0  5.2  2.0  Iris-virginica
148  6.2  3.4  5.4  2.3  Iris-virginica
149  5.9  3.0  5.1  1.8  Iris-virginica
>>>
```

Next, we extract the first 100 class labels that correspond to the 50 *Iris-Setosa* and 50 *Iris-Versicolor* flowers, respectively, and convert the class labels into the two integer class labels `1` (*Versicolor*) and `-1` (*Setosa*) that we assign to a vector `y` where the values
method of a pandas `DataFrame` yields the corresponding NumPy representation. Similarly, we extract the first feature column (*sepal length*) and the third feature column (*petal length*) of those 100 training samples and assign them to a feature matrix $$X$$, which we can visualize via a two-dimensional scatter plot:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
pl.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
pl.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
pl.xlabel('petal length')
pl.ylabel('sepal length')
pl.legend(loc='upper left')
pl.show()
```

After executing the preceding code example we should now see the following scatterplot:

![scatterplot](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/scatter_plot.png)

Now it's time to train our perceptron algorithm on the Iris data subset that we just extracted. Also, we will plot the *misclassification error* for each epoch to check if the algorithm converged and found a decision boundary that separates the two Iris flower classes:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import perceptron as pc

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

ppn = pc.Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

pl.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
pl.xlabel('Epoches')
pl.ylabel('Number of misclassifications')
pl.show()
```

After executing the preceding code, we should see the plot of the misclassification errors versus the number of epochs, as shown next:

![misclassification errors](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/misclassification_errors.png)

As we can see in the preceding plot, our perceptron already converged after the sixth epoch and should now be able to classify the training samples perfectly. Let us implement a small convenience function to visualize the decision boundaries for 2D datasets:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcl
import perceptron as pc

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

ppn = pc.Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = mcl.ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0],y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_region(X, y, classifier=ppn)
pl.xlabel('sepal length [cm]')
pl.ylabel('petal length [cm]')
pl.legend(loc='upper left')
pl.show()
```

After executing the preceding code example, we should now see a plot of the decision regions, as shown in the following figure:

![decision boundary](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/decision_boundaries.png)

As we can see in the preceding plot, the perceptron learned a decision boundary that was able to classify all flower samples in the Iris training subset perfectly.

> Although the perceptron classified the two Iris flower classes perfectly, convergence is one of the biggest problems of the perceptron. Frank Rosenblatt proved mathematically that the perceptron learning rule converges if the two classes can be separated by a linear hyperplane. However, if classes cannot be separated perfectly by such a linear decision boundary, the weights will never stop updating unless we set a maximum number of epochs.

### 2.3 Adaptive linear neurons and the convergence of learning
In this section, we will take a look at another type of single-layer neural network: **ADAptive LInear NEuron (Adaline)**. Adaline was published, only a few years after Frank Rosenblatt's perceptron algorithm, by Bernard Widrow and his doctoral student Tedd Hoff, and can be considered as an improvement on the latter (B. Widrow et al. Adaptive *"Adaline" neuron using chemical "memistors"*. Number Technical Report 1553-2. Stanford Electron. Labs. Stanford, CA, October 1960). The Adaline algorithm is particularly interesting because it illustrates the key concept of defining and minimizing cost functions, which will lay the groundwork for understanding more advanced machine learning algorithms for classification, such as logistic regression and support vector machines, as well as regression models that we will discuss in future chapters.

The key difference between the Adaline rule (also known as the *Widrow-Hoff* rule) and Rosenblatt's perceptron is that the weights are updated based on a linear activation function rather than a unit step function like in the perceptron. In Adaline, this linear activation function $$\phi(z)$$ is simply the identity function of the net input so that
$$\phi (w^Tx) = w^Tx$$.

While the linear activation function is used for learning the weights, a *quantizer*, which is similar to the unit step function that we have seen before, can then be used to predict the class labels, as illustrated in the following figure:

![Adaline rule](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/Adaline_rule.png)

If we compare the preceding figure to the illustration of the perceptron algorithm that we saw earlier, the difference is that we know to use the continuous valued output from the linear activation function to compute the model error and update the weights, rather than the binary class labels.

#### 2.3.1 Minimizing cost functions with gradient descent
One of the key ingredients of supervised machine learning algorithms is to define an objective function that is to be optimized during the learning process. This objective function is often a cost function that we want to minimize. In the case of Adaline, we can define the cost function J to learn the weights as the Sum of
**Squared Errors (SSE)** between the calculated outcome and the true class label

$$J(w)=\frac{1}{2}\sum_i\left(y^{(i)}-\phi(z^{(i)})\right)^2$$

The term $$1/2$$ is just added for our convenience; it will make it easier to derive the gradient, as we will see in the following paragraphs. The main advantage of this continuous linear activation function is—in contrast to the unit step function—that the cost function becomes differentiable. Another nice property of this cost function is that it is convex; thus, we can use a simple, yet powerful, optimization algorithm called *gradient descent* to find the weights that minimize our cost function to classify the samples in the Iris dataset.

As illustrated in the following figure, we can describe the principle behind gradient descent as *climbing down a hill* until a local or global cost minimum is reached. In each iteration, we take a step away from the gradient where the step size is determined by the value of the learning rate as well as the slope of the gradient:

![gradient descent](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/gradient_descent.png)

Using gradient descent, we can now update the weights by taking a step away from the gradient $$\Delta J ( w )$$ of our cost function $$J ( w )$$ :

$$w := w + \Delta w$$

Here, the weight change $$\Delta w$$ is defined as the negative gradient multiplied by the
learning rate $$\eta$$ :

$$\Delta w = − \eta \Delta J ( w )$$

To compute the gradient of the cost function, we need to compute the partial derivative of the cost function with respect to each weight $$w_j$$:

$$\frac{\partial J}{\partial w_j}=-\sum_i\left(y^{(i)}-\phi(z^{(i)})\right)x_j^{(i)}$$

so that we can write the update of weight $$w_j$$ as:

$$\Delta w_j=-\eta\frac{\partial J}{\partial w_j}=\eta\sum_i\left(y^{(i)}-\phi(z^{(i)})\right)x_j^{(i)}$$

Since we update all weights simultaneously, our Adaline learning rule becomes $$w := w + \Delta w$$.

Although the Adaline learning rule looks identical to the perceptron rule, the $$\phi(z^{(i)})$$ with $$Z^{(i)}=w^Tx_i$$ is a real number and not an integer class label. Furthermore, the weight update is calculated based on all samples in the training set (instead of updating the weights incrementally after each sample), which is why this approach is also referred to as "batch" gradient descent.

#### 2.3.2 Implementing an Adaptive Linear Neuron in Python
Since the perceptron rule and Adaline are very similar, we will take the perceptron implementation that we defined earlier and change the fit method so that the weights are updated by minimizing the cost function via gradient descent:

```py
import numpy as np

class AdalineGD(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta:float
        Learning rate (between 0.0 and 1.0)
    n_iter:int
        Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array
        Weights after fitting.
    cost_: list
        Numebr of misclassifications in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples
        and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
            Target values.

        Returns
        ----------
        self: object
        """

        self.w_ = np.zeros(1 + X.shape[1]) # Add w_0
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1) #analoge ? : in C++
```

Instead of updating the weights after evaluating each individual training sample, as in the perceptron, we calculate the gradient based on the whole training dataset via `self.eta * errors.sum()` for the zero-weight and via `self.eta * X.T.dot(errors)` for the weights 1 to $$m$$ where `X.T.dot(errors)` is a *matrix-vector multiplication* between our feature matrix and the error vector. Similar to the previous perceptron implementation, we collect the cost values in a list `self.cost_` to check if the algorithm converged after training.

In practice, it often requires some experimentation to find a good learning rate $$\eta$$ for optimal convergence. So, let's choose two different learning rates $$\eta = 0.1$$ and $$\eta = 0.0001$$ to start with and plot the cost functions versus the number of epochs to see how well the Adaline implementation learns from the training data.

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import AdalineGD as ad

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

def standardize(X):
    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    return X_std
ada1 = ad.AdalineGD(n_iter=10, eta=0.01)
ada1.fit(X, y)
ada2 = ad.AdalineGD(n_iter=10, eta=0.0001)
ada2.fit(X, y)

fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.0001')
pl.show()
```

As we can see in the resulting cost function plots next, we encountered two different types of problems. The left chart shows what could happen if we choose a learning rate that is too large—instead of minimizing the cost function, the error becomes larger in every epoch because we *overshoot* the global minimum:

![adaline-learning-rate](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/adaline-learning-rate.png)

Although we can see that the cost decreases when we look at the right plot, the chosen learning rate $$\eta = 0.0001$$ is so small that the algorithm would require a very large number of epochs to converge. The following figure illustrates how we change the value of a particular weight parameter to minimize the cost function J (left subfigure). The subfigure on the right illustrates what happens if we choose a learning rate that is too large, we overshoot the global minimum:

![overshoot](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/overshoot.png)

Many machine learning algorithms that we will encounter throughout this book require some sort of feature scaling for optimal performance.  Gradient descent is one of the many algorithms that benefit from feature scaling. Here, we will use a feature scaling method called *standardization,* which gives our data the property of a standard normal distribution. The mean of each feature is centered at value 0 and the feature column has a standard deviation of 1. For example, to standardize the $$j$$ th feature, we simply need to subtract the sample mean $$\mu_j$$ from every training sample and divide it by its standard deviation $$\sigma_j$$ :

$$x_j'=\frac{x_j-\mu_j}{\sigma_j}$$

Here $$x_j$$ is a vector consisting of the $$j$$ th feature values of all training samples $$n$$ . Standardization can easily be achieved using the NumPy methods `mean` and `std` :

```py
def standardize(X):
    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    return X_std
```

After standardization, we will train the Adaline again and see that it now converges using a learning rate $$\eta=0.01$$ :

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcl
import AdalineGD as ad

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

def standardize(X):
    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    return X_std
ada = ad.AdalineGD(n_iter=10, eta=0.01)
ada.fit(standardize(X), y)

pl.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
pl.xlabel('Epochs')
pl.ylabel('Sum-squared-error')
pl.title('Adaline - Learning rate 0.01')
pl.show()

def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = mcl.ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0],y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_region(standardize(X), y, classifier=ada)
pl.xlabel('sepal length [standardized]($$cm$$)')
pl.ylabel('petal length [standardized]($$cm$$)')
pl.title('Adaline - Gradient Descent')
pl.legend(loc='upper left')
pl.show()
```

After executing the preceding code, we should see a figure of the decision regions as well as a plot of the declining cost, as shown in the following figure:

![adaline-learning-rate-std](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/adaline-learning-rate-std.png)
![gradient_descent_std](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/gradient_descent_std.png)

As we can see in the preceding plots, the Adaline now converges after training on the standardized features using a learning rate $$\eta=0.01$$. However, note that the SSE remains non-zero even though all samples were classified correctly.

#### 2.3.3 Large scale machine learning and stochastic gradient descent
In the previous section, we learned how to minimize a cost function by taking a step into the opposite direction of a gradient that is calculated from the whole training set; this is why this approach is sometimes also referred to as batch gradient descent. Now imagine we have a very large dataset with millions of data points, which is not uncommon in many machine learning applications. Running batch gradient descent can be computationally quite costly in such scenarios since we need to reevaluate the whole training dataset each time we take one step towards the global minimum.

A popular alternative to the batch gradient descent algorithm is stochastic gradient descent, sometimes also called iterative or on-line gradient descent. Instead of updating the weights based on the sum of the accumulated errors over all samples $$x^{(i)}$$ :

$$\Delta w_j=\eta\sum_i\left(y^{(i)}-\phi(z^{(i)})\right)x_j^{(i)}$$

We update the weights incrementally for each training sample:

$$\eta\left(y^{(i)}-\phi(z^{(i)})\right)x_j^{(i)}$$

Although stochastic gradient descent can be considered as an approximation of gradient descent, it typically reaches convergence much faster because of the more frequent weight updates. Since each gradient is calculated based on a single training example, the error surface is noisier than in gradient descent, which can also have the advantage that stochastic gradient descent can escape shallow local minima more readily. To obtain accurate results via stochastic gradient descent, it is important to present it with data in a random order, which is why we want to shuffle the training set for every epoch to prevent cycles.

> In stochastic gradient descent implementations, the fixed learning rate $$\eta$$ is often replaced by an adaptive learning rate that decreases over time, for example, $$\frac{c_1}{[number of iterations]+c_2}$$ where $$c_1$$ and $$c_2$$ are constants. Note that stochastic gradient descent does not reach the global minimum but an area very close to it. By using an adaptive learning rate, we can achieve further annealing to a better global minimum

Another advantage of stochastic gradient descent is that we can use it for *online learning*. In online learning, our model is trained on-the-fly as new training data arrives. This is especially useful if we are accumulating large amounts of data—for example, customer data in typical web applications. Using online learning, the system can immediately adapt to changes and the training data can be discarded after updating the model if storage space in an issue.

> A compromise between batch gradient descent and stochastic gradient
descent is the so-called mini-batch learning. Mini-batch learning can be
understood as applying batch gradient descent to smaller subsets of
the training data—for example, 50 samples at a time. The advantage
over batch gradient descent is that convergence is reached faster
via mini-batches because of the more frequent weight updates.
Furthermore, mini-batch learning allows us to replace the for-loop
over the training samples in **Stochastic Gradient Descent (SGD)** by
vectorized operations, which can further improve the computational
efficiency of our learning algorithm.

Since we already implemented the Adaline learning rule using gradient descent, we only need to make a few adjustments to modify the learning algorithm to update the weights via stochastic gradient descent. Inside the `fit` method, we will now update the weights after each training sample. Furthermore, we will implement an additional `partial_fit` method, which does not reinitialize the weights, for on-line learning. In order to check if our algorithm converged after training, we will calculate the cost as the average cost of the training samples in each epoch. Furthermore, we will add an option to `shuffle` the training data before each epoch to avoid cycles when we are optimizing the cost function; via the `random_state` parameter, we allow the specification of a random seed for consistency:

```py
import numpy as np

class AdalineSGD(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta:float
        Learning rate (between 0.0 and 1.0)
    n_iter:int
        Passes over the training dataset.

    Attributes
    -------------
    w_: 1d-array
        Weights after fitting.
    cost_: list
        Numebr of misclassifications in every epoch.
    shuffle: bool(default: True)
        Shuffles training data every epoch
        if True to prevent cycles.
    random_state: int(default: None)
        Set random state for shuffling
        and initializing the weights

    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ------------
        X: {array-like}, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples
        and n_featuers is the number of features.
        y: array-like, shape=[n_smaples]
            Target values.

        Returns
        ----------
        self: object
        """

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = np.mean(cost)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1) #analoge ? : in C++
```

The `_shuffle` method that we are now using in the `AdalineSGD` classifier works as follows: via the permutation function in numpy.random , we generate a random sequence of unique numbers in the range 0 to 100. Those numbers can then be used as indices to shuffle our feature matrix and class label vector.

We can then use the `fit` method to train the `AdalineSGD` classifier and use our `plot_decision_regions` to plot our training results:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcl
import AdalineSGD as ad

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
"""
pl.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
pl.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
pl.xlabel('petal length')
pl.ylabel('sepal length')
pl.legend(loc='upper left')
pl.show()
"""
def standardize(X):
    X_std = np.copy(X)
    for i in range(X.shape[1]):
        X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    return X_std
ada = ad.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(standardize(X), y)

pl.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
pl.xlabel('Epochs')
pl.ylabel('Average Cost')
pl.title('Adaline - Learning rate 0.01')
pl.show()

def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = mcl.ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0],y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_region(standardize(X), y, classifier=ada)
pl.xlabel('sepal length [standardized]($$cm$$)')
pl.ylabel('petal length [standardized]($$cm$$)')
pl.title('Adaline - Stochastic Gradient Descent')
pl.legend(loc='upper left')
pl.show()

```

The two plots that we obtain from executing the preceding code example are shown in the following figure:

![Adaline SGD](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/Adaline_SGD.png)
![Adaline_SGD_learning_rate](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/Adaline_SGD_learning_rate.png)

As we can see, the average cost goes down pretty quickly, and the final decision boundary after 15 epochs looks similar to the batch gradient descent with Adaline. If we want to update our model—for example, in an on-line learning scenario with streaming data—we could simply call the `partial_fit` method on individual samples—for instance, `ada.partial_fit(X_std[0, :], y[0])` .

## 3 A Tour of Machine Learning Classifiers Using Scikit-learn
In this section, we will take a tour through a selection of popular and powerful machine learning algorithms that are commonly used in academia as well as in the industry. While learning about the differences between several supervised learning algorithms for classification, we will also develop an intuitive appreciation of their individual strengths and weaknesses. Also, we will take our first steps with the scikit-learn library, which offers a user-friendly interface for using those algorithms efficiently and productively.

The topics that we will learn about throughout this section are as follows:

> - Introduction to the concepts of popular classification algorithms
> - Using the scikit-learn machine learning library
> - Questions to ask when selecting a machine learning algorithm

### 3.1 Choosing a classification algorithm
Choosing an appropriate classification algorithm for a particular problem task requires practice: each algorithm has its own quirks and is based on certain assumptions. To restate the "No Free Lunch" theorem: no single classifier works best across all possible scenarios. In practice, it is always recommended that you compare the performance of at least a handful of different learning algorithms to select the best model for the particular problem; these may differ in the number of features
or samples, the amount of noise in a dataset, and whether the classes are linearly separable or not.

Eventually, the performance of a classifier, computational power as well as predictive power, depends heavily on the underlying data that are available for learning. The five main steps that are involved in training a machine learning algorithm can be summarized as follows:

> 1. Selection of features.
> 2. Choosing a performance metric.
> 3. Choosing a classifier and optimization algorithm.
> 4. Evaluating the performance of the model.
> 5. Tuning the algorithm.

### 3.2 First steps with scikit-learn
In Section 2, *Training Machine Learning Algorithms for Classification*, you learned about two related learning algorithms for classification: the **perceptron** rule and **Adaline**, which we implemented in Python by ourselves. Now we will take a look at the scikit-learn API, which combines a user-friendly interface with a highly optimized implementation of several classification algorithms. However, the scikit-learn library offers not only a large variety of learning algorithms, but also many convenient functions to preprocess data and to fine-tune and evaluate our models. We will discuss this in more detail together with the underlying concepts in following sections.

### 3.3 Training a perceptron via scikit-learn
To get started with the `scikit-learn` library, we will train a perceptron model similar to the one that we implemented in *Section 2, Training Machine Learning Algorithms for Classification*. For simplicity, we will use the already familiar **Iris** dataset throughout the following sections. Conveniently, the Iris dataset is already available via `scikit-learn`, since it is a simple yet popular dataset that is frequently used for testing and experimenting with algorithms. Also, we will only use two features from the **Iris flower** dataset for visualization purposes.

We will assign the *petal length* and *petal width* of the 150 flower samples to the feature
matrix $$X$$ and the corresponding class labels of the flower species to the vector $$y$$ :

```py
>>> import numpy as np
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> X = iris.data[:, [2, 3]]
>>> y = iris.target
>>> np.unique(y)
array([0, 1, 2])
>>>
```

If we executed `np.unique(y)` to return the different class labels stored in `iris.target `, we would see that the Iris flower class names, *Iris-Setosa*, *Iris-Versicolor*, and *Iris-Virginica*, are already stored as integers $$( 0 , 1 , 2 )$$, which is recommended for the optimal performance of many machine learning libraries.

To evaluate how well a trained model performs on unseen data, we will further split the dataset into separate training and test datasets.

```py
>>> from sklearn.cross_validation import train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

Using the `train_test_split` function from scikit-learn's `cross_validation` module, we randomly split the $$X$$ and $$y$$ arrays into 30 percent test data (45 samples) and 70 percent training data (105 samples).

Many machine learning and optimization algorithms also require feature scaling for optimal performance, as we remember from the **gradient descent** example in *Section 2, Training Machine Learning Algorithms for Classification*. Here, we will standardize the features using the `StandardScaler` class from scikit-learn's `preprocessing` module:

```py
>>> from sklearn.preprocessing import StandardScaler
>>> sc = StandardScaler()
>>> sc.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
>>> X_train_std = sc.transform(X_train)
>>> X_test_std = sc.transform(X_test)
```

Using the preceding code, we loaded the `StandardScaler` class from the `preprocessing` module and initialized a new `StandardScaler` object that we assigned to the variable `sc` . Using the `fit `method, `StandardScaler` estimated the parameters $$\mu$$ (sample mean) and $$\sigma$$ (standard deviation) for each feature dimension from the training data. By calling the transform method, we then standardized the training data using those estimated parameters $$\mu$$ and  $$\sigma$$ . Note that we used the same scaling parameters to standardize the test set so that both the values in the training and test dataset are comparable to each other.

Having standardized the training data, we can now train a perceptron model. Most algorithms in scikit-learn already support multiclass classification by default via the **One-vs.-Rest (OvR)** method, which allows us to feed the three flower classes to the perceptron all at once. The code is as follows:

```py
>>> from sklearn.linear_model import Perceptron
>>> ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
>>> ppn.fit(X_train_std, y_train)
Perceptron(alpha=0.0001, class_weight=None, eta0=0.1, fit_intercept=True,
      n_iter=40, n_jobs=1, penalty=None, random_state=0, shuffle=True,
      verbose=0, warm_start=False)
>>> 
```

The scikit-learn interface reminds us of our perceptron implementation in *Section 2, Training Machine Learning Algorithms for Classification*: after loading the `Perceptron` class from the `linear_model` module, we initialized a new `Perceptron` object and trained the model via the `fit` method. Here, the model parameter `eta0` is equivalent to the learning rate `eta` that we used in our own perceptron implementation, and the parameter `n_iter` defines the number of epochs (passes over the training set). As we remember from *Section 2, Training Machine Learning Algorithms for Classification*, finding an appropriate learning rate requires some experimentation. If the learning rate is too large, the algorithm will overshoot the global cost minimum. If the learning rate is too small, the algorithm requires more epochs until convergence, which can make the learning slow—especially for large datasets. Also, we used the `random_state` parameter for reproducibility of the initial shuffling of the training dataset after each epoch.

Having trained a model in scikit-learn, we can make predictions via the `predict` method, just like in our own perceptron implementation in *Section 2, Training Machine Learning Algorithms for Classification*. The code is as follows:

```py
>>> y_pred = ppn.predict(X_test_std)
>>> print('Misclassified samples: %d' % (y_test != y_pred).sum())
Misclassified samples: 4
>>> 
```

On executing the preceding code, we see that the perceptron misclassifies 4 out of the 45 flower samples. Thus, the misclassification error on the test dataset is 0.089 or 8.9 percent ( $$4 / 45 \approx 0.089$$ ) .

> Instead of the misclassification error, many machine learning
practitioners report the classification accuracy of a model, which is
simply calculated as follows:

1 - misclassification error = 0.911 or 91.1 percent.

Scikit-learn also implements a large variety of different performance metrics that are
available via the `metrics` module. For example, we can calculate the classification
accuracy of the perceptron on the test set as follows:

```py
>>> from sklearn.metrics import accuracy_score
>>> print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
Accuracy: 0.91
>>> 
```

Here, `y_test` are the true class labels and `y_pred` are the class labels that we predicted previously.

Finally, we can use our `plot_decision_regions` function from Section 2, Training Machine Learning Algorithms for Classification, to plot the decision regions of our newly trained perceptron model and visualize how well it separates the different flower samples. However, let's add a small modification to highlight the samples from the test dataset via small circles:

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
pl.xlabel('petal length [standardized]')
pl.ylabel('petal width [standardized]')
pl.legend(loc='upper left')
pl.show()
```

As we can see in the resulting plot, the three flower classes cannot be perfectly separated by a linear decision boundaries:

![trained perceptron](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/trained_perceptron.png)

We remember from our discussion in *Section 2, Training Machine Learning Algorithms for Classification*, that the perceptron algorithm never converges on datasets that aren't perfectly linearly separable, which is why the use of the perceptron algorithm is typically not recommended in practice. In the following sections, we will look at more powerful linear classifiers that converge to a cost minimum even if the classes are not perfectly linearly separable.

### 3.4 Modeling class probabilities via logistic regression
Although the perceptron rule offers a nice and easygoing introduction to machine learning algorithms for classification, its biggest disadvantage is that it never converges if the classes are not perfectly linearly separable. The classification task in the previous section would be an example of such a scenario. Intuitively, we can think of the reason as the weights are continuously being updated since there is always at least one misclassified sample present in each epoch. Of course, you can change the learning rate and increase the number of epochs, but be warned that the perceptron will never converge on this dataset. To make better use of our time, we will now take a look at another simple yet more powerful algorithm for linear and binary classification problems: logistic regression. Note that, in spite of its name, [**logistic regression**](https://en.wikipedia.org/wiki/Logistic_regression) is a model for classification, not regression.

#### 3.4.1 Logistic regression intuition and conditional probabilities
Logistic regression is a classification model that is very easy to implement but performs very well on linearly separable classes. It is one of the most widely used algorithms for classification in industry. Similar to the perceptron and Adaline, the logistic regression model in this section is also a linear model for binary classification that can be extended to multiclass classification via the OvR technique.

To explain the idea behind logistic regression as a probabilistic model, let's first introduce the **odds ratio**, which is the odds in favor of a particular event. The odds ratio can be written as $$\frac{p}{1-p}$$ , where $$p$$ stands for the probability of the positive event. The term positive event does not necessarily mean good, but refers to the event that we want to predict, for example, the probability that a patient has a certain disease; we can think of the positive event as class label $$y = 1$$ . We can then further define the **logit** function, which is simply the logarithm of the odds ratio (log-odds):

$$logit(p)=\log(\frac{p}{1-p})$$

The logit function takes input values in the range 0 to 1 and transforms them to values over the entire real number range, which we can use to express a linear relationship between feature values and the log-odds:

$$logit ( p ( y = 1\vert x ) ) = w_0 x_0 + w_1 x_1 + \cdots+w_m x_m = \sum w_mx_m = w^Tx$$

Here, $$p(y=1\vert x)$$ is the conditional probability that a particular sample belongs to class 1 given its features $$x$$.

Now what we are actually interested in is predicting the probability that a certain sample belongs to a particular class, which is the inverse form of the *logit* function. It is also called the logistic function, sometimes simply abbreviated as *sigmoid* function due to its characteristic S-shape.

Here, $$z$$ is the net input, that is, the linear combination of weights and sample features and can be calculated as $$z= w_0 x_0 + w_1 x_1 + \cdots+w_m x_m = w^Tx$$.

Now let's simply plot the sigmoid function for some values in the range -7 to 7 to see what it looks like:

```py
import matplotlib.pyplot as pl
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
pl.plot(z, phi_z)
pl.axvline(0.0, color='k')
pl.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
pl.axhline(y=0.5, ls='dotted', color='k')
pl.yticks([0.0, 0.5, 1.0])
pl.ylim(-0.1, 1.1)
pl.xlabel('z')
pl.ylabel('$$\phi (z)$$')
pl.title('sigmoid function')
pl.show()
```

As a result of executing the previous code example, we should now see the **S-shaped**
(sigmoidal) curve:

![sigmoid function](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/sigmoid_function.png)

We can see that $$\phi ( z )$$ approaches 1 if z goes towards infinity $$( z \to \infty )$$, since $$e^{-z}$$
becomes very small for large values of z. Similarly, $$\phi ( z )$$ goes towards 0 for $$z \to -\infty$$ as the result of an increasingly large denominator. Thus, we conclude that this sigmoid function takes real number values as input and transforms them to values in the range $$[0, 1]$$ with an intercept at $$\phi ( z )=0.5$$ .

To build some intuition for the logistic regression model, we can relate it to our previous Adaline implementation in *Section 2, Training Machine Learning Algorithms for Classification*. In Adaline, we used the identity function $$\phi ( z )=z$$  as the activation function. In logistic regression, this activation function simply becomes the sigmoid function that we defined earlier, which is illustrated in the following figure:

![logit_activation_function](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/logit_activation_function.png)

The output of the sigmoid function is then interpreted as the probability of particular sample belonging to class 1 $$\phi(z)=P(y=1\vert x;w)$$ , given its features $$x$$ parameterized by the weights $$w$$. For example, if we compute $$\phi(z)=0.8$$ for a particular flower sample, it means that the chance that this sample is an Iris-Versicolor flower is 80 percent. Similarly, the probability that this flower is an Iris-Setosa flower can be calculated as
$$P ( y = 0 \vert  x ; w ) = 1 − P ( y = 0 \vert  x ; w ) = 0.2$$ or 20 percent. The predicted probability can then
simply be converted into a binary outcome via a quantizer (unit step function):

$$\hat{y}=\left\{\begin{aligned}1&\quad if\quad\phi(z)\ge0.5\\-1&\quad otherwise\end{aligned}\right.$$
If we look at the preceding sigmoid plot, this is equivalent to the following:
$$\hat{y}=\left\{\begin{aligned}1&\quad if \quad z\ge0\\-1&\quad otherwise\end{aligned}\right.$$

In fact, there are many applications where we are not only interested in the predicted class labels, but where estimating the class-membership probability is particularly useful. Logistic regression is used in weather forecasting, for example, to not only predict if it will rain on a particular day but also to report the chance of rain. Similarly, logistic regression can be used to predict the chance that a patient has a particular disease given certain symptoms, which is why logistic regression enjoys wide popularity in the field of medicine.

#### 3.4.2 Learning the weights of the logistic cost function
You learned how we could use the logistic regression model to predict probabilities and class labels. Now let's briefly talk about the parameters of the model, for example, weights w. In the previous chapter, we defined the sum-squared-error cost function:

$$J(w)=\frac{1}{2}\sum_i\left(y^{(i)}-\phi(z^{(i)})\right)^2$$

We minimized this in order to learn the weights w for our Adaline classification model. To explain how we can derive the cost function for logistic regression, let's first define the likelihood $$L$$ that we want to maximize when we build a logistic regression model, assuming that the individual samples in our dataset are independent of one another. The formula is as follows:

$$L(w)=P(y\vert x;w)=\Pi_{i=0}^n P(y^{(i)}\vert x^{(i)};w)=\Pi_{i=0}^n\left(\phi(z^{(i)})\right)^{y^{(i)}}\left(1-\phi(z^{(i)})\right)^{1-y^{(i)}}$$

In practice, it is easier to maximize the (natural) log of this equation, which is called
the log-likelihood function:

$$l ( w ) = \log L ( w ) = \sum_{i=0}^n y^{(i)}\log\phi(z^{(i)})+(1-y^{(i)})\log(1-\phi(z^{(i)})$$

Firstly, applying the log function reduces the potential for numerical underflow, which can occur if the likelihoods are very small. Secondly, we can convert the product of factors into a summation of factors, which makes it easier to obtain the derivative of this function via the addition trick, as you may remember from calculus.

Now we could use an optimization algorithm such as gradient ascent to maximize this log-likelihood function. Alternatively, let's rewrite the log-likelihood as a cost function $$J$$ that can be minimized using gradient descent as in *Section 2, Training Machine Learning Algorithms for Classification*:

$$J ( w ) = \sum_{i=0}^n -y^{(i)}\log\phi(z^{(i)})-(1-y^{(i)})\log(1-\phi(z^{(i)})$$

To get a better grasp on this cost function, let's take a look at the cost that we calculate for one single-sample instance:

$$J ( \phi ( z ) , y; w ) = − y \log ( \phi ( z ) ) − ( 1 − y ) \log ( 1 − \phi ( z ) )$$

Looking at the preceding equation, we can see that the first term becomes zero if
$$y = 0 $$, and the second term becomes zero if $$y = 1$$ , respectively:

$$J ( \phi ( z ) , y; w ) = \left\{\begin{aligned}−  \log ( \phi ( z ) )\quad&if\quad y=1\\-\log ( 1 − \phi ( z ) )\quad&if\quad y=0\end{aligned}\right.$$

The following plot illustrates the cost for the classification of a single-sample instance
for different values of $$\phi ( z )$$:

![likelihood phi](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/likelihood_phi.png)

We can see that the cost approaches 0 (plain blue line) if we correctly predict that a sample belongs to class 1. Similarly, we can see on the y axis that the cost also approaches 0 if we correctly predict y = 0 (dashed line). However, if the prediction is wrong, the cost goes towards infinity. The moral is that we penalize wrong predictions with an increasingly larger cost.

#### 3.4.3 Training a logistic regression model with scikit-learn
If we were to implement logistic regression ourselves, we could simply substitute the cost function $$J$$ in our Adaline implementation from *Section 2, Training Machine Learning Algorithms for Classification*, by the new cost function:

$$J ( w ) = -\sum_{i=0}^n y^{(i)}\log\phi(z^{(i)})+(1-y^{(i)})\log(1-\phi(z^{(i)})$$

This would compute the cost of classifying all training samples per epoch and we would end up with a working logistic regression model. However, since scikit-learn implements a highly optimized version of logistic regression that also supports multiclass settings off-the-shelf, we will skip the implementation and use the `sklearn.linear_model.LogisticRegression` class as well as the familiar `fit` method to train the model on the standardized flower training dataset:

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105,150))
pl.xlabel('petal length [standardized]')
pl.ylabel('petal width [standardized]')
pl.legend(loc='upper left')
pl.show()
```

After fitting the model on the training data, we plotted the decision regions, training samples and test samples, as shown here:

![logit regression](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/logit_regression.png)

Looking at the preceding code that we used to train the LogisticRegression model, you might now be wondering, "What is this mysterious parameter C ?" We will get to this in a second, but let's briefly go over the concept of overfitting and regularization in the next subsection first.

Furthermore, we can predict the class-membership probability of the samples via the `predict_proba method`. For example, we can predict the probabilities of the first *Iris-Setosa* sample:

```py
lr.predict_proba(X_test_std[0,:])
array([[0.000, 0.063, 0.937]])
```

The preceding array tells us that the model predicts a chance of 93.7 percent that the sample belongs to the Iris-Virginica class, and a 6.3 percent chance that the sample is a Iris-Versicolor flower.

We can show that the weight update in logistic regression via gradient descent is indeed equal to the equation that we used in Adaline in *Section 2, Training Machine Learning Algorithms for Classification*. Let's start by calculating the partial derivative of the log-likelihood function with respect to the $$j$$th weight:

$$\frac{\partial}{\partial w_j}\phi(z)=\frac{\partial}{\partial z}\frac{1}{1+e^{-z}}=\frac{1}{(1+e^{-z})^2}e^{-z}\\=\frac{1}{1+e^{-z}}\left(1-\frac{1}{1+e^{-z}}\right)=\phi(z)(1-\phi(z))$$

Now we can resubstitute $$\frac{\partial}{\partial w_j}\phi(z)=\phi(z)(1-\phi(z))$$ in our first equation to obtain the following:

$$\begin{aligned}&\left(y\frac{1}{\phi(z)}-(1-y)\frac{1}{1-\phi(z)}\right)\frac{\partial}{\partial w_j}\\=&\left(y\frac{1}{\phi(z)}-(1-y)\frac{1}{1-\phi(z)}\right)\phi(z)(1-\phi(z))\frac{\partial}{\partial w_j}z\\=&\left(y(1-\phi(z))-(1-y)\phi(z)\right)x_j\\=&(y-\phi(z))x_j\end{aligned}$$

Remember that the goal is to find the weights that maximize the log-likelihood so that we would perform the update for each weight as follows:

$$w:=w_j+\eta\sum_{i=0}^n\left(y^{(i)}-\phi(z^{(i)})\right)x^{(i)}$$

Since we update all weights simultaneously, we can write the general update rule as follows:

$$w=w+\Delta w$$

We define $$\Delta w$$ as follows:

$$\Delta w=\eta\nabla l(w)$$

Since maximizing the log-likelihood is equal to minimizing the cost function $$J$$ that we defined earlier, we can write the gradient descent update rule as follows:

$$\Delta w_j=-\eta\frac{\partial J}{\partial w_j}=\eta\sum_{i=0}^n\left(y^{(i)}-\phi(z^{(i)})\right)x^{(i)}$$

$$w:=w+\Delta w,\quad \Delta w=-\eta J(w)$$

This is equal to the gradient descent rule in Adaline in *Section 2, Training Machine
Learning Algorithms for Classification*.

#### 3.4.4 Tackling overfitting via regularization
Overfitting is a common problem in machine learning, where a model performs well on training data but does not generalize well to unseen data (test data). If a model suffers from overfitting, we also say that the model has a high variance, which can be caused by having too many parameters that lead to a model that is too complex given the underlying data. Similarly, our model can also suffer from **underfitting** (high bias), which means that our model is not complex enough to capture the pattern in the training data well and therefore also suffers from low performance on unseen data.

Although we have only encountered linear models for classification so far, the problem of overfitting and underfitting can be best illustrated by using a more complex, nonlinear decision boundary as shown in the following figure:

![overfitting](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/overfitting.png)

> Variance measures the consistency (or variability) of the model
prediction for a particular sample instance if we would retrain
the model multiple times, for example, on different subsets of
the training dataset. We can say that the model is sensitive to the
randomness in the training data. In contrast, bias measures how far
off the predictions are from the correct values in general if we rebuild
the model multiple times on different training datasets; bias is the
measure of the systematic error that is not due to randomness.

One way of finding a good bias-variance tradeoff is to tune the complexity of the model via regularization. Regularization is a very useful method to handle collinearity (high correlation among features), filter out noise from data, and eventually prevent overfitting. The concept behind regularization is to introduce additional information (bias) to penalize extreme parameter weights. The most common form of regularization is the so-called **L2 regularization** (sometimes also called **L2** shrinkage or weight decay), which can be written as follows:

$$\frac{\lambda}{2}\vert \vert w\vert \vert ^2=\frac{\lambda}{2}\sum_{j=0}^m w_j^2$$

> Regularization is another reason why feature scaling such as
standardization is important. For regularization to work properly,
we need to ensure that all our features are on comparable scales.

In order to apply regularization, we just need to add the regularization term to the cost function that we defined for logistic regression to shrink the weights:

$$J ( w ) = \left[\sum_{i=0}^n -y^{(i)}\log\phi(z^{(i)})-(1-y^{(i)})\log(1-\phi(z^{(i)})\right]+\frac{\lambda}{2}\vert \vert w\vert \vert ^2$$

Via the regularization parameter $$\lambda$$, we can then control how well we fit the training data while keeping the weights small. By increasing the value of $$\lambda$$ , we increase the regularization strength.

The parameter C that is implemented for the LogisticRegression class in scikit-learn comes from a convention in support vector machines, which will be the topic of the next section. $$C$$ is directly related to the regularization parameter $$\lambda$$ , which is its inverse:

$$C=\frac{1}{\lambda}$$

So we can rewrite the regularized cost function of logistic regression as follows:

$$J ( w ) = C\left[\sum_{i=0}^n -y^{(i)}\log\phi(z^{(i)})-(1-y^{(i)})\log(1-\phi(z^{(i)})\right]+\frac{1}{2}\vert \vert w\vert \vert ^2$$

Consequently, decreasing the value of the inverse regularization parameter C means that we are increasing the regularization strength, which we can visualize by plotting the L2 regularization path for the two weight coefficients:

```py
import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
pl.plot(params, weights[:, 0], label='petal length')
pl.plot(params, weights[:, 1], linestyle='--', label='petal width')
pl.ylabel('weight coefficient')
pl.xlabel('C')
pl.legend(loc='upper left')
pl.xscale('log')
pl.show()
```

By executing the preceding code, we fitted ten logistic regression models with different values for the inverse-regularization parameter C . For the purposes of illustration, we only collected the weight coefficients of the class 2 vs. all classifier. Remember that we are using the OvR technique for multiclass classification.

As we can see in the resulting plot, the weight coefficients shrink if we decrease the parameter $$C$$, that is, if we increase the regularization strength:

![L2_regulation](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/L2_regulation.png)

### 3.5 Maximum margin classification with support vector machines
Another powerful and widely used learning algorithm is the **support vector machine (SVM)**, which can be considered as an extension of the perceptron. Using the perceptron algorithm, we minimized misclassification errors. However, in SVMs, our optimization objective is to maximize the **margin**. The margin is defined as the distance between the separating hyperplane (decision boundary) and the training samples that are closest to this hyperplane, which are the so-called **support vectors**. This is illustrated in the following figure:

![SV](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/SV.png)

#### 3.5.1 Maximum margin intuition
The rationale behind having decision boundaries with large margins is that they tend to have a lower generalization error whereas models with small margins are more prone to overfitting. To get an intuition for the margin maximization, let's take a closer look at those *positive and negative* hyperplanes that are parallel to the decision boundary, which can be expressed as follows:

$$\begin{aligned}w_0+w^Tx_{pos}=1\quad(1)\\w_0+w^Tx_{neg}=-1\quad(2)\end{aligned}$$

If we subtract those two linear equations (1) and (2) from each other, we get:

$$w^T ( x_{pos}− x_{neg} ) = 2$$

We can normalize this by the length of the vector w, which is defined as follows:

$$\vert \vert w\vert \vert =\sqrt{\sum_{j=1}^m w_j^2}$$

So we arrive at the following equation:

$$\frac{w^T ( x_{pos}− x_{neg} )}{\vert \vert w\vert \vert }=\frac{2}{\vert \vert w\vert \vert }$$

The left side of the preceding equation can then be interpreted as the distance between the positive and negative hyperplane, which is the so-called margin that we want to maximize.

Now the objective function of the SVM becomes the maximization of this margin by maximizing $$\frac{2}{\vert\vert w\vert\vert}$$ under the constraint that the samples are classified correctly, which can be written as follows:

$$\begin{aligned}w_0 + w_T x^{( i )}\ge1\quad &if\quad y ( i ) = 1\\w_0 + w_T x^{( i )} <- 1\quad &if\quad y ( i ) = -1\end{aligned}$$

These two equations basically say that all negative samples should fall on one side of the negative hyperplane, whereas all the positive samples should fall behind the positive hyperplane. This can also be written more compactly as follows:

$$y^{( i )}( w_0 + w^T x ^{( i ) }\ge 1 \forall \quad i$$

In practice, though, it is easier to minimize the reciprocal term solved by quadratic programming. However, a detailed discussion about quadratic programming is beyond the scope of this book, but if you are interested, you can learn more about **Support Vector Machines (SVM)** in Vladimir Vapnik's *The Nature of Statistical Learning Theory,* Springer Science & Business Media, or Chris J.C. Burges' excellent explanation in *A Tutorial on Support Vector Machines for Pattern Recognition* (Data mining and knowledge discovery, 2(2):121–167, 1998).

#### 3.5.2 Dealing with the nonlinearly separable case using slack variables
Although we don't want to dive much deeper into the more involved mathematical concepts behind the margin classification, let's briefly mention the slack variable $$\xi$$ . It was introduced by Vladimir Vapnik in 1995 and led to the so-called soft-margin classification. The motivation for introducing the slack variable $$\xi$$ was that the linear constraints need to be relaxed for nonlinearly separable data to allow convergence of the optimization in the presence of misclassifications under the appropriate cost penalization.

The positive-values slack variable is simply added to the linear constraints:

$$\begin{aligned}w^Tx^{(i)}\ge1\quad &if\quad y^{(i)}=1-\xi^{(i)}\\w^Tx^{(i)}<1\quad &if \quad y^{(i)}=1+\xi^{(i)}\end{aligned}$$

So the new objective to be minimized (subject to the preceding constraints) becomes:

$$\frac{1}{2}\vert \vert w\vert \vert ^2+C\left(\sum_i\xi^{(i)}\right)$$

Using the variable $$C$$ , we can then control the penalty for misclassification. Large values of $$C$$ correspond to large error penalties whereas we are less strict about misclassification errors if we choose smaller values for $$C$$ . We can then we use the parameter $$C$$ to control the width of the margin and therefore tune the bias-variance trade-off as illustrated in the following figure:

![slack_variable](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/slack_variable.png)

This concept is related to regularization, which we discussed previously in the context of regularized regression where increasing the value of $$C$$ increases the bias and lowers the variance of the model.

Now that we learned the basic concepts behind the linear SVM, let's train a SVM model to classify the different flowers in our Iris dataset:

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105,150))
pl.xlabel('petal length [standardized]')
pl.ylabel('petal width [standardized]')
pl.legend(loc='upper left')
pl.show()
```

The decision regions of the SVM visualized after executing the preceding code example are shown in the following plot:

![svm](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/svm.png)

> **Logistic regression versus SVM**
In practical classification tasks, linear logistic regression and
linear SVMs often yield very similar results. Logistic regression
tries to maximize the conditional likelihoods of the training
data, which makes it more prone to outliers than SVMs. The
SVMs mostly care about the points that are closest to the
decision boundary (support vectors). On the other hand, logistic
regression has the advantage that it is a simpler model that can
be implemented more easily. Furthermore, logistic regression
models can be easily updated, which is attractive when working
with streaming data.

#### 3.5.3 Alternative implementations in scikit-learn
The Perceptron and LogisticRegression classes that we used in the previous sections via scikit-learn make use of the `LIBLINEAR` library, which is a highly optimized `C/C++` library developed at the National Taiwan University ( <http://www.csie.ntu.edu.tw/~cjlin/liblinear/> ). Similarly, the SVC class that we used to train an SVM makes use of LIBSVM, which is an equivalent `C/C++` library specialized for SVMs ( <http://www.csie.ntu.edu.tw/~cjlin/libsvm/> ).

The advantage of using `LIBLINEAR` and `LIBSVM` over native Python implementations is that they allow an extremely quick training of large amounts of linear classifiers. However, sometimes our datasets are too large to fit into computer memory. Thus, `scikit-learn` also offers alternative implementations via the `SGDClassifier` class, which also supports online learning via the partial_fit method. The concept behind the `SGDClassifier` class is similar to the stochastic gradient algorithm that we implemented in *Scetion 2, Training Machine Learning Algorithms for Classification*, for Adaline. We could initialize the stochastic gradient descent version of the perceptron, logistic regression, and support vector machine with default parameters as follows:

```py
>>> from sklearn.linear_model import SGDClassifier
>>> ppn = SGDClassifier(loss='perceptron')
>>> lr = SGDClassifier(loss='log')
>>> svm = SGDClassifier(loss='hinge')
```

### 3.6 Solving nonlinear problems using a kernel SVM
Another reason why SVMs enjoy high popularity among machine learning practitioners is that they can be easily kernelized to solve nonlinear classification problems. Before we discuss the main concept behind kernel SVM, let's first define and create a sample dataset to see how such a nonlinear classification problem may look.

Using the following code, we will create a simple dataset that has the form of an XOR gate using the `logical_xor` function from NumPy, where 100 samples will be assigned the class label 1 and 100 samples will be assigned the class label -1, respectively:

```py
import numpy as np
import matplotlib.pyplot as pl

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

pl.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
pl.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
pl.ylim(-3.0)
pl.legend()
pl.show()
```

After executing the code, we will have an XOR dataset with random noise, as shown in the following figure:

![XOR](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/XOR_dataset.png)

Obviously, we would not be able to separate samples from the positive and negative class very well using a linear hyperplane as the decision boundary via the linear logistic regression or linear SVM model that we discussed in earlier sections.

The basic idea behind kernel methods to deal with such linearly inseparable data is to create nonlinear combinations of the original features to project them onto a higher dimensional space via a mapping function $$\phi ( \cdot)$$ where it becomes linearly separable. As shown in the next figure, we can transform a two-dimensional dataset onto a new three-dimensional feature space where the classes become separable via the following projection:

$$\phi ( x_1 , x_2 ) = ( z_1 , z_2 , z_3 ) = ( x _1 , x_2 , x_1^2 + x_2^2 )$$

This allows us to separate the two classes shown in the plot via a linear hyperplane that becomes a nonlinear decision boundary if we project it back onto the original feature space:

![hyperplane](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/hyperplane.png)

#### 3.6.1 Using the kernel trick to find separating hyperplanes in higher dimensional space
To solve a nonlinear problem using an SVM, we transform the training data onto a higher dimensional feature space via a mapping function $$\phi ( \cdot)$$ and train a linear SVM model to classify the data in this new feature space. Then we can use the same mapping function $$\phi ( \cdot)$$ to transform new, unseen data to classify it using the linear SVM model.

However, one problem with this mapping approach is that the construction of the new features is computationally very expensive, especially if we are dealing with high dimensional data. This is where the so-called kernel trick comes into play. Although we didn't go into much detail about how to solve the quadratic programming task to train an SVM, in practice all we need is to replace the dot product $$x ^{( i ) T} x ^{( j )}$$ by $$\phi ( x ^{( i )} )^T \phi ( x ^{( j )} )$$ . In order to save the expensive step of calculating this dot product between two points explicitly, we define a so-called kernel function:

$$k(x ^{( i ) },  x ^{( j )})=\phi ( x ^{( i )} )^T \phi ( x ^{( j )} )$$

One of the most widely used kernels is the **Radial Basis Function** kernel
(**RBF** kernel) or Gaussian kernel:

$$k(x ^{( i ) },  x ^{( j )})=\exp\left(-\frac{\vert \vert x ^{( i ) },  x ^{( j )}\vert \vert ^2}{2\sigma^2}\right)$$

This is often simplified to:

$$k(x ^{( i ) },  x ^{( j )})=\exp\left(-\gamma\vert \vert x ^{( i ) },  x ^{( j )}\vert \vert ^2\right)$$

Here, $$\gamma=\frac{1}{2\sigma^2}$$  is a free parameter that is to be optimized.

Roughly speaking, the term *kernel* can be interpreted as a *similarity function* between a pair of samples. The minus sign inverts the distance measure into a similarity score and, due to the exponential term, the resulting similarity score will fall into a range between 1 (for exactly similar samples) and 0 (for very dissimilar samples).

Now that we defined the big picture behind the kernel trick, let's see if we can train a kernel SVM that is able to draw a nonlinear decision boundary that separates the XOR data well. Here, we simply use the `SVC` class from `scikit-learn` that we imported earlier and replace the parameter `kernel='linear'` with `kernel='rbf'` :

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

svm = SVC(kernel='rbf', C=10.0, gamma=0.10, random_state=0)
svm.fit(X_xor, y_xor)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

plot_decision_regions(X_xor, y_xor, classifier=svm)
pl.xlabel('petal length ')
pl.ylabel('petal width ')
pl.legend(loc='upper left')
pl.show()
```

As we can see in the resulting plot, the kernel SVM separates the XOR data relatively well:

![svm kernel](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/svm_kernel.png)

The $$\gamma$$ parameter, which we set to gamma=0.1 , can be understood as a cut-off parameter for the Gaussian sphere. If we increase the value for  $$\gamma$$ , we increase the influence or reach of the training samples, which leads to a softer decision boundary. To get a better intuition for  $$\gamma$$ , let's apply RBF kernel SVM to our Iris flower dataset:

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svm = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105,150))
pl.xlabel('petal length [standardized]')
pl.ylabel('petal width [standardized]')
pl.legend(loc='upper left')
pl.show()
```

Since we chose a relatively small value for $$\gamma$$ , the resulting decision boundary of the
RBF kernel SVM model will be relatively soft, as shown in the following figure:

![svm_iris](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/svm_iris.png)

Now let's increase the value of $$\gamma$$ and observe the effect on the decision boundary:

```py
svm = SVC(kernel='rbf', C=1.0, gamma=100.0, random_state=0)
```

In the resulting plot, we can now see that the decision boundary around the classes 0 and 1 is much tighter using a relatively large value of $$\gamma$$  :

![svm iris gamma 100](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/svm_iris100.png)

Although the model fits the training dataset very well, such a classifier will likely have a high generalization error on unseen data, which illustrates that the optimization of $$\gamma$$ also plays an important role in controlling overfitting.

### 3.7 Decision tree learning
**Decision tree** classifiers are attractive models if we care about interpretability. Like the name *decision tree* suggests, we can think of this model as breaking down our data by making decisions based on asking a series of questions.

Let's consider the following example where we use a decision tree to decide upon an activity on a particular day:

![decision_tree_illustration](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/decision_tree_illustration.png)

Based on the features in our training set, the decision tree model learns a series of questions to infer the class labels of the samples. Although the preceding figure illustrated the concept of a decision tree based on categorical variables, the same concept applies if our features. This also works if our features are real numbers like in the Iris dataset. For example, we could simply define a cut-off value along the **sepal width** feature axis and ask a binary question "sepal width ≥ 2.8 cm?"

Using the decision algorithm, we start at the tree root and split the data on the feature that results in the largest **information gain (IG)**, which will be explained in more detail in the following section. In an iterative process, we can then repeat this splitting procedure at each child node until the leaves are pure. This means that the samples at each node all belong to the same class. In practice, this can result in a very deep tree with many nodes, which can easily lead to overfitting. Thus, we typically want to *prune* the tree by setting a limit for the maximal depth of the tree.

#### 3.7.1 Maximizing information gain – getting the most bang for the buck
In order to split the nodes at the most informative features, we need to define an objective function that we want to optimize via the tree learning algorithm. Here, our objective function is to maximize the information gain at each split, which we define as follows:

$$IG ( D_p , f ) = I ( D_p ) − \sum_{j=1}^m\frac{N_j}{N_p} I ( D_j )$$

Here, $$f$$ is the feature to perform the split, $$D_p$$ and $$D_j$$ are the dataset of the parent and $$j$$th child node, $$I$$ is our impurity measure, $$N_p$$ is the total number of samples at the parent node, and $$N_j$$ is the number of samples in the jth child node. As we can see, the information gain is simply the difference between the impurity of the parent node and the sum of the child node impurities—the lower the impurity of the child nodes, the larger the information gain. However, for simplicity and to reduce the combinatorial search space, most libraries (including scikit-learn) implement binary decision trees. This means that each parent node is split into two child nodes, $$D_{left}$$ and $$D_{right}$$ :

$$IG ( D_p , a ) = I ( D_p ) − \frac{N_{left}}{N_p}I ( D_{left} ) −\frac{N_{right}}{N_p}I ( D_{right} )$$

Now, the three impurity measures or splitting criteria that are commonly used in binary decision trees are **Gini index ( $$I_G$$ )**, **entropy ( $$I_H$$ )**, and the **classification error  ($$I_E$$ )**. Let's start with the definition of entropy for all non-empty classes $$( p ( i \vert t ) \ne 0 )$$:

$$I_H ( t ) = − \sum_{i=1}^c p ( i \vert t ) \log_2 p ( i \vert t )$$

Here, $$p ( i \vert t )$$ is the proportion of the samples that belongs to class c for a particular node $$t$$. The entropy is therefore 0 if all samples at a node belong to the same class, and the entropy is maximal if we have a uniform class distribution. For example, in a binary class setting, the entropy is 0 if $$p ( i = 1\vert t ) = 1$$ or $$p ( i = 0 \vert t ) = 0$$ . If the classes are distributed uniformly with $$p ( i = 1\vert t ) = 0.5$$ and $$p ( i = 0 \vert t ) = 0.5$$ , the entropy is 1. Therefore, we can say that the entropy criterion attempts to maximize the mutual information in the tree.

Intuitively, the Gini index can be understood as a criterion to minimize the probability of misclassification:

$$I_G ( t ) = \sum_{i=1}^c p ( i \vert t ) ( − p ( i \vert  t ) ) = 1 − \sum_{i=1}^c p ( i \vert  t )^2$$

Similar to entropy, the Gini index is maximal if the classes are perfectly mixed, for example, in a binary class setting ( $$c = 2$$ ):

$$1-\sum_{i=1}^c 0.5^2=0.5$$

However, in practice both the Gini index and entropy typically yield very similar results and it is often not worth spending much time on evaluating trees using different impurity criteria rather than experimenting with different pruning cut-offs.

Another impurity measure is the classification error:

$$I_E = 1 − \max \{ p ( i \vert  t ) \}$$

This is a useful criterion for pruning but not recommended for growing a decision tree, since it is less sensitive to changes in the class probabilities of the nodes.

For a more visual comparison of the three different impurity criteria that we discussed previously, let's plot the impurity indices for the probability range $$[0, 1]$$ for class 1. Note that we will also add in a scaled version of the entropy (entropy/2) to observe that the Gini index is an intermediate measure between entropy and the classification error. The code is as follows:

```py
import matplotlib.pyplot as pl
import numpy as np

def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = pl.figure()
ax = pl.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'], ['-', '-', '--', '-.'], ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
pl.ylim([0, 1.1])
pl.xlabel('p(i=1)')
pl.ylabel('Impurity Index')
pl.show()
```

The plot produced by the preceding code example is as follows:

![impurity](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/impurity_measure.png)

#### 3.7.2 Building a decision tree
Decision trees can build complex decision boundaries by dividing the feature space into rectangles. However, we have to be careful since the deeper the decision tree, the more complex the decision boundary becomes, which can easily result in overfitting. Using scikit-learn, we will now train a decision tree with a maximum depth of 3 using entropy as a criterion for impurity. Although feature scaling may be desired for visualization purposes, note that feature scaling is not a requirement for decision tree algorithms. The code is as follows:

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined, y=y_combined, classifier=tree, test_idx=range(105,150))
pl.xlabel('petal length ')
pl.ylabel('petal width ')
pl.legend(loc='upper left')
pl.show()
```

After executing the preceding code example, we get the typical axis-parallel decision boundaries of the decision tree:

![decision tree](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/decision_tree3.png)

A nice feature in scikit-learn is that it allows us to export the decision tree as a `.dot` file after training, which we can visualize using the GraphViz program. This program is freely available at <http://www.graphviz.org> and supported by Linux, Windows, and Mac OS X.

First, we create the .dot file via scikit-learn using the export_graphviz function from the tree submodule, as follows:

```py
>>> from sklearn.tree import export_graphviz
>>> export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])
```

After we have installed GraphViz on our computer, we can convert the tree.dot file into a PNG file by executing the following command from the command line in the location where we saved the `tree.dot` file:

```bash
$ dot -Tpng tree.dot -o tree.png
```

![tree](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/tree.png)

Looking at the decision tree figure that we created via GraphViz, we can now nicely trace back the splits that the decision tree determined from our training dataset. We started with 105 samples at the root and split it into two child nodes with 34 and 71 samples each using the petal with cut-off ≤ 0.75 cm. After the first split, we can see that the left child node is already pure and only contains samples from the Iris-Setosa class (entropy = 0). The further splits on the right are then used to separate the samples from the Iris-Versicolor and Iris-Virginica classes.

#### 3.7.3 Combining weak to strong learners via random forests
**Random forests** have gained huge popularity in applications of machine learning during the last decade due to their good classification performance, scalability, and ease of use. Intuitively, a random forest can be considered as an ensemble of decision trees. The idea behind *ensemble* learning is to combine **weak learners** to build a more robust model, a **strong learner**, that has a better generalization error and is less susceptible to overfitting. The random forest algorithm can be summarized in four simple steps:

> 1. Draw a random **bootstrap** sample of size n (randomly choose n samples from
the training set with replacement).
> 2. Grow a decision tree from the bootstrap sample. At each node:
>> 1. Randomly select d features without replacement.
>> 2. Split the node using the feature that provides the best split according to the objective function, for instance, by maximizing the information gain.
> 3. Repeat the steps 1 to 2 $$k$$ times.
> 4. Aggregate the prediction by each tree to assign the class label by **majority vote**. 

There is a slight modification in step 2 when we are training the individual decision trees: instead of evaluating all features to determine the best split at each node, we only consider a random subset of those.

Although random forests don't offer the same level of interpretability as decision trees, a big advantage of random forests is that we don't have to worry so much about choosing good hyperparameter values. We typically don't need to prune the random forest since the ensemble model is quite robust to noise from the individual decision trees. The only parameter that we really need to care about in practice is the number of trees $$k$$ (step 3) that we choose for the random forest. Typically, the larger the number of trees, the better the performance of the random forest classifier at the expense of an increased computational cost.

Although it is less common in practice, other hyperparameters of the random forest classifier that can be optimized, Compressing Data via Dimensionality Reduction—are the size n of the bootstrap sample (step 1) and the number of features d that is randomly chosen for each split (step 2.1), respectively. Via the sample size $$n$$ of the bootstrap sample, we control the bias-variance tradeoff of the random forest. By choosing a larger value for n, we decrease the randomness and thus the forest is more likely to overfit. On the other hand, we can reduce the degree of overfitting by choosing smaller values for $$n$$ at the expense of the model performance. In most implementations, including the `RandomForestClassifier` implementation in scikit-learn, the sample size of the bootstrap sample is chosen to be equal to the number of samples in the original training set, which usually provides a good bias-variance tradeoff. For the number of features $$d$$ at each split, we want to choose a value that is smaller than the total number of features in the training set. A reasonable default that is used in scikit-learn and other implementations is $$d = \sqrt{m}$$ , where $$m$$ is the number of features in the training set.

Conveniently, we don't have to construct the random forest classifier from individual decision trees by ourselves; there is already an implementation in scikit-learn that we can use:

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=4)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined, y=y_combined, classifier=forest, test_idx=range(105,150))
pl.xlabel('petal length ')
pl.ylabel('petal width ')
pl.legend(loc='upper left')
pl.show()
```

After executing the preceding code, we should see the decision regions formed by the ensemble of trees in the random forest, as shown in the following figure:
![decision forest](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/decision_forest.png)

Using the preceding code, we trained a random forest from 10 decision trees via the `n_estimators `parameter and used the entropy criterion as an impurity measure to split the nodes. Although we are growing a very small random forest from a very small training dataset, we used the `n_jobs` parameter for demonstration purposes, which allows us to parallelize the model training using multiple cores of our computer (here, four).

### 3.8 K-nearest neighbors – a lazy learning algorithm
The last supervised learning algorithm that we want to discuss in this section is the **k-nearest neighbor classifier (KNN)**, which is particularly interesting because it is fundamentally different from the learning algorithms that we have discussed so far.

KNN is a typical example of a **lazy learner**. It is called *lazy* not because of its apparent simplicity, but because it doesn't learn a discriminative function from the training data but memorizes the training dataset instead.

> Parametric versus nonparametric models
Machine learning algorithms can be grouped into parametric and
nonparametric models. Using parametric models, we estimate
parameters from the training dataset to learn a function that can
classify new data points without requiring the original training dataset
anymore. Typical examples of parametric models are the perceptron,
logistic regression, and the linear SVM. In contrast, nonparametric
models can't be characterized by a fixed set of parameters, and the
number of parameters grows with the training data. Two examples of
nonparametric models that we have seen so far are the decision tree
classifier/random forest and the kernel SVM.
KNN belongs to a subcategory of nonparametric models that is
described as instance-based learning. Models based on instance-based
learning are characterized by memorizing the training dataset, and lazy
learning is a special case of instance-based learning that is associated
with no (zero) cost during the learning process.

The KNN algorithm itself is fairly straightforward and can be summarized by the following steps:

> 1. Choose the number of k and a distance metric.
> 2. Find the k nearest neighbors of the sample that we want to classify.
> 3. Assign the class label by majority vote.

The following figure illustrates how a new data point  is assigned the triangle class label based on majority voting among its five nearest neighbors.

![KNN_illustration](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/KNN_illustration.png)

Based on the chosen distance metric, the KNN algorithm finds the k samples in the training dataset that are closest (most similar) to the point that we want to classify. The class label of the new data point is then determined by a majority vote among its $$k$$ nearest neighbors.

The main advantage of such a memory-based approach is that the classifier immediately adapts as we collect new training data. However, the downside is that the computational complexity for classifying new samples grows linearly with the number of samples in the training dataset in the worst-case scenario—unless the dataset has very few dimensions (features) and the algorithm has been implemented using efficient data structures such as KD-trees. J. H. Friedman, J. L. Bentley, and R. A. Finkel. An algorithm for finding best matches in logarithmic expected time. ACM Transactions on Mathematical Software (TOMS), 3(3):209–226, 1977. Furthermore, we can't discard training samples since no training step is involved. Thus, storage space can become a challenge if we are working with large datasets.

By executing the following code, we will now implement a KNN model in scikit-learn using an Euclidean distance metric:

```py
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    pl.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pl.xlim(xx1.min(), xx1.max())
    pl.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        pl.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        pl.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn, test_idx=range(105,150))
pl.xlabel('petal length [standardized]')
pl.ylabel('petal width [standardized]')
pl.legend(loc='upper left')
pl.show()
```

By specifying five neighbors in the KNN model for this dataset, we obtain a relatively smooth decision boundary, as shown in the following figure:

![knn](https://github.com/Mageluer/computational_physics_N2014301040052/raw/master/final/img/knn.png)

The *right* choice of $$k$$ is crucial to find a good balance between over- and underfitting. We also have to make sure that we choose a distance metric that is appropriate for the features in the dataset. Often, a simple Euclidean distance measure is used for real-valued samples, for example, the flowers in our Iris dataset, which have features measured in centimeters. However, if we are using a Euclidean distance measure, it is also important to standardize the data so that each feature contributes equally to the distance. The '*minkowski*' distance that we used in the previous code is just a generalization of the Euclidean and Manhattan distance that can be written as follows:

$$d(x^{(i)},x^{(i)})=\sqrt[p]{\sum_k\left\vert x^{(k)}x^{(k)}\right\vert ^p}$$

It becomes the Euclidean distance if we set the parameter $$p=2$$ or the Manhatten distance at $$p=1$$ , respectively. Many other distance metrics are available in scikit-learn and can be provided to the metric parameter. They are listed at <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html> .

> It is important to mention that KNN is very susceptible to
overfitting due to the **curse of dimensionality**. The curse of
dimensionality describes the phenomenon where the feature
space becomes increasingly sparse for an increasing number
of dimensions of a fixed-size training dataset. Intuitively, we
can think of even the closest neighbors being too far away in a
high-dimensional space to give a good estimate.

## 4 Conclusion
We learned about many different machine algorithms that are used to tackle linear and nonlinear problems. We have seen that decision trees are particularly attractive if we care about interpretability. Logistic regression is not only a useful model for online learning via stochastic gradient descent, but also allows us to predict the probability of a particular event. Although support vector machines are powerful linear models that can be extended to nonlinear problems via the kernel trick, they have many parameters that have to be tuned in order to make good predictions. In contrast, ensemble methods such as random forests don't require much parameter tuning and don't overfit so easily as decision trees, which makes it an attractive model for many practical problem domains. The K-nearest neighbor classifier offers an alternative approach to classification via lazy learning that allows us to make predictions without any model training but with a more computationally expensive prediction step.

However, even more important than the choice of an appropriate learning algorithm is the available data in our training dataset. No algorithm will be able to make good predictions without informative and discriminatory features.

## Reference
1. <https://www.gitbook.com/book/ljalphabeta/python-/details>

