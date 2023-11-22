# POPL - Milestone for project - Group 20

## Problem Statement
Sentence Autocompletion using Pyro and Python - A comparative study based on metrics like accuracy, performance and efficiency

- This project aims to compare two sentence completion techniques: a conventional PyTorch-based approach using LSTM for text generation and a probabilistic method employing Pyro. The traditional method employs LSTM to generate sentence completions from a large training dataset. 
- The Pyro-based approach introduces uncertainty by modeling the language as a word probability distribution, potentially yielding more diverse and creative completions. The project will evaluate both methods based on criteria like accuracy, speed, scalability, and explore Pyro's utility in other NLP tasks.
- This comparison has not been studied before for this problem and our study provides a comprehensive study of this comparison

## Software Architecture

### Python architecture:
- Sentence Autocompletion using Python focuses on developing a natural language processing model (NLP) for auto-completing half written sentences. It tackles the challenges posed by the News Articles and Blogs dataset, which contains special characters, non-meaningful words, and unfamiliar terms by cleaning the dataset before model training.
- To prepare the data, all words are converted to lowercase to ensure consistency. We use a custom Python class which utilizes the torch.utils.data.Dataset framework.
- We construct a Long Short-Term Memory (LSTM) network for the language model. The initialization method configures the model's architecture, specifying parameters like LSTM hidden state size, word embedding dimensions, the number of layers, and a fully connected layer for generating predictions.
- Training the sentence autocompletion model involves specifying key hyperparameters such as sequence length, batch size, learning rate, and the number of training epochs.
- To make predictions, the model accepts incomplete sentences, converts them into word indexes using a dictionary, and transforms these indexes into tensors for predicting the next word in the sequence. 

In the below diagram we have portrayed our architecture for our LSTM model
![software_architecture](https://github.com/jahnavirishikesh/POPL-G20-milestone/assets/101913971/2efd7ab1-0eb5-4e20-b7b8-5e73f2e1342f)

### Pyro Architecture
- Pyro's architecture follows a layered architecture, where each layer serves a specific purpose and interacts with the layers above and below it.
- The architecture consists of three main layers:
  1. **Data Layer** : The data layer is responsible for loading and preprocessing the dataset. In this case, the dataset argument passed to the LSTMModel constructor likely represents the training data.
  2. **Model Layer**: The model layer is responsible for defining and training the probabilistic model. The LSTMModel class represents the core of the model, encapsulating the neural network architecture and its parameters. The to_pyro_module_ function converts the model into a Pyro module, enabling it to be integrated with Pyro's inference and optimization frameworks.
  3. **Inference and Optimization Layer** : The inference and optimization layer is responsible for performing inference and training the model. The autoguide.AutoNormal class serves as the inference guide, approximating the posterior distribution of the model's parameters. The optimizer and scheduler objects manage the optimization process, updating the model's parameters based on the loss function and adjusting the learning rate over time. The SVI class orchestrates the inference and optimization process, leveraging the guide, optimizer, and scheduler to train the model effectively.

## POPL Aspects
- **Bayesian neural network (BNN)** : Bayesian neural network (BNN) is a probabilistic neural network that models uncertainty in the network's weights and biases by using probabilistic programming techniques. BNNs give the network's weights and biases probability distributions. The likelihood function indicates the likelihood of seeing the provided data in accordance with the predictions of the network. It shows how closely the output of the network matches the real data.
  
  ![pyro_model](https://github.com/jahnavirishikesh/POPL-G20-milestone/assets/101913971/1ca566a0-9d14-4eaf-89cf-4c797ae98cc9)

- **Inference with evaluators** : In Pyro, inference is the process of estimating the parameters of a probabilistic model given observed data. Pyro uses inference algorithms like Stochastic Variational Inference (SVI). Evaluators are used to evaluate various metrics during the sampling process, such as the loss function, predictive accuracy, or model complexity. This information can be used to guide the sampling process and improve the overall performance of the inference algorithm.
It takes a model, a guide, and an optimizer as arguments. The model is the probabilistic model that you want to learn the parameters of. The guide is a variational distribution that is used to approximate the posterior distribution. The optimizer is a stochastic optimization algorithm that is used to find the parameters of the guide that minimize the variational Loss.

![svi](https://github.com/jahnavirishikesh/POPL-G20-milestone/assets/101913971/38387e28-835a-480c-b215-cfb375c3003a)

- The SVI class is designed in a different way than the model.train() function that we use in python. This resulted in the need to read the source code to understand the differences between the two. What we found is that the SVI implementation passes different parameters to the functions compared to the pytorch implementation.
- Understanding the SVI, optimizer and guide was challenging as pyro documentation is small and expects us to have prior knowledge regarding the topic, creating difficulty in understanding their use and how to implement them.
- Due to pyro yet being a relatively new language finding resources online that we can refer to was not possible.

- **Probabilistic programming language** : The paradigm of probabilistic programming enables the explicit modeling of uncertainty in models. Deep probabilistic models can be created with Pyro, a probabilistic programming language (PPL). Pyro's language constructs can be used to define a wide range of probabilistic models, from simple to complicated, as it can represent any computable probability distribution.

- **Optimiser** : The optimization techniques we employ are stochastic gradient-based optimizers, which update the model parameters by utilizing the estimated gradients of the loss function. They lie within the category of gradient-based optimizers, but at each iteration they employ an estimated gradient derived from a tiny portion of the training set, rather than the precise gradient. This makes them more effective than gradient-based optimizers, particularly when dealing with big datasets.
We compared different optimisers for our implementation and settled on adam as it gave the best results. Pyro code is given below:

![optim](https://github.com/jahnavirishikesh/POPL-G20-milestone/assets/101913971/8776e03a-56d6-4917-8126-640d11c1cf7f)

- **Programming with objects** : Object-oriented programming, or OOP: A programming paradigm known as object-oriented programming (OOP) arranges code around objects, which are data structures that combine behavior and data. The project makes use of OOP concepts to produce modular and reusable code components.

<img width="758" alt="oops" src="https://github.com/jahnavirishikesh/POPL-G20-milestone/assets/101913971/015cdcd9-3a5a-477c-a440-0ae0f4ff46bb">

## Results
- We tested the model against data in our dataset that had not been used for training and validation.
- We got an average loss of 4.2 in python while getting 3.8 in pyro.
- As the model structure was similar, epochs and data used were identical we can show that the probabilistic nature of pyro improved model accuracy.
- We also tested the model against different setences and observed the pyro model was able to give more diverse range of predictions to the setences giving it an advantage in real world applications. 
- The models we have used are relatively small and further testing and analysis needs to be done.
We have attached graphs showing the comparsison between 
