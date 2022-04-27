# Project proposal DD2424


1) The names of the people in the group. 

Carina Wickström, Arash Dabiri, Isabella Tapper

2) A working title for the project. 

Phrase to Phrase Matching using Bert and SVR

3) Whether you will complete a custom, default or default + extension project. 

Custom

4) A brief description of the problem that you will work on and how you will try to solve it. Reference to at least one paper (or webpage) that provided inspiration as regards the problem statement and/or proposed approach to tackling the problem. (It is completely fine if you just want to replicate results of some papers and explore the influence of parameter settings, training conditions, see if the method can be transferred to another dataset etc...) 

Challenge:

We want to attempt to do the U.S. Patent Phrase to Phrase Matching challenge on Keggle.
https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching

The challenge is to train models on a novel semantic similarity dataset, and the task is to get important information by matching key phrases in patent documents. Deciding the semantic similarity between different phrases is crucial in the patent search and examination process to decide if an invention has previously been written about. If an invention declares "television set" and a prior publication informs about "TV set", a deep learning model would hopefully calculate that these inventions are too similar. Later the model could assist an examiner or attorney in retrieving relevant documents. This challenge and competition is beyond paraphrase identification. For example: if one invention describes a "strong material" and another says "steel", that is also a similarity match. The phrase ”strong material" is different in different contexts. Therefore, the competition material includes the Cooperative Patent Classification as the technical domain context as an additional feature to help us disambiguate these situations.

The question we want to answer is the following:

Can a ML model match phrases in order to extract contextual information? If so, can this model help the patent community connect the dots between millions of patent documents?

Implementation:

We want to use Bert. The idea with Bert is to use a model pre-trained on general language understanding on huge texts, like Wikipedia. One example uses “bert base uncased”. It is an NLP model that is pretrained on uncased english (English = english). It exists in the transformers library in Python, like this: tokenizer = transformers.TFBertModel.from_pretrained("bert-base-uncased"). 

Example 1: 
Semantic Similarity using BERT with 
https://keras.io/examples/nlp/semantic_similarity_with_bert/

It is common to connect several trainable layers after the Bert frozen layer, in order to adapt to the new data. In Example 1, they do pooling using lstm. They also use Dropout. We intend to use their idea of using multiple trainable layers on top of Bert.

Example 2:
Beginner : Bert For Patents + SVR
https://www.kaggle.com/code/nayakroshan/beginner-bert-for-patents-svr?fbclid=IwAR3ELBUtYfVq7JDdV_0pKuyL5l2btI-ZyaC79BOMHJItv--D0IZMud1XdD8

In Example 2 they use Bert for the NLP, and as output they express each phrase as a 1023-dimensional vector. (Each phrase has 1023 different features). Then they use SVR to determine the similarity of two phrases. Then they tune C and epsilon for the SVM.

Instead of SVM, it has also been attempted to use simple cosine similarity:
https://www.kaggle.com/code/damianpanek/bert-uncased-cosine-similarity-baseline

We intend to use SVR or cosine similarity (whichever we find works best) to find the similarities of the phrases in the end.

5) The data that you will use for training, validation and testing. (In most cases this training data should be labeled.) 

Kaggle provides both training and testing data
https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching

6) The deep learning software package(s) you will use.

tensorflow, transformers

7) How much of the software implementation your group will write and how much you will rely on open-source implementations. It is not acceptable though to rely completely on open-source implementations and just change parameter settings. One goal of the project is to learn some deep learning packages to allow you to set up architectures and training procedures. 

We will use open source implementations of Bert and SVM/cosine similarity. But we will need to connect new layers after Bert to adapt it to the patent data, which is not effortless. Also, a data pipeline will need to be built to first feed the retrained Bert with text, and then take the output vector to a SVM or a cosine similarity function to assess similarity. 


8) The initial set of experiments you will run and your baselines. 

Initially we will experiment with the adaptation of the Bert model, and see whether it leads to better results than an unchanged Bert model. Also, we will experiment with different implementations of SVM and cosine similarity and see which gives best results.

Our baselines will be decided by Kaggle, as there is a set scoring in the competition. We will try to get a result over 0.3, where the scoring goes from -1 to 1.

9) Which achieved milestones during your project should be attached to the grades from E to your target grade: (E)

E - adapt Bert to patent data, apply SVM or cosine similarity, get a Kaggle score over 0.3.

10) Specify for each group member the skills/knowledge w.r.t. deep learning “theory” and practice they aim to acquire from completing the project

Isabella: I want to learn more about deep learning and sentence similarity, which I think is very interesting. I am in my masters in Machine Learning and I have therefore previous knowledge about ML, AI and neural networks.

Carina: I am in my second year of master in Computer Science and have some previous experience in ML. From this assignment, I would like to get experience in how to design networks (which I find difficult today), to choose the amount of hidden layers, nodes per layer, hyperparameters, dropout, etc. 

Arash: I have previously worked with similar data pipelines at an NLP start-up, but now I want to learn how to adapt an already trained model. 

11) The grade your project group is aiming for in the range A-E
Aiming for E.


