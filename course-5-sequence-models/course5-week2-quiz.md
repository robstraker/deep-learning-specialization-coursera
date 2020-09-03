## Course 5 - Week 2 Quiz - Natural Language Processing & Word Embeddings

1. Suppose you learn a word embedding for a vocabulary of 10000 words. Then the embedding vectors should be 10000 dimensional, so as to capture the full range of variation and meaning in those words.
	
	- [ ] True
	- [x] False
	
	> The dimension of word vectors is usually smaller than the size of the vocabularly. Most common sizes for word vectors ranges between 50 and 400.
	
2. What is t-SNE?

    - [ ] A linear transformation that allows us to solve analogies on word vectors.
    - [x] A non-linear dimensionality reduction technique.
    - [ ] A supervised learning algorithm for learning word embeddings.
    - [ ] An open-source sequence modeling library.
    
3. Suppose you download a pre-trained word embedding which has been trained on a huge corpus of text. You then use this word embedding to train an RNN for a language task of recognizing if someone is happy from a short snippet of text, using a small training set.
  
    | **x (input text)** | **y (happy?)** |
    |------------------- | -------------- |
    | I'm feeling wonderful today! | 1 |
    | I'm bummed my cat is ill.    | 0 |
    | Really enjoying this!        | 1 |
  
    Then even if the word "ecstatic" does not appear in your small training set, your RNN might reasonably be expected to recognize "I'm ecstatic" as deseving a label y = 1.

    - [x] True
    
    > Yes, word vectors empower your model with an incredible ability to generalize. The vector for "ecstatic" would contain a positive/happy connotation which will probably make your model classified the sentence as a "1".
    
    - [ ] False
    
4. Which of these equations do you think should hold for a good word embedding? (Check all that apply)

    - [x] e<sub>boy</sub> - e<sub>girl</sub> ≈ e<sub>brother</sub> - e<sub>sister</sub>
    - [ ] e<sub>boy</sub> - e<sub>girl</sub> ≈ e<sub>sister</sub> - e<sub>brother</sub>
    - [x] e<sub>boy</sub> - e<sub>brother</sub> ≈ e<sub>girl</sub> - e<sub>sister</sub>
    - [ ] e<sub>boy</sub> - e<sub>brother</sub> ≈ e<sub>sister</sub> - e<sub>girl</sub>
    
5. Let *E* be an embedding matrix, and let o<sub>1234</sub> be a one-hot vector corresponding to word 1234. Then to get the embedding of word 1234, why don't we call *E* * o<sub>1234</sub> in Python?

    - [x] It is computationally wasteful.
    
    > Yes, the element-wise multiplication will be extremely inefficient.
    
    - [ ] The correct forumla is *E*<sup>T</sup> * o<sub>1234</sub>.
    - [ ] This doesn't handle unknown words (<UNK>).
    - [ ] None of the above: calling the Python snippet as described above is fine.

6. When learning word embeddings, we create an artificial task of estimating *P(target | context)*. It is okay if we do poorly on this artificial prediction task; the more important by-product of this task is that we learn a useful set of word embeddings.

    - [x] True
    - [ ] False

7. In the word2vec algorithm, you estimate *P(t | c)*, where *t* is the target word and *c* is a context word. How are *t* and *c* chosen from the training set? Pick the best answer.

    - [x] *c* and *t* are chosen to be nearby words.
    - [ ] *c* is the one word that comes immediately before *t*.
    - [ ] *c* is the sequence of all the words in teh sentence before *t*.
    - [ ] *c* is a sequence of several words immediately before *t*.

8. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The word2vec model uses the following softmax function:

    *P(t | c)* = *e<sup>0</sup><sup><sup>T<sub>ec<sub></sup></sup>* / *∑<sub>t'</sub><sup>10000</sup> e<sup>0</sup><sup><sup>T<sub>ec<sub></sup></sup>*

    Which of these statements are correct? Check all that apply.
    
    - [x] 0<sub>t</sub> and *e<sub>c</sub>* are both 500 dimensional vectors.
    - [ ] 0<sub>t</sub> and *e<sub>c</sub>* are both 1000 dimensional vectors.
    - [x] 0<sub>t</sub> and *e<sub>c</sub>* are both trained with an optimization algorithm such as Adam or gradient descent.
    - [ ] After training, we should expect 0<sub>t</sub> to be very close to *e<sub>c</sub>* when *t* and *c* are the same word.

9. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The GloVe model minimizes this objective:

    min ∑<sub>i=1</sub><sup>10,000</sup> ∑<sub>j=1</sub><sup>10,000</sup> *f(X<sub>ij</sub>)(0<sub>i</sub><sup>T</sup>e<sub>j</sub> + b<sub>i</sub> + b'<sub>j</sub> - logX<sub>ij</sub>)<sup>2</sup>*

    Which of these statements are correct? Check all that apply.
    
    - [ ] *0<sub>i</sub>* and *e<sub>j</sub>* should be initialized to 0 at the beginning of training.
    - [x] *0<sub>i</sub>* and *e<sub>j</sub>* should be initialized randomly at the beginning of training.
    - [x] *X<sub>ij</sub>* is the number of times word i appears in the context of word j.
    - [x] The weighting function *f*(.) must satisfy *f*(0) = 0.
    
    > The weighting function helps prevent learning only from extremely common word pairs. It is not necessary that it satisfies this function.

10. You have trained word embeddings using a text dataset of *m*<sub>1</sub> words. You are considering using these word embeddings for alangufage task, for which you have a separate labeled dataset of *m*<sub>2</sub> words. Keeping in mind that using word embeddings is a form of transfer learning, under which of these circumstances would you expect the word embeddings to be helpful?

    - [x] *m*<sub>1</sub> >> *m*<sub>2</sub>
    - [ ] *m*<sub>1</sub> << *m*<sub>2</sub>