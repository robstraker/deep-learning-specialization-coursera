## Course 5 - Week 1 Quiz - Recurrent Neural Networks


1. Suppose your training examples are sentences (sequences of words). Which of the following refers to the jth word in the ith training example?

	- [x] x<sup>(i)\<j></sup>
	
	> We index into the i-th row first to get the ith training example (represented by parentheses), then the j-th column to get the jth word (represented by the brackets).

	- [ ] x<sup>\<i>(j)</sup>
	- [ ] x<sup>(j)\<i></sup>
	- [ ] x<sup>\<j>(i)</sup>

2. Consider this RNN: This specific type of architecture is appropriate when:

	- [x] T<sub>x</sub> = T<sub>y</sub>
	
	> It is appropriate when every input should be matched to an output.

    - [ ] T<sub>x</sub> < T<sub>y</sub>
    - [ ] T<sub>x</sub> > T<sub>y</sub>
    - [ ] T<sub>x</sub> = 1

3. To which of these tasks would you apply a many-to-one RNN architecture? (Check all that apply).
	
	- [ ] Speech recognition (input an audio clip and output a transcript)
	- [x] Sentiment classification (input a piece of text and output a 0/1 to denote positive or negative sentiment)
    - [ ] Image classification (input an image and output a label)
	- [x] Gender recognition from speech (input an audio clip and output a label indicating the speaker’s gender)

4. At the t-th time step, what is the RNN doing? Choose the best answer.
	
	- [ ] Estimating P(y<sup><1></sup>,y<sup><2></sup>,...,y<sup><t−1></sup>)
	- [ ] Estimating P(y<sup>\<t></sup>)
	- [x] Estimating P(y<sup>\<t></sup> | y<sup><1></sup>,y<sup><2></sup>,..., y<sup><t−1></sup>)
	
	> Yes, in language model we try to predict the next step based on the knowledge of all prior steps.
    
    - [ ] Estimating P(y<sup>\<t></sup> | y<sup><1></sup>,y<sup><2></sup>,...,y<sup><t−1></sup>)

5. You have finished training a language model RNN and are using it to sample random sentences, as follows: What are you doing at each time step t?
	
	- [ ] (i) Use the probabilities output by the RNN to pick the highest probability word for that time-step as yhat<sup>\<t></sup>. (ii) Then pass the ground-truth word from the training set to the next time-step.
	- [ ] (i) Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as yhat<sup>\<t></sup>. (ii) Then pass the ground-truth word from the training set to the next time-step.
	- [ ] (i) Use the probabilities output by the RNN to pick the highest probability word for that time-step as yhat<sup>\<t></sup>. (ii) Then pass this selected word to the next time-step.
	- [x] (i) Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as yhat<sup>\<t></sup>. (ii) Then pass this selected word to the next time-step.

6. You are training an RNN, and find that your weights and activations are all taking on the value of NaN (“Not a Number”). Which of these is the most likely cause of this problem?
	
	- [ ] Vanishing gradient problem.
	- [x] Exploding gradient problem.
	- [ ] ReLU activation function g(.) used to compute g(z), where z is too large.
	- [ ] Sigmoid activation function g(.) used to compute g(z), where z is too large.

7. Suppose you are training a LSTM. You have a 10000 word vocabulary, and are using an LSTM with 100-dimensional activations a<t>. What is the dimension of Γu at each time step?

    - [ ] 1
	- [x] 100
	
	> Correct, Γu is a vector of dimension equal to the number of hidden units in the LSTM.

    - [ ] 300
    - [ ] 10000

8. Here’re the update equations for the GRU. Alice proposes to simplify the GRU by always removing the Γ<sub>u</sub>. I.e., setting Γ<sub>u</sub> = 1. Betty proposes to simplify the GRU by removing the Γ<sub>r</sub>. I. e., setting Γ<sub>r</sub> = 1 always. Which of these models is more likely to work without vanishing gradient problems even when trained on very long input sequences?

    - [ ] Alice's model (removing Γ<sub>u</sub>), because if Γ<sub>r</sub> ≈ 0 for a timestep, the gradient can propagate back through that timestep without much decay.
    - [ ] Alice's model (removing Γ<sub>u</sub>), because if Γ<sub>r</sub> ≈ 1 for a timestep, the gradient can propagate back through that timestep without much decay.
	- [x] Betty’s model (removing Γ<sub>r</sub>), because if Γ<sub>u</sub> ≈ 0 for a timestep, the gradient can propagate back through that timestep without much decay.
    - [ ] Betty’s model (removing Γ<sub>r</sub>), because if Γ<sub>u</sub> ≈ 1 for a timestep, the gradient can propagate back through that timestep without much decay.

9. Here are the equations for the GRU and the LSTM: From these, we can see that the Update Gate and Forget Gate in the LSTM play a role similar to _______ and ______ in the GRU. What should go in the the blanks?

	- [x] Γ<sub>u</sub> and 1−Γ<sub>u</sub>
	- [ ] Γ<sub>u</sub> and Γ<sub>r</sub>
	- [ ] 1−Γ<sub>u</sub> and Γ<sub>u</sub>
	- [ ] Γ<sub>r</sub> and Γ<sub>u</sub>

10. You have a pet dog whose mood is heavily dependent on the current and past few days’ weather. You’ve collected data for the past 365 days on the weather, which you represent as a sequence as x<1>,…,x<365>. You’ve also collected data on your dog’s mood, which you represent as y<1>,…,y<365>. You’d like to build a model to map from x→y. Should you use a Unidirectional RNN or Bidirectional RNN for this problem?

	- [ ] Bidirectional RNN, because this allows the prediction of mood on day t to take into account more information.
	- [ ] Bidirectional RNN, because this allows backpropagation to compute more accurate gradients.
	- [x] Unidirectional RNN, because the value of y<sup>\<t></sup> depends only on x<1>,…,x<t>, but not on x<sup>\<t+1></sup>,…,x<sup>\<365></sup>
	- [ ] Unidirectional RNN, because the value of y<sup>\<t></sup> depends only on x<t>, and not other days' weather.