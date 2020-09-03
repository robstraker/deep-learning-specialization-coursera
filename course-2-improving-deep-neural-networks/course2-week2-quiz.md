## Course 2 - Week 2 Quiz - Optimization Algorithms

1. Which notation would you use to denote the 3rd layer’s activations when the input is the 7th example from the 8th minibatch?

    - [ ] a<sup>[3]{7}(8)</sup>
    - [x] a<sup>[3]{8}(7)</sup>   
    - [ ] a<sup>[8]{3}(7)</sup>      
    - [ ] a<sup>[8]{7}(3)</sup>
    
2. Which of these statements about mini-batch gradient descent do you agree with?

    - [ ] Training one epoch (one pass through the training set) using mini-batch gradient descent is faster than training one epoch using batch gradient descent.  
    - [x] One iteration of mini-batch gradient descent (computing on a single mini-batch) is faster than one iteration of batch gradient descent.
    - [ ] You should implement mini-batch gradient descent without an explicit for-loop over different mini-batches, so that the algorithm processes all mini-batches at the same time (vectorization).
    
3. Why is the best mini-batch size usually not 1 and not m, but instead something in-between?

    - [x] If the mini-batch size is m, you end up with batch gradient descent, which has to process the whole training set before making progress.    
    - [ ] If the mini-batch size is 1, you end up having to process the entire training set before making any progress.
    - [x] If the mini-batch size is 1, you lose the benefits of vectorization across examples in the mini-batch.
    - [ ] If the mini-batch size is m, you end up with stochastic gradient descent, which is usually slower than mini-batch gradient descent.
    
4. Suppose your learning algorithm’s cost ***J***, plotted as a function of the number of iterations, looks like this:

    - [x] If you’re using mini-batch gradient descent, this looks acceptable. But if you’re using batch gradient descent, something is wrong.   
    - [ ] Whether you're using batch gradient decent or mini-batch gradient descent, this looks acceptable.   
    - [ ] If you're using mini-batch gradient descent, something is wrong. But if you're using  batch gradient descent, this looks acceptable.   
    - [ ] Whether you're using batch gradient desceent or mini-batch gradient descent, something is wrong.
   
5. Suppose the temperature in Casablanca over the first three days of January are the same:

    Jan 1st: θ<sub>1</sub> = 10
    
    Jan 2nd: θ<sub>2</sub> = 10
    
    Say you use an exponentially weighted average with β = 0.5 to track the temperature: v<sub>0</sub> = 0, v<sub>t</sub> = βv<sub>t−1</sub> + (1 − β)θ<sub>t</sub>. If v<sub>2</sub> is the value computed after day 2 without bias correction, and v<sup>corrected<sub>2</sub></sup> is the value you compute with bias correction. What are these values?

    - [ ] v<sub>2</sub> = 7.5, v<sup>corrected</sup><sub>2</sub> = 7.5    
    - [ ] v<sub>2</sub> = 10, v<sup>corrected</sup><sub>2</sub> = 10    
    - [ ] v<sub>2</sub> = 10, v<sup>corrected</sup><sub>2</sub> = 7.5   
    - [x] v<sub>2</sub> = 7.5, v<sup>corrected</sup><sub>2</sub> = 10
    
6. Which of these is NOT a good learning rate decay scheme? Here, t is the epoch number.

    - [ ] α = 1 / sqrt(t) * α<sub>0</sub>    
    - [ ] α = 0.95<sup>t</sup>α<sub>0</sub>    
    - [ ] α = 1 / (1 + 2*t<sup>α</sup><sub>0</sub>)
    - [x] α = e<sup>t</sup> * α<sub>0</sub> 
    
7. You use an exponentially weighted average on the London temperature dataset. You use the following to track the temperature: v<sub>t</sub> = βv<sub>t−1</sub> + (1 − β)θ<sub>t</sub>. The red line below was computed using β = 0.9. What would happen to your red curve as you vary β? (Check the two that apply)

    - [ ] Decreasing β will shift the red line slightly to the right.    
    - [x] Increasing β will shift the red line slightly to the right.
    
    > True, remember that the red line corresponds to β = 0.9. In lecture we had a green line $$\beta = 0.98 that is slightly shifted to the right.  
    
    - [x] Decreasing β will create more oscillation within the red line.
    
    > True, remember that the red line corresponds to β = 0.9. In lecture we had a yellow line $$\beta = 0.98 that had a lot of oscillations.
    
    - [ ] Increasing β will create more oscillations within the red line.
    
8. Consider this figure:

    These plots were generated with gradient descent; with gradient descent with momentum (β = 0.5) and gradient descent with momentum (β = 0.9). Which curve corresponds to which algorithm?

    - [x] (1) is gradient descent. (2) is gradient descent with momentum (small β). (3) is gradient descent with momentum (large β)
    - [ ] (1) is gradient descent. (2) is gradient descent with momentum (large β). (3) is gradient descent with momentum (small β)
    - [ ] (1) is gradient descent with momentum (small β). (2) is gradient descent. (3) is gradient descent with momentum (large β)   
    - [ ] (1) is gradient descent with momentum (small β). (2) is gradient descent with momentum (small β). (3) is gradient descent

9. Suppose batch gradient descent in a deep network is taking excessively long to find a value of the parameters that achieves a small value for the cost function J(W[1],b[1],...,W[L],b[L]). Which of the following techniques could help find parameter values that attain a small value for J? (Check all that apply)

    - [x] Try mini-batch gradient descent   
    - [x] Try better random initialization for the weights  
    - [ ] Try initializing all the weights to zero   
    - [x] Try using Adam   
    - [x] Try tuning the learning rate α  

10. Which of the following statements about Adam is False? 

    - [x] Adam should be used with batch gradient computations, not with mini-batches.   
    - [ ] Adam combines the advantage of RMSProp and momentum.  
    - [ ] We usually use "default' values for the hyperparameters β<sub>1</sub>, β<sub>2</sub> and e in Adam (β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.999, e = 10<sup>-8</sup>).   
    - [ ] Adam should be used with batch gradient computations, not with mini-batches. 
   
  
  
  
    