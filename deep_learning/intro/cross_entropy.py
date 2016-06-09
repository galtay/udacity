"""

W X + b = y

  W = weights
  X = input
  b = bias
  y = logit
  S(y) = softmax on logit

Cross-entropy (distance between two vectors)
   D(S,L) = -Sum_i L_i log(S_i)

Here S is the prediction probability vector and L is the one-hot vector


Concisely,

  D[ S(W X + b), L ]

"""
