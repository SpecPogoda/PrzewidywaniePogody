
"""
    https://colab.research.google.com/drive/1-OdCF3H_XB6XPItV68Lzy1I8fyTzzSz8
"""

import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

weather = [18.3, 21., 13.6, 9.8, 13.5, 12.99, 16., 16.5, 14.3]
weatherMean = np.mean(weather)
warmDay= []
coldDay= []
for i in weather:
  if i >= weatherMean:
    warmDay.append(i)
  else:
    coldDay.append(i)

wMean = np.mean(warmDay)
cMean = np.mean(coldDay)
wStanDev = np.std(warmDay)
cStanDev = np.std(coldDay)

wMean = np.float32(wMean)
cMean = np.float32(cMean)
wStanDev = np.float32(wStanDev)
cStanDev = np.float32(cStanDev)

tfd = tfp.distributions

if weather[len(weather)-1]>=wMean:
  initial_distribution = tfd.Categorical(probs = [.2, .8])
else:
   initial_distribution = tfd.Categorical(probs = [.8, .2])

transition_distribution = tfd.Categorical(probs = [[.7, .3], # zmiana po dniu zimnym
                                                   [.2,.8]]) # po dniu ciepłym
                                        
observation_distribution = tfd.Normal(loc = [cMean, wMean], scale = [cStanDev, wStanDev])

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=3)

mean = model.mean()

with tf.compat.v1.Session() as session:
  print(mean.numpy())