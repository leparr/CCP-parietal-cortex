# CCP-parietal-cortex
Code to train, test and analyze Winner-Take-All activity as described in 'Computational Primitives in the Population Code of Parietal Circuits'.
The architecture was developed and tested on the 14/07/2023 with Python 3.10.

## Information analysis
Code to perfrom information analysis on the learned weights:
  - Principal Component analysis
  - K-means clustering
  - Within Cluster Sum of Squares
  - Cosine similarity
  - Jensen Shannon divergence

## Network Activity
Code to visualize the activity of the SNN WTA after testing:
  - Power Spectral Analysis
  - Accuracy
  - Difference in Mean Increase in Firing Rate (DIMFR)
  - Cumulative regime of WTA clusters
  - Rasterplot
  - Evolution of Mean Firing rate
  - 
## Network Visualization
Code to visualize the SNN inner structure:
  - Synaptic Kernel
  - Lerned weights
  - Temoral evolution of weights
  - Kernel Density Estimation of weights

## Training & testing
Code to instantiate, train, post-process synaptic weights and test the SNN WTA for classification given an input of in-vivo single cells recording in response to a delayed fix-to-reach task.
