# Protonet_w_resnet152_embedding
Using resnet152 as the embedding for prototypical network to predict Omniglot characters in a 50-way 5-shot learning task. The model performance was compared with naive approach including linear model, KNN, and Mixture density estimation. Demonstrates the superior performance of the protonet and shows that the embedding function is not trivial.

demo is the main body. prototypical_network is the architecture of the protonet, data_load_helpers has helpful functions to prepare data. $K_2 \to$
