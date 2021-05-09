## Lambdatest Assignment

### 1. Hospital_Bed_Classification  
This is done using Decision Tree classifier after converting categorical features to OneHotEncodings.

**Improvements:** Various models can be tested along with their respective hyperparameter search.  

### 2. Low_Dimensional_Image_Representation  
This is done using Deep Convolutional Autoencoder (Encoder + Decoder) architecture to create approx. 18x reduction in dimension. Extensive architecture or hyperparameter search was not performed. Architecture is inspired from DCGAN paper.  

**Improvements:** A loss function suitable for images could be used, such as Structural Similarity score or SSIM as MSE/BCE promotes bluriness. Also, kernel_size and strides could be choosed better to increase sharpness. A different color-space could be used, as RGB promotes graying effect.  
Further, other methods could be used like KernelPCA or Adversarial Autoencoders (in case you need some control over the latent space)

### 3. News_Classification
Could be directly done based on the subject tag. But ignoring that, Multinomial Naive Bayes classifier with simple Count Vectorizer yields 96% accuracy.

**Improvements:** More sophisticated techniques like TF-IDF vectorizer or models like LSTM, GRU could be used.

### 4. Cartpole
Considering that this is a simple environment to solve, and since there was no restriction on the method to be used, solved this using Evolutionary Strategy. A neural network (2 hidden layers with 32 neurons each) is used to model the probabilty of taking an action given current state. This neural network was then optimized using Simple Gaussian Evolutionary Strategy i.e, to tune the weights of the neural network since gradients are not available for gradient descent. Each candidate or solution (from population) is plugged into the network and tested multiple times to deal with 'luck' to some extent.

**Improvements:** A RL algorithm or advanced Evolutionary Strategy like CMA-ES could be used.