# Detecting Differentiating Features: AI Generated vs Human-Created Art 

## Overview 
Given the advancements in AI, we were inspired to focus on the increasing challenge of distinguishing between AI-generated and human-created art. This issue raises ethical concerns, 
including scams, lack of proper credit, and more. Initially, we created a model that analyzed the entire dataset, but we found it didn't yield meaningful results. As a result, we 
shifted our approach, focusing on two models and narrowing our dataset to specific art styles. This is the link to the dataset we used: https://www.kaggle.com/datasets/ravidussilva/real-ai-art/data. 
It includes various art styles from Latent Diffusion Models (LDM), Standard Diffusion Models (SDM), and human-created art. We created a binary classification (AI-generated vs. 
human-created) and a 3-class classification (human-created, LDM, or SDM). We also implemented Grad-CAM to generate heatmaps, highlighting areas where each class focused and 
analysing how the model differentiates between the creators.
 

## Structure of Repository

### Branch: BinaryClassification

This branch contains the code for the binary classification model, which focuses specifically on the Ukiyo-E art style. We initially chose this approach to test and compare the results of the two models.

The images were preprocessed using various techniques, such as random horizontal flips, rotations, and color jittering. The model consists of four convolutional layers, each followed by ReLU activation, 
batch normalization, and pooling. The final two layers include a dropout of 0.25 to help prevent overfitting. These layers are followed by a classification head that flattens the data, applies a fully 
connected layer with ReLU activation, and returns the output layer.

The model was trained with the following specifications:

* Loss Function: Cross-Entropy Loss
* Optimizer: Adam with a learning rate of 0.001 and weight decay of 1e-4
* Training Duration: 10 epochs
* Batch Size: 32

After training, we plotted the training and test accuracy curves and applied Grad-CAM visualizations to each convolutional layer.


### Branch: Human_SD_LD_Classification

This branch contains the code for the three-class classification model, which focuses on five art styles: Ukiyo-e, Art Nouveau, Baroque, Impressionism, and Surrealism. Similar to the binary classification 
model, it originally focused only on the Ukiyo-e art style. However, after deciding to expand the model, we included additional art styles for analysis. This model is largely similar to the binary 
classification model, but it outputs three classes and features smaller feature maps.

We visualized the feature maps after each convolution layer to observe how the model's focus evolves (CNN on ArtBench.ipynb), and after a certain number of feature layers (CNN on Impressionism.ipynb). 
We also examine specific images to highlight the differences between the classes (CNN on Surrealism.ipynb, CNN on Ukiyo-e.ipynb (Older Version)).

To analyze the results (CNN on Ukiyo-e.ipynb), we compute the standard deviation (analyze_activation_uniformity) and the center of mass (analyze_center_of_mass) of the Grad-CAM heatmap for multiple samples.
These metrics help us understand activation patterns. We also visualize the results by plotting 2D and 3D scatter plots, which compare the spatial distribution of activation patterns across the three classes 
(LD, SD, and human).

Additionally, the code calculates the edge activation ratio, which measures the proportion of the heatmap's edge response relative to the total activation. This is computed using the Laplacian operator on the heatmap.

We create various plots to show:

* The activation uniformity for each class.
* The spatial distribution of the center of mass of activations in both 2D and 3D.
* The spread of the activation distribution and the edge activation ratio.
