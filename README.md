# Multi-scale-convolutional-neural-network-for-automated-AMD-classification-using-retinal-OCT-images

### This repo contains the implelmentation of the following paper available using this link: https://www.sciencedirect.com/science/article/abs/pii/S0010482522001603
Sotoudeh-Paima, S., Jodeiri, A., Hajizadeh, F., & Soltanian-Zadeh, H. (2022). Multi-scale convolutional neural network for automated AMD classification using retinal OCT images. Computers in biology and medicine, 144, 105368.
### The dataset used in this paper is publicly available using this link: https://data.mendeley.com/datasets/8kt969dhx6/1
Sotoudeh-Paima, Saman; Hajizadeh, Fedra; Soltanian-Zadeh, Hamid (2021), “Labeled Retinal Optical Coherence Tomography Dataset for Classification of Normal, Drusen, and CNV Cases”, Mendeley Data, V1, doi: 10.17632/8kt969dhx6.1

#### Please cite the paper in case you have used our work in your research/project.

# Overview

### Background and objective
![alt text](<./figures/Fig_1.png>) 
Age-related macular degeneration (AMD) is the most common cause of blindness in developed countries, especially in people over 60 years of age. The workload of specialists and the healthcare system in this field has increased in recent years mainly due to three reasons:
1) increased use of retinal optical coherence tomography (OCT) imaging technique,
2) prevalence of population aging worldwide, and
3) chronic nature of AMD.

Recent advancements in the field of deep learning have provided a unique opportunity for the development of fully automated diagnosis frameworks. Considering the presence of AMD-related retinal pathologies in varying sizes in OCT images, our objective was to propose a multi-scale convolutional neural network (CNN) that can capture inter-scale variations and improve performance using a feature fusion strategy across convolutional blocks.

### Methods
Our proposed method introduces a multi-scale CNN based on the feature pyramid network (FPN) structure. This method is used for the reliable diagnosis of normal and two common clinical characteristics of dry and wet AMD, namely drusen and choroidal neovascularization (CNV). The proposed method is evaluated on the national dataset gathered at Hospital (NEH) for this study, consisting of 12649 retinal OCT images from 441 patients, and the UCSD public dataset, consisting of 108312 OCT images from 4686 patients.

### Results
Experimental results show the superior performance of our proposed multi-scale structure over several well-known OCT classification frameworks. This feature combination strategy has proved to be effective on all tested backbone models, with improvements ranging from 0.4% to 3.3%. In addition, gradual learning has proved to be effective in improving performance in two consecutive stages. In the first stage, the performance was boosted from 87.2%+-2.5% to 92.0%+-1.6% using pre-trained ImageNet weights. In the second stage, another performance boost from 92.0%+-1.6% to 93.4%+-1.4% was observed as a result of fine-tuning the previous model on the UCSD dataset. Lastly, generating heatmaps provided additional proof for the effectiveness of our multi-scale structure, enabling the detection of retinal pathologies appearing in different sizes.

### Conclusion
The promising quantitative results of the proposed architecture, along with qualitative evaluations through generating heatmaps, prove the suitability of the proposed method to be used as a screening tool in healthcare centers assisting ophthalmologists in making better diagnostic decisions.

# Code Explanation

1. **'main.py'** is the main function used to *load the data* from a structured dataframe (.csv file created from the dataset), *load the model* either from **'basemodels.py'** or **'vggCombinations.py'**, and run it on the data.
2. **'basemodels.py'** includes all the FPN-based VGG16, ResNet50, DenseNet121, and EfficientNetB0 models.
3. **'vggCombinations.py'** includes all the FPN-based VGG16 models with different combination of scales. The VGG16 model is composed of five convolutional blocks. Merging all convolutional blocks would not necessarily result in the best performance. To study the effect of feature fusion, we have run the models with five different fusion strategies:
    * In the first setting, we only used the top convolutional block (scale i = {5}) for retinal pathology classification.
    * In the second setting, we fused features of the last two convolutional blocks (scales i = {4, 5}) and measured the performance.
    * The other three settings include adding one more scale each time (scales i ={3, 4, 5}, {2, 3, 4, 5}, {1, 2, 3, 4, 5})

# Steps to Run the Code

1. prepare your dataset in the form of a csv file *(you can load your data in any other way by changing the data loader part of the **'main.py'** code)*
2. load your preferred model
3. select the proper parameters
4. run the model