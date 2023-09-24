# EDCNN
### Ensemble Deep Neural Networks integrating genomics and histopathological images for predicting stages and survival time-to-event in colon cancer.

Salient and complementary features are extracted from histopathological whole slide images (WSI) and genomics (mRNA, miRNA and DNA methylation) expression. Colon cancer stages prediction estimated with DNN model using (1) image features (2) genomics (3) integrated features as input into predictive model. The result from each features compare and contrast.  Also, extracted features tested on its capacity to stratify the samples into low or high risk survival groups

#### Materials and Methods

#### Preprocess the genomics datasets: 
- First, a biological feature (in any of methylation or mRNA or miRNA data) is removed, if >20% of the patients have a 0 value for it. 
- A sample is removed, if >20% of its features are missing. 
- Then fill out the missing values using a python function. 
- Only common samples(individuals) existing in all the data sets (mRNA, miRNA, Methylation) are kept. 
- The data sets (mRNA, miRNA, Methylation) are individually normalized with z-score. 
- The z-score normalized data are combined and re-normalized with unit scale (L2-norm) transform. 

Summary result of Genomics preprocessing.
Note: Before Preprocessing(BP), After Preprocessing (AP).
| Biological Data | No. of Samples (BP) | No. of Features (BP) | No. of Overlapping Samples | No. of Features (AP}|
	| :--------------- |:--------------------|:-------------------- |:-------------------- |:-------------------- |
	| mRNA		| 328	| 20,502 | 255	| 16,377 |
	| miRNA		| 261	| 1,870 | 255 | 420 |
	| DNA methylation | 353 | 20,759 | 255 |20,129 |

#### H&E histopathological image preprocessing
The stained hematoxylin and eosin whole slide images downloaded from The Genome Atlas (TCGA) of colon cancer are large in size with average of 2GB and of high resolution. Each of the 177 whole slide image samples used in the study was divided into several tiles of size 224 x 224 pixel with openslide-python packages, we build a sub python function called MX to filter out patches with less than 30% cellular tumor content. After preprocessing a total of 112,161 image tiles satisfy the condition and requirement set out for the research study. Salient features are extracted from the resulting 112,161 image samples and integrated with equivalent genomics features extracted from combinations of mRNA, miRNA and DNA methylation

