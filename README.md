# Antropogenic detector for the north sea. 
Training and testing of a DNN  for the detection of airguns and sonar in PAM data from the North Sea.

## Airgun detector 

For the airguns we will use the catalogue of sounds produced by previous surveys and train a model in an incremental procedure, trying to reduce the amount of manual labelling necessary. 
- Manual labelling of a subset of files in Raven. 
- Training of a first ensemble model, based on two small architectures (resnet18, efficientnet0) and a shallow classifier (Logisitc regression on a set of Perch 2 embeddings)
- Label prediciton based on the ensemble for additional files, high scores above a defined threshold are kept and the rest are reviewed, in this step we start to randomly include more negative samples to balance the model predicitons. 

All the work is being done in opensoundscape using a WSL environment on windows, for now I'm keeping everything on notebooks to know what I did. 

