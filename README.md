There are two python scripts : (1) customersuccessbox_1.py , (2) customersuccessbox_2.py
In file number (1), the model stack architecture has been presented where the stack model consists of 4 algorithms[Naive Bayse,RidgeClassifier,Logistic Regression and SVM(SVC)]. SVM(SVC) acts as meta model in this project and rest of the algorithms as intial models. We could also use any ensamble learning tree based algorithms for imbalanced data predictions but we need to keep in mind about binary classification also. 
In file number (2), the reading of the dataset,descriptions, seperation of train features and labels and finally calling the model architecture from file number(1) ahs been presented.
To run the script just run "python customersucessbox_2.py"
As results the script will generate classification report and precision-recall plot.
In the result the precision will be more than recall because the imbalance factor of the two classes present in the dataset is too high.
To deal with it option is resampling i.e. either upsample the minority class or downsample the majority class. But of we upsample the minority class here then we are defining the other class as minority and the result will be high recall but less precision. On other hand if we downsample the majority class to match with the minority class then we are loosing more information. So the best possible way is to collect more data or choose a good model.
Further changes could be done in the data preprocessing step.

************************* SUGGESTION FOR IMPROVMENT ARE ALWAYZ WELCOMED. *******************************************************
