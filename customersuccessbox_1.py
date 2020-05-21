import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


class Crossbox_Test:

	### Constructor of the class. ###
	def __init__(self,training_features,testing_features,training_label,testing_label,train_feature_names,test_feature_names):
		self.train_feature = training_features
		self.test_feature = testing_features
		self.train_label = training_label
		self.test_label = testing_label
		self.train_feature_name = train_feature_names
		self.test_feature_name = test_feature_names


	### Defining the Stack model architecture for prediction. ###
	def stack_model(self):

		data_split_size = int(len(self.train_feature) * .80 ) #### Spliting the training data into 80-20 ratio for train and validation sets.
		
		"""
		spliting the training data into
		sub_train datasets and validation sets.
		"""
		train_x = self.train_feature[self.train_feature_name][:data_split_size]

		sub_data_split = int(len(train_x) * .50 )

		sub_data_x1 = train_x[:sub_data_split]
		sub_data_x2 = train_x[sub_data_split:]

		train_y = self.train_label[:data_split_size]
		sub_data_y1 = train_y[:sub_data_split]
		sub_data_y2 = train_y[sub_data_split:]

		validation_x = self.train_feature[self.train_feature_name][data_split_size:]
		validation_y = self.train_label[data_split_size:]

		

		"""
			defining the stack architecture/model
		"""

		sub_model_1 = GaussianNB().fit(sub_data_x1,sub_data_y1)
		sub_model_2 = LogisticRegression(random_state=2).fit(sub_data_x1,sub_data_y1)
		sub_model_3 = RidgeClassifier().fit(sub_data_x1,sub_data_y1)

		sub_model_4 = GaussianNB().fit(sub_data_x2,sub_data_y2)
		sub_model_5 = LogisticRegression(random_state=2).fit(sub_data_x2,sub_data_y2)
		sub_model_6 = RidgeClassifier().fit(sub_data_x2,sub_data_y2)

		meta_model = SVC(kernel='linear',class_weight='balanced',probability=True)



		"""
			Performing validation prediction with the trained models.
		"""

		validation_pred_1 = sub_model_1.predict(validation_x)
		validation_pred_2 = sub_model_2.predict(validation_x)
		validation_pred_3 = sub_model_3.predict(validation_x)

		validation_pred_4 = sub_model_6.predict(validation_x)
		validation_pred_5 = sub_model_5.predict(validation_x)
		validation_pred_6 = sub_model_6.predict(validation_x)


		"""
			Performing test prediction with the trained models.
		"""
		
		testdata_pred_1 = sub_model_1.predict(self.test_feature[self.test_feature_name])
		testdata_pred_2 = sub_model_2.predict(self.test_feature[self.test_feature_name])
		testdata_pred_3 = sub_model_3.predict(self.test_feature[self.test_feature_name])

		testdata_pred_4 = sub_model_4.predict(self.test_feature[self.test_feature_name])
		testdata_pred_5 = sub_model_5.predict(self.test_feature[self.test_feature_name])
		testdata_pred_6 = sub_model_6.predict(self.test_feature[self.test_feature_name])



		"""
			Stacking the validation results. Stacking the test results.
			Stacked validation data will be the input for the meta model.
			Stacked test data will be the test dataset for the meta model.
		"""
		#stack_validation_predictions = np.column_stack((validation_pred_1,validation_pred_2,validation_pred_3))
		#stack_testdata_predictions = np.column_stack((testdata_pred_1,testdata_pred_2,testdata_pred_3))

		stack_validation_predictions = np.column_stack((validation_pred_1,validation_pred_2,validation_pred_3,validation_pred_4,validation_pred_5,validation_pred_6))
		stack_testdata_predictions = np.column_stack((testdata_pred_1,testdata_pred_2,testdata_pred_3,testdata_pred_4,testdata_pred_5,testdata_pred_6))

		"""
			Training the meta model and 
			predicting with it.
		"""
		meta_trained_model = meta_model.fit(stack_validation_predictions,validation_y)
		meta_prediction = meta_trained_model.predict(stack_testdata_predictions)


		"""
			Showing the prediction statistics of the meta model.
			This includes classification report,
			precision,recall,thresholds and the plot for the precision and recall curve.
		"""
		print("Classification_Report::",'\n',classification_report(self.test_label, meta_prediction))
		
		precision, recall, thresholds = precision_recall_curve(self.test_label, meta_prediction)

		plt.step(recall, precision, color='b', alpha=0.2,where='post')
		#plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.0])
		plt.xlim([0.0, 1.0])
		plt.show() 
