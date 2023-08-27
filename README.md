# Traffic-Sign-Recognition
There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. The process of determining which class a traffic sign belongs to is known as categorization of traffic signs
In this Python project, I have built a deep neural network model that can classify traffic signs present in the image into different categories. With this model, we are able to read and understand traffic signs which are a very important task for all autonomous vehicles.
Dataset:
For this project, I am using the public dataset available at Kaggle: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
The dataset contains more than 50,000 images of different traffic signs. It is further classified into 43 different classes. The dataset is quite varying, some of the classes have many images while some classes have few images. The size of the dataset is around 300 MB.

The approach to building this traffic sign classification model can be described in four steps:
-Explore the dataset
-Build a CNN model
-Train and validate the model
-Test the model with test dataset

The architecture of our model is:

2 Conv2D layer (filter=32, kernel_size=(5,5), activation=”relu”)
MaxPool2D layer ( pool_size=(2,2))
Dropout layer (rate=0.25)
2 Conv2D layer (filter=64, kernel_size=(3,3), activation=”relu”)
MaxPool2D layer ( pool_size=(2,2))
Dropout layer (rate=0.25)
Flatten layer to squeeze the layers into 1 dimension
Dense Fully connected layer (256 nodes, activation=”relu”)
Dropout layer (rate=0.5)
Dense layer (43 nodes, activation=”softmax”)
I compiled the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because I have multiple classes to categorise.

The model got a 95% accuracy on the training dataset. With matplotlib, I plotted the graph for accuracy and the loss:
![image](https://github.com/AnindyaSD/Traffic-Sign-Recognition/assets/93717247/62f8149a-4e30-44ab-81d1-309830e881a5)
![image](https://github.com/AnindyaSD/Traffic-Sign-Recognition/assets/93717247/0caf9b5e-ef4c-4aa0-b013-154789e1d2a5)

The dataset contains a test folder and in a test.csv file, I have the details related to the image path and their respective class labels. I have extracted the image path and labels using pandas. Then to predict the model, I resized the images to 30×30 pixels and make a numpy array containing all image data. From the sklearn.metrics, I imported the accuracy_score and observed how our model predicted the actual labels. We achieved a 95% accuracy in this model.

GUI application(using tkinter module):
Lastly, I also built a graphical user interface for the traffic signs classifier with Tkinter. Tkinter is a GUI toolkit in the standard python library.
In this file, I have first loaded the trained model ‘traffic_classifier.h5’ using Keras. And then I built the GUI for uploading the image and a button is used to classify which calls the classify() function. The classify() function is converting the image into the dimension of shape (1, 30, 30, 3). This is because to predict the traffic sign we have to provide the same dimension we have used when building the model. Then I predict the class, the model.predict_classes(image) returns us a number between (0-42) which represents the class it belongs to. I used the dictionary to get the information about the class.


