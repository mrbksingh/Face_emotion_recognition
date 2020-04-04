# Face_emotion_recognition
  We explore human recognition system to identify 7 types of emotions by using ***FER2013 dataset.***
  Further we build neural network and train our model using fer2013 dataset. Finally, we evaluate 
  the accuracy and predict facial emotions by giving pics as inputs and also by camera. 
## Overview
   Finally, testing with 2151 samples of fer2013 daatset we got an accuracy of 66.7% . 
#### Dependencies
 - Python == 3.6
 - opencv == 4.1.2
 - Keras
 - numpy
 - tensorflow with gpu support (***pip install tensorflow-gpu***)
 - tflearn (***pip install tflearn***)
#### Dataset
  [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data):-
  This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project.The data
  consists of 48x48 pixel grayscale images of faces.The public test set used for the leaderboard consists of 3,589 examples.
  The training set consists of 28,709 examples.***Download " fer2013.tar.gz " and decompress it.*** Put ***" fer2013.csv "***
  in the data directory.
  
### Prediction
   Images stored in the test gallery is used for prediction.And the prediction seems better using these different images.
<img src="testgallery/detected_faces.png" >
<img src="testgallery/detected_faces1.png" >
<img src="testgallery/detected_faces2.png" >
<img src="testgallery/detected_faces3.png" >
<img src="testgallery/detected_faces4.png" >
<img src="testgallery/detected_faces5.png" >
<img src="testgallery/detected_faces6.png" >

### References
   [http://tflearn.org/examples/](http://tflearn.org/examples/)


