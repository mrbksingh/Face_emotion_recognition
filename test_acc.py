import time
from parameters import *
from train import *
from dataset_loader import DatasetLoader
from os.path import isfile, join
import numpy as np

def start():
    predict = 'True'
    
    test = dict() 
    if predict:
        print("Loading pretrained model....")
        network = EmotionRecognition()
        model = network.build_network()
        data = DatasetLoader()
        data.load_from_save()
        test_dataX = data.images_test
        test_dataY = data.labels_test

        test['X'] = test_dataX
        test['Y'] = test_dataY       

        test['X2'] = None
        
        print( "Test samples: {}".format(len(test['X'])))
        print( "--")
        print( "start evaluation...")
        start_time = time.time()
        
        
        test_accuracy = evaluate(model, test['X'], test['X2'], test['Y'])
        print( "  - test accuracy = {0:.1f}".format(test_accuracy*100))
        print( "  - evalution time = {0:.1f} sec".format(time.time() - start_time))
        return test_accuracy
        
def evaluate(model, X, X2, Y):
    accuracy = model.evaluate(X, Y)
    return accuracy[0]

start()


  

    
    





 




    
