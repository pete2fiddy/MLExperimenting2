import numpy as np
from classification.classifier import Classifier


class ImageCNN(Classifier):

    #layers must be nested prior (for now). Note: Error layer needs to be the last layer of the network when traversing
    #starting from first_layer
    def __init__(self, layers):
        self.__layers = layers
        self.__link_layers()

    #"connects" layers so that data can transfer through them, and initializes __first_layer and __last_layer
    def __link_layers(self):
        for i in range(1, len(self.__layers) - 1):
            self.__layers[i]._set_prev_layer(self.__layers[i-1])
            self.__layers[i]._set_next_layer(self.__layers[i+1])
        self.__layers[0]._set_next_layer(self.__layers[1])
        self.__layers[-1]._set_prev_layer(self.__layers[len(self.__layers)-2])


    def fit(self, X, y, **kwargs):
        learn_rate = kwargs["learn_rate"]
        num_steps = kwargs["num_steps"]
        for step in range(num_steps):
            tot_error = 0
            for img_index in range(X.shape[0]):

                if img_index % 25 == 0:
                    print("batch " + str(step) + " image " + str(img_index))
                predict, inouts = self.predict(X[img_index], inouts = True)

                inouts[-1] = y[img_index]
                tot_error += self.__layers[-1].error(predict, y[img_index])#BAD PRACTICE, last layer not guaranteed to be an error layer, append
                #there is no rigidity that guarantees error layers to have an error method
                self.__layers[-1].backprop_update_gradient(inouts, print_times = False)
                #self.__layers[0].step_net_params(learn_rate)
                for i in range(len(self.__layers)):
                    #print("layer " + str(i) + ": ")
                    self.__layers[i].step_params(learn_rate)
            #print("training batch " + str(step))
            print("Mean Error: ", tot_error/X.shape[0])


    '''assumes argmax last layer is the class of x, where x is either a single image, or an array of single images'''
    def predict(self, x, inouts = False):
        #TODO: code in multiple input case
        if inouts:
            predict, inouts = self.__layers[0].predict(x, inouts = inouts)
            #may need to delete the last layer since the error layer will duplicate the final output, since its transformation
            #does nothing (may be "None")
            #If this is the case, replace the last element of inouts with the target for x
            return predict, inouts

        return self.__layers[0].predict(x, inouts = False)
