import tensorflow as tf

class Classifier():
    def __init__(self,training_map,testing_map) -> None:
        self.training_map = training_map
        self.testing_map = testing_map
        
    def match_closest_result(self,key_item:str):
        test_item = self.testing_map[key_item]
        res_ = []
        for values in range(len(test_item)):
            max_diff = 0.0
            prediction_str = ''
            for keys in self.training_map.keys():
                self.__get_y(self.training_map[keys][values],test_item[values])
                self.__cosine_similarity()
                test_val= max(self.cosine_sim.min(),max_diff)
                if test_val > max_diff:
                    max_diff = test_val
                    prediction_str= keys
            res_.append((prediction_str,max_diff))
        return res_

    def __get_y(self,training_data,testing_data):
        y_true = tf.convert_to_tensor(training_data)
        y_pred = tf.convert_to_tensor(testing_data)
        self.y_true_normalize = tf.nn.l2_normalize(y_true,axis=-1)
        self.y_pred_normalize = tf.nn.l2_normalize(y_pred,axis=-1)
    def __cosine_similarity(self):
        self.cosine_sim = tf.reduce_sum(tf.multiply(self.y_true_normalize,self.y_pred_normalize), axis = -1).numpy() #returns the prediction result
