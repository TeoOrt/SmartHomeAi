import tensorflow as tf

class Classifier():
    def __init__(self,training_map) -> None:
        self.training_map:dict = training_map
        
    def match_closest_result(self,key_item:str):
        test_item = key_item
        res_ = None
        max_diff = 0.0
        prediction_str = ''
        for keys,values in self.training_map.items():
            for predictions in values:
                self.__get_y(predictions,test_item)
                self.__cosine_similarity()
                test_val= max(self.cosine_sim.min(),max_diff)
                if test_val > max_diff:
                    max_diff = test_val
                    prediction_str= keys
        res_ = (max_diff,prediction_str)
        return res_


    def __get_y(self,training_data,testing_data):
        y_true = tf.convert_to_tensor(training_data)
        y_pred = tf.convert_to_tensor(testing_data)
        self.y_true_normalize = tf.nn.l2_normalize(y_true,axis=-1)
        self.y_pred_normalize = tf.nn.l2_normalize(y_pred,axis=-1)
    def __cosine_similarity(self):
        self.cosine_sim = tf.reduce_sum(tf.multiply(self.y_true_normalize,self.y_pred_normalize), axis = -1).numpy() #returns the prediction result
