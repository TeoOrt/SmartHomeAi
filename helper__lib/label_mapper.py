


class Label_mapper():

    def __init__(self) -> None:
        self.result_mapper = {}
        self.result_mapper['H-0'] = 0
        self.result_mapper['H-1'] = 1
        self.result_mapper['H-2'] = 2
        self.result_mapper['H-3'] =3
        self.result_mapper['H-4'] =4
        self.result_mapper['H-5'] =5
        self.result_mapper['H-6'] =6
        self.result_mapper['H-7'] =7
        self.result_mapper['H-8'] =8
        self.result_mapper['H-9'] = 9
        self.result_mapper['H-DecreaseFanSpeed'] = 10
        self.result_mapper['H-FanOff'] =11
        self.result_mapper['H-FanOn'] =12
        self.result_mapper['H-IncreaseFanSpeed'] =13
        self.result_mapper['H-LightOff'] = 14
        self.result_mapper['H-LightOn'] = 15
        self.result_mapper['H-SetThermo'] =16

    def get_result(self,classification:str)-> int:
        return self.result_mapper[classification]
    

class Training_map():
    def __init__(self) -> None:
        self.training_map = {}
        self.training_map['DecreaseFanSpeed'] = 'H-DecreaseFanSpeed'
        self.training_map['IncreaseFanSpeed'] = 'H-IncreaseFanSpeed'
        self.training_map['Temperature'] = 'H-SetThermo'
        self.training_map['FanOn'] = 'H-FanOn'
        self.training_map['FanOff'] = 'H-FanOff'
        self.training_map['LightOn'] = 'H-LightOn'
        self.training_map['LightOff'] = 'H-LightOff'
        self.training_map['Zero'] = 'H-0'
        self.training_map['Onetimes'] = 'H-1'
        self.training_map['Twotimes'] = 'H-2'
        self.training_map['Threetimes'] = 'H-3'
        self.training_map['Fourtimes'] = 'H-4'
        self.training_map['Fivetimes'] ='H-5'
        self.training_map['Sixtimes'] = 'H-6'
        self.training_map['Seventimes'] ='H-7'
        self.training_map['Eighttimes'] = 'H-8'
        self.training_map['Ninetimes'] = 'H-9'
    def get_map(self):
        return self.training_map
    
