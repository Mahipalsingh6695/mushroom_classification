import os
import sys
import pandas as pd
from src.mushroom_classification.exception import customexception
from src.mushroom_classification.logger import logging
from src.mushroom_classification.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
        
        except Exception as e:
            raise customexception(e,sys)
    

class CustomData:
    def __init__(self,
                 class:,
                 cap-shape:
                 cap-surface:
                 cap-color:
                 bruises:
                 odor:
                 gill-attachment:
                 gill-spacing:
                 gill-size:
                 gill-color:
                 stalk-shape:
                 stalk-roots:
                 talk-surface-above-ring:
                 stalk-surface-below-rings:
                 talk-color-above-ring:
                 stalk-color-below-ring:
                 veil-type:
                 veil-color:
                 ring-number:
                 ring-type:
                 spore-print-color:
                 population:
                 habitat):
        
        self.class=class
        self.cap-shape=cap-shape
        self.cap-surface=cap-surface
        self.cap-color=cap-color
        self.bruises=bruise
        self.odor=odor
        self.gill-attachment=gill-attachment
        self.gill-spacing=gill-spacing
        self.gill-size=gill-size
        self.gill-color=gill-color
        self.stalk-shape=stalk-shape
        self.stalk-roots=stalk-roots
        self.talk-surface-above-ring=talk-surface-above-ring
        self.stalk-surface-below-rings=stalk-surface-below-rings
        self.talk-color-above-ring=talk-color-above-ring
        self.stalk-color-below-ring=stalk-color-below-ring
        self.veil-type=veil-type
        self.veil-color=veil-color
        self.ring-numbe=ring-number
        self.ring-type=ring-type
        self.spore-print-color=spore-print-color
        self.population=population
        self.habitat=habitat
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'carat':[self.carat],
                    'class' :[self.class],
                    'cap-shape' :[self.cap-shape],
                    'cap-surface' :[self.cap-surface],
                    'cap-color' :[self.cap-color],
                    'bruises' :[self.bruises],
                    'odor' :[self.odor],
                    'gill-attachment' :[self.gill-attachment],
                    'gill-spacing' :[self.gill-spacing],
                    'gill-size' :[self.gill-size],
                    'gill-color' :[self.gill-color],
                    'stalk-shape' :[self.stalk-shape],
                    'stalk-roots' :[self.stalk-roots],
                    'talk-surface-above-ring' :[self.talk-surface-above-ring],
                    'stalk-surface-below-rings' :[self.stalk-surface-below-rings],
                    'talk-color-above-ring' :[self.talk-color-above-ring],
                    'stalk-color-below-ring' :[self.stalk-color-below-ring],
                    'veil-type' :[self.veil-type],
                    'veil-color' :[self.veil-color], 
                    'ring-numbe' :[self.ring-number],
                    'ring-type' :[self.ring-type],
                    'spore-print-color' :[self.spore-print-color],
                    'population' :[self.population],
                    'habita' :[self.habitat] 
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)