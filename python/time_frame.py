import numpy as np
from sklearn.preprocessing import MinMaxScaler
import coloredlogs, logging
import requests
from yahoo_fin import stock_info as si
global logger
global barset
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
class Time_frame:
    _frames = []
    _NUMBARS = 0
    _model = 0
    _api = 0
    # time_frame can be 1Min, 5Min, 15Min, or 1D
    def __init__(self, frame, symbol):
        self.frame_name = frame
        self.symbol = symbol
        self._frames.append(self)
    

    def get_current_price(self):
        global barset
        try:
            global barset
            barset = (Time_frame._api.get_barset(self.symbol,'1Min',limit=1))
        except:
            print('Stock Failsafe Activated, skipped', self.symbol)
        symbol_bars = barset[self.symbol]
        current_price = symbol_bars[0].c
        return current_price
    
    def get_gain(self):
        prediction = self.get_prediction()
        current = self.get_current_price()
        
        gain = prediction/current
        self.gain = round((gain -1) * 100, 3)
    
    
    def get_prediction(self):
        logger.info('Getting prediction for '+ self.symbol+ ' on a ' + self.frame_name + ' time frame')
        # Get bars
        try:
            barset = (Time_frame._api.get_barset(self.symbol,self.frame_name,limit=Time_frame._NUMBARS))
        except:
            try:
                barset = (Time_frame._api.get_barset(self.symbol,self.frame_name,limit=Time_frame._NUMBARS))
            except:
                try:
                    barset = (Time_frame._api.get_barset(self.symbol,self.frame_name,limit=Time_frame._NUMBARS))
                except:
                    barset = (Time_frame._api.get_barset(self.symbol,self.frame_name,limit=Time_frame._NUMBARS))
        # Get symbol's bars
        symbol_bars = barset[self.symbol]

        # Convert to list
        dataSet = []

        for barNum in symbol_bars:
            bar = []
            bar.append(barNum.o)
            bar.append(barNum.c)
            bar.append(barNum.h)
            bar.append(barNum.l)
            bar.append(barNum.v)
            dataSet.append(bar)
            
          
        # Convert to numpy array
        npDataSet = np.array(dataSet)
        try:
            reshapedSet = np.reshape(npDataSet, (1, Time_frame._NUMBARS, 5))
        except ValueError:
            reshapedSet = np.reshape(npDataSet, (1, Time_frame._NUMBARS, 2))
            logger.warning('Reshaped np by 2')
        
        # Normalize Data
        sc = MinMaxScaler(feature_range=(0,1))

        try:
            normalized = np.empty(shape=(5, 5))
            normalized[0] = sc.fit_transform(reshapedSet[0])
            # Predict Price
            predicted_price = Time_frame._model.predict(normalized)
            logger.warning('ValueError Occurred, used failsafe')
        except ValueError:
            try:
                normalized = np.empty(shape=(1, 7, 5))
                predicted_price = Time_frame._model.predict(normalized)
            except ValueError:
                normalized = np.empty(shape=(1, 7, 5))
                #normalized[0] = sc.fit_transform(reshapedSet[0])
                predicted_price = Time_frame._model.predict(normalized)

        
        # Add 4 columns of 0 onto predictions so it can be fed back through sc
        shaped_predictions = np.empty(shape = (1, 5))
        for row in range(0, 1):
            shaped_predictions[row, 0] = predicted_price[row, 0]
        for col in range (1, 5):
            shaped_predictions[row, col] = 0
        
        
        # undo normalization
        #
        try:
            predicted_price = sc.inverse_transform(shaped_predictions)
        except ValueError:
            predicted_price = Time_frame._model.predict(normalized)
            logger.warning("Rare Export Error Occured")
        return predicted_price[0][0]
    

    def setup(NUMBARS, model, api):
        Time_frame._NUMBARS = NUMBARS
        Time_frame._model = model
        Time_frame._api = api
    
    def get_max_gain(_model):
        prediction_list = []
        frame_names = []
        for frame in Time_frame._frames:
            frame.get_prediction(_model)
            prediction_list.append(frame.prediction)
            frame_names.append(frame.frame_name)
        
        current_price = self.get_current_price()
        
        # predict percent gain
        highest_prediction = prediction_list.index(max(prediction_list))
        gain = prediction_list[highest_prediction]/current_price
        gain = round((gain -1) * 100, 3)
        
        gain_time_period = frame_names[highest_prediction]
        return gain, gain_time_period
