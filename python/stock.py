try:
    from python.time_frame import Time_frame
except ModuleNotFoundError:
    from time_frame import Time_frame
import datetime
import coloredlogs, logging
from yahoo_fin import stock_info as si
from datetime import datetime
global current_time
global logger
global now
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
class Stock:
    _stocks = []
    owned = []
    _api = 0
    _period = 0
    
    def setup(NUMBARS, model, api, period):
        Time_frame.setup(NUMBARS, model, api)
        Stock._api = api
        Stock._period = Stock._convert_frame_name(period)
        
    
    def _convert_frame_name(time_frame):
        if time_frame == '1Min':
            time_frame = 0
        elif time_frame == '5Min':
            time_frame = 1
        elif time_frame == '15Min':
            time_frame = 2
        elif time_frame == '1D':
            time_frame = 3
        else:
            raise ValueError('Incorrect time frame')
        return time_frame
    
    def highest_gain(num_stocks): # returns num_stocks best stocks
                
        def get_gain(stock):
                return stock.frames[Stock._period].gain
        
        # Currently only using 5 max gains
        logger.info("Getting highest gain...")
        max_stocks = []
        for stock in Stock._stocks:
            stock.frames[Stock._period].get_gain()
            logger.info(stock.symbol + "'s gain is " + str(stock.frames[Stock._period].gain))
            if len(max_stocks) < num_stocks:
                max_stocks.append(stock)
            elif stock.frames[Stock._period].gain > max_stocks[num_stocks - 1].frames[Stock._period].gain:
                max_stocks.pop()
                max_stocks.append(stock)
            
            # sort list so lowest gain is at the end
            max_stocks.sort(reverse=True, key=get_gain)
        return max_stocks
        
    def __init__(self, symbol):
        self.symbol = symbol
        self._stocks.append(self)
        
        self.frames = [Time_frame('1Min', symbol), Time_frame('5Min', symbol),
                        Time_frame('15Min', symbol), Time_frame('1D', symbol)]
        
    # returns saved gain
    def return_gain(self):
        return self.frames[Stock._period].gain
    
    # updates gain and returns it
    def get_gain(self):
        self.frames[Stock._period].get_gain()
        return self.frames[Stock._period].gain
    
    def buy(self):
        Stock.owned.append(self)
        try:
            global now
            global current_time
            global bbbpower
            now = datetime.now()
            current_time = now.strftime("%H")
            account = Stock._api.get_account()
            perstockpower = int(float(account.buying_power)/4)
            bbbpower = int(perstockpower/float(si.get_live_price(self.symbol)))
            if 14<= int(current_time) <16:
                if int(round(float(si.get_live_price(self.symbol)))) <= perstockpower:
                    logger.info('Bought ' + self.symbol)
                    Stock._api.submit_order(
                        symbol=self.symbol,
                        qty=bbbpower,
                        side='buy',
                        type='market',
                        time_in_force='gtc',
                        extended_hours= True)
                elif int(round(float(si.get_live_price(self.symbol)))) <= perstockpower:
                    logger.info('Bought ' + self.symbol)
                    Stock._api.submit_order(
                        symbol=self.symbol,
                        qty=1,
                        side='buy',
                        type='market',
                        time_in_force='gtc',
                        extended_hours= True)
                else:
                    logger.error('Internal Error in Line 99, python/stock.py')
            else:
                if int(round(float(si.get_live_price(self.symbol)))) <= perstockpower:
                    logger.info ('Bought ' + self.symbol)
                    Stock._api.submit_order(
                        symbol=self.symbol,
                        qty=bbbpower,
                        side='buy',
                        type='market',
                        time_in_force='gtc')
                elif int(round(float(si.get_live_price(self.symbol)))) >= perstockpower:
                    logger.info ('Bought ' + self.symbol)
                    Stock._api.submit_order(
                        symbol=self.symbol,
                        qty=1,
                        side='buy',
                        type='market',
                        time_in_force='gtc')
                else:
                    logger.error('Internal Error in Line 112, python/stock.py')
        except:
            print('Insufficient Buying Power For Stock', self.symbol + " sorry...")
        return bbbpower
    def sell(self):
        Stock.owned.remove(self)
        print('Sold ' + self.symbol+" At time "+ current_time)
        position = Stock._api.get_position(self.symbol)
        print(position)
        Stock._api.submit_order(
            symbol=self.symbol,
            qty=position.qty,
            side='sell',
            type='market',
            time_in_force='gtc')
        


        



    
