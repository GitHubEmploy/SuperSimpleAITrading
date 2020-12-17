# This file manages the other files

# Number of bars to predict on
# Ex: If NUMBARS=4 use monday-thursday to predict friday

NUMBARS = 7

import coloredlogs, logging
global logger
from yahoo_fin import stock_info as si
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
global var
global keys
def train():
    from python.train_rnn import prepare
    from python.train_rnn import train_network
    from python.train_rnn import test_results
    import keras

    x_train, y_train = prepare(args.trainset, NUMBARS)
    model = train_network(x_train, y_train, args.epochs)

    logger.info('Saving model...')
    model.save('data/Trade-Model.h5')

    test_results(args.trainset, args.testset, model, NUMBARS)
    
def test():
    from python.train_rnn import test_results
    import keras.models as model
    model = model.load_model(args.model, compile=False)
    test_results(args.trainset, args.testset, model, NUMBARS)
    
def trade(is_test, time_period):
    from python.stock import Stock
    from python.PYkeys import Keys
    import alpaca_trade_api as tradeapi
    import pandas as pd
    from tensorflow.keras.models import load_model
    logger.info('Logging in...')
    keys = Keys()
    api = tradeapi.REST(key_id = keys.get_key_id(),
                        secret_key = keys.get_secret_key(),
                        base_url = keys.get_base_url(),
                        api_version = "v2")
    
    # Load S&P500
    logger.info('Loading stock list...')
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    #table=pd.read_html('https://www.benzinga.com/money/best-penny-stocks')
    df = table[0]
    sp = df['Symbol']
    
    if is_test:
        sp = sp[0:10]

    logger.info('Loading AI...')
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow import keras
    model = keras.models.load_model('data/Trade-Model.h5', compile=False)
    Stock.setup(NUMBARS, model, api, time_period)
    for symbol in sp:
        this_stock = Stock(symbol)
        
    global var
    var = 0
    while True:
        if len(Stock.owned) < 5:
            best_stocks = Stock.highest_gain(5 - len(Stock.owned))
            for stock in best_stocks:
                logger.info('Stock: ' + stock.symbol)
                logger.info('Gain = ' + str(stock.return_gain())+ '%')
                stock.buy()
                logger.info('-------------------------------')
        var = var + 1
        if var == 500:
            break
        print('numero', var, 'of 500, but anyways... Selling stocks...')
        for stock in Stock.owned:
            if stock.get_gain() < 0:
                stock.sell()
                logger.warning('!!!SHORTED STOCK!!! I REQUIRE MORE TRAINING!!!')
    
#############################################
# Command Line
#############################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Control Trading AI')
    parser.add_argument("command", metavar="<command>",
                        help="'train', 'trade', 'test'")
    parser.add_argument("--trainset", default='data/dataset.csv',
                        metavar="path/to/training/dataset",
                        help="Path to training dataset")
    parser.add_argument("--testset", default='data/ZION5Min.csv',
                        metavar="path/to/test/dataset",
                        help="Path to test dataset")
    parser.add_argument("--model", default='data/Trade-Model.h5',
                        metavar="path/to/model",
                        help="Path to model")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs to use in training")
    # New Data
    parser.add_argument("-d", action='store_true', required=False,
                        help="Include -d if you want to include new data")
    # Test
    parser.add_argument("-t", action='store_true', required=False,
                        help='Include -t if this is a shortened test')
    parser.add_argument("--time", default='5Min',
                        help = "Time period to buy and sell on")
    args = parser.parse_args()

    
    # Run based on arguments
    if args.d == True:
        import os
        os.system("python python/collect_data.py")
    elif args.command == 'train':
        train()

    elif args.command == 'test':
        test()

    elif args.command == 'trade':
        trade(args.t, args.time)
        
    else:
        raise NotImplementedError("Hmmm... I dont belive the stock market is open for you to be trading, if you ran option 1 contact mohit. But if you ran option 2 outside of open hours and picked 'trade' its going to give this error. Please wait till market open to run this command.")

