import coloredlogs, logging
global logger
from yahoo_fin import stock_info as si
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
class Keys:
    def __init__(self):
        import pandas as pd
        
        try:
            import os
            keys = pd.read_csv("keys.csv", sep=r'\s*,\s*', engine='python')
            
            self.key_id = keys['key_id'][0]
            self.secret_key = keys['secret_key'][0]
            self.base_url = keys['base_url'][0]
        except:
            logger.info('No keys have been set yet')
            logger.info('Enter keys now')
            self.edit_keys()

            #self.key_id = keys['key_id'][0]
            #self.secret_key = keys['secret_key'][0]
            #self.base_url = keys['base_url'][0]

    def edit_keys(self):
        import os      
        keys = open(r"keys.csv",'r')

        self.key_id = input('Enter key_id: ')
        self.secret_key = input('Enter secret_key: ')
        self.base_url = input('Enter base_url: ')

        keys.write('key_id, secret_key, base_url\n')
        
        keys.write(self.key_id + ',')
        keys.write(self.secret_key + ',')
        keys.write(self.base_url + '\n')

        keys.close()
        
    
    def get_key_id(self):
        return self.key_id
    
    def get_secret_key(self):
        return self.secret_key
    
    def get_base_url(self):
        return self.base_url
    
    def print_key(self):
        logger.info('Key id: ' + self.get_key_id())
