import os
import coloredlogs, logging
global logger
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
from datetime import datetime
global current_time
now = datetime.now()
current_time = now.strftime("%H")
print("Current Time =", str(current_time), "Hour")
while True:
    now = datetime.now()
    current_time = now.strftime("%H")
    if 6 <= int(current_time) < 13:
        print("Current Time =", current_time, "Hour")
        logger.info('Trading Now...')
        current_time = now.strftime("%H")
        os.system('python tradeAI.py trade')
        if current_time == 13:
            break
    elif 13 <= int(current_time) <14 or 0<= int(current_time) < 6:
        now = datetime.now()
        current_time = now.strftime("%H")
        print("Current Time =", current_time, "Hour")
        logger.info('Training Now...')
        now = datetime.now()
        current_time = now.strftime("%H")
        os.system('python tradeAI.py train')
        if 7 <= int(current_time) < 13:
            break

