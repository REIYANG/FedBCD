DEVICE_NUM = 100 # 
DATASIZE_LOCAL = int(60000/DEVICE_NUM)
SERVER_NUM = 10
DEVICE_PER_SERVER = int(DEVICE_NUM/SERVER_NUM)
BATCH_SIZE = 32
STEP_NUM = 5
LABEL_DIVERSITY = 6
ACTIVE_PER_SERVER = 3
AVAILABLE_PER_SERVER = 8
CLOUD_STEP_NUM = 1

LEARNING_RATE = 0.005
NUM_EPOCHS = {0: 200, 3: 2000, 5: 2000, 7: 2000}
INPUT_SIZE = 784
HIDDEN_SIZES = [128, 64]
OUTPUT_SIZE = 10
RHO = 1.0
LRZ = {x: 0.5 for x in (0, 3, 5, 7)}
ELASTIC_LR = 0.2

EDGE_DEVICE_WAKEUP_RATE = 1.
EDGE_DEVICE_COMPUTATION_RATE = 1.
SERVERS = ['192.168.16.3', '192.168.16.4', '192.168.16.5', '192.168.16.6']
PORT_OFFSET = 23000
