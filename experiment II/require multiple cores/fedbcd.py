import sys
import itertools
import socket
import select
import pickle
import time
import copy
import random
import logging
import os
import multiprocessing
import warnings
import datetime
import colorsys

# third party modules
import redis
import networkx as nx
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import sampler
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch.optim as optim


logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"]="1"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug(device)


from config import *
from parafuncs import *
from network import *


REDIS_SERVER = '192.168.16.6'
REDIS_PREFIX = 'fedbcd-{ASYNC}-{ELASTIC}-'


def coordinator_process(ASYNC, ELASTIC, num_servers):
    ''' This is the function started for the coordinator '''

    r = redis.Redis(host=REDIS_SERVER, port=6379, db=0) # open redis connection
    r.set(REDIS_PREFIX.format(ASYNC=ASYNC, ELASTIC=ELASTIC) + 'start', pickle.dumps(datetime.datetime.now())) # create the starting time

    try:
        conns = []

        # Create a listening socket to accept incoming connections
        listen_sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = PORT_OFFSET
        listen_sckt.bind(('0.0.0.0', port)) # bind to all interfaces
        listen_sckt.listen(5) # maximum number of open connections
        logger.info(f"Coordination server listening on port {port}")

        # Wait for `num_servers` connections
        while len(conns) < num_servers:
            sckt, addr = list(listen_sckt.accept())
            device_id = recv_pickle(sckt) # first message is the cloud server id
            sckt.setblocking(0)
            conn = AsyncSocketWrapper(sckt)
            conns.append((conn, addr, device_id))
            logger.debug(f"Accepted connection from {addr} cloud id {device_id} edges {len(conns)}/{num_servers}")

        logger.debug("Established all incoming connections")

        # Close the listening socket
        listen_sckt.shutdown(socket.SHUT_RDWR)
        listen_sckt.close()

        conns = sorted(conns, key=lambda x: x[2]) # Sort connections by cloud server id

        epoch = 0
        # Number of incoming messages to wait for at each round
        INCOMING = num_servers if ASYNC == 0 else min(ASYNC, num_servers) 
        assert INCOMING > 1

        # Primary loop
        for epoch in range(NUM_EPOCHS[ASYNC]):
            logger.debug(f"Starting cloud epoch {epoch}")

            # Wait for sufficient number of incoming messages
            incoming_messages = {}
            while len(incoming_messages) < INCOMING:
                readable, _, _ = select.select([x[0].sckt for x in conns], [], [])
                
                for sckt in readable:
                    conn, addr, device_id = [c for c in conns if  c[0].sckt is sckt][0]
                    out = conn.read() 
                    if out is None:
                        conn.sckt.close()
                        for i, (c, _, _) in enumerate(conns):
                            if c.sckt is sckt:
                                conns.pop(i)
                                break
                    elif out is True:
                        incoming_messages[device_id] = conn.shift()
            logger.info(f"Calculating averages for epoch {epoch} servers {sorted(list(incoming_messages.keys()))} and replying")

            lrz = LRZ[ASYNC] * (1-0.0001)**epoch

            # Run appropriate calculation (different for sync/async versions)
            if ASYNC == 0:
                if epoch == 0:
                    # initialize the first global model (should be identical everywhere)
                    para_global = list(ps for ps, pm in incoming_messages.values())[0]

                para_avg    = avg_dict(list(x for ps, pm in incoming_messages.values() for x in pm)) # all device messages
                para_global = sub_dict(para_global, mul_dict(sub_dict(para_global, para_avg), lrz))
                # Send the global parameter set to all devices
                for k, conn in enumerate(conns):
                    conns[k][0].sendall(para_global)
            else:
                # In the async version we simply average the incoming messages
                avg = avg_dict(list(incoming_messages.values()))
                # Reply to the incoming servers with the average value
                for k in incoming_messages.keys():
                    conns[k][0].sendall((lrz, avg))

            # This loop is simply pushing out replies to the tcp buffers as messages get sent
            while True:
                not_empty = {i: conn[0].sckt for i, conn in enumerate(conns) if not conn[0].empty()}
                if len(not_empty) == 0: break
                buffer_lengths = {i: len(conns[i][0].outgoing_buffer) for i in not_empty.keys()}
                _, writeable, _ = select.select([], list(not_empty.values()), [])
                
                for sckt in writeable:
                    conn, addr, device_id = [c for c in conns if  c[0].sckt is sckt][0]
                    conn.send()
            logger.info(f"Multicast reply sent to {len(incoming_messages)} servers")

        logger.info("Shutting down")

        for c, _, _ in conns:
            c.sckt.shutdown(socket.SHUT_RDWR)
            c.sckt.close()

    finally:
        try:
            listen_sckt.shutdown(socket.SHUT_RDWR)
            listen_sckt.close()
        except OSError as e:
            pass
            
        for c, _, _ in conns:
            c.sckt.close()


def cloud_process(ASYNC, ELASTIC, server_id, n_edges, coordinator_ip):
    ''' This is the function started for the cloud server '''
    r = redis.Redis(host=REDIS_SERVER, port=6379, db=0) # Open redis connection

    # Load the MNIST dataset
    testset = torchvision.datasets.MNIST('.data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
    ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=8)

    torch.manual_seed(0) # Seed the random number generator

    # Define the NN structure
    net = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]), nn.ReLU(), nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]), nn.ReLU(), nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE), nn.LogSoftmax(dim=1))
    for p in net.parameters():
        p.requires_grad = False

    try:
        edge_conns= []
        cord_conn = None
        # Open a listening socket for incoming connections from edge devices
        listen_sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = PORT_OFFSET + 1 + server_id
        listen_sckt.bind(('0.0.0.0', port)) # bind to all interfaces
        listen_sckt.listen(5) # maximum number of open connections
        logger.info(f"Cloud server listening on port {port}")

        # Wait for `n_edges` number of connections
        while len(edge_conns) < n_edges:
            sckt, addr = list(listen_sckt.accept())
            device_id = recv_pickle(sckt) # first message is the edge device id

            edge_conns .append((sckt, addr, device_id))
            logger.debug(f"Accepted connection from {addr} device id {device_id} edges {len(edge_conns)}/{n_edges}")

        logger.debug("Established all incoming connections")

        # Establish a connection to the cloud server
        cord_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for i in range(120):
            try:
                cord_conn.connect((coordinator_ip, PORT_OFFSET))
                logger.debug(f"Connected to coordinator")
                send_pickle(cord_conn, server_id)
                break
            except ConnectionRefusedError:
                logger.debug(f"Reconnecting to coordination server (attempt {i})")
                time.sleep(random.random())

        logger.info(f"Server {server_id} established all connections")
        # Close the listening socket
        listen_sckt.shutdown(socket.SHUT_RDWR)
        listen_sckt.close()

        # Initialize parameters
        para_server = copy.deepcopy(net.state_dict()) # the server parameters
        para_devices = [copy.deepcopy(para_server) for i in range(n_edges)]

        # Primary loop
        for epoch in range(NUM_EPOCHS[ASYNC]):
            logger.info(f"Starting cloud epoch {epoch}")
            start_time = time.time()

            # Choose edge devices to train
            wake_up_time = np.random.exponential(scale=1/EDGE_DEVICE_WAKEUP_RATE, size=len(edge_conns))
            if ELASTIC:
                idx = np.argpartition(wake_up_time, AVAILABLE_PER_SERVER)
                AVAILABLE_DEVICE = idx[:AVAILABLE_PER_SERVER]
            idx = np.argpartition(wake_up_time, ACTIVE_PER_SERVER)
            picks = idx[:ACTIVE_PER_SERVER]
            if ELASTIC:
                picks = [i for i in picks if i in AVAILABLE_DEVICE]

            logger.debug(f"Picked children {picks}")

            time.sleep(np.max(wake_up_time[picks]))

            # Convert the parameter format
            net.load_state_dict(copy.deepcopy(para_server))
            z_n = list(net.parameters())
            for i, (conn, addr, edge_id) in enumerate(edge_conns):
                if ELASTIC and i in AVAILABLE_DEVICE:
                    # Send edge devices `para_server` if they should train and answer, otherwise send them `None`
                    send_pickle(conn, (para_server, z_n) if i in picks else (None, None))
                elif not ELASTIC and i in picks:
                    send_pickle(conn, (para_server, z_n))

            logger.debug(f"Sent parameters to edge devices")
                    
            # Listen for responses from edge devices
            for i in picks:
                conn, addr, edge_id = edge_conns[i]
                para_devices[i] = recv_pickle(conn)

            logger.debug(f"Received parameters {len(picks)}/{len(para_devices)} edges {picks}")
            int_time = time.time()

            para_devices_subset = [x for i, x in enumerate(para_devices) if i in picks] if ELASTIC else para_devices

            if ASYNC == 0:
                # If synchronized, send device messages for coordinator to average
                send_pickle(cord_conn, (para_server, para_devices_subset))
                logger.info("Sent message to coordinator")
                para_server = recv_pickle(cord_conn) # average over all servers
                logger.info("Received sync average from coordinator")
            else:
                # If async, send server parameters to coordinator and use response to calculate update
                send_pickle(cord_conn, para_server)
                logger.info("Sent message to coordinator")
                para_mean = avg_dict(para_devices_subset) # average of edge device values
                lrz, para_server_mean = recv_pickle(cord_conn) # average over all servers
                logger.info("Received async average from coordinator")
                para_server = sub_dict(para_server_mean, mul_dict(sub_dict(para_server_mean, para_mean), lrz))

            stop_time = time.time()

            # Test accuracy
            solve_time, correct, total = test(net, testloader, para_server)
            logger.info('[%d, %d] cloud test accuracy: %.2f %%' %  (NUM_EPOCHS[ASYNC] + 1, epoch + 1, 100 * float(correct) / total))
            # Write accuracy solution to redis
            r.rpush(REDIS_PREFIX.format(ASYNC=ASYNC, ELASTIC=ELASTIC) + 'cloud', pickle.dumps((datetime.datetime.now(), epoch, 'cloud', server_id,  int_time - start_time, stop_time - int_time, para_server, correct, total)))
    except BrokenPipeError as e:
        logger.warning("Broken pipe, shutting down")
    finally:
        for c, _, _ in edge_conns:
            c.shutdown(socket.SHUT_RDWR)
            c.close()

        try:
            listen_sckt.shutdown(socket.SHUT_RDWR)
            listen_sckt.close()
        except OSError as e: pass

        try:
            cord_conn.shutdown(socket.SHUT_RDWR)
            cord_conn.close()
        except OSError as e: pass

    logger.debug("Shutting down")

def edge_process(ASYNC, ELASTIC, device_ID, cloud_server):
    ''' This is the function started for the edge device process '''

    r = redis.Redis(host=REDIS_SERVER, port=6379, db=0) # Open redis connection

    torch.manual_seed(device_ID) # Seed random number generator

    # Load training dataset
    trainset = torchvision.datasets.MNIST('.data/', train=True , download=True, transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Normalize( (0.1307,), (0.3081,)) ]))

    # Choose subset of data
    label_set = random.sample(range(0, 10), LABEL_DIVERSITY)
    idx = trainset.targets.clone().detach() == label_set[0]
    for label_val in label_set[1:]:
        idx += trainset.targets.clone().detach() == label_val
    indx = np.random.permutation(np.where(idx==1)[0])[0:DATASIZE_LOCAL]
    trainset_indx = torch.utils.data.Subset(trainset, indx)
    trainloader = torch.utils.data.DataLoader(trainset_indx, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    torch.manual_seed(0) # Re-initialize the RNG
    # Define NN structure
    net = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]), nn.ReLU(), nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]), nn.ReLU(), nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE), nn.LogSoftmax(dim=1))

    para = net.state_dict() # Obtain NN state dict

    # Define solver parameters
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)

    try:
        # Connect to cloud server
        sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for i in range(120):
            try:
                sckt.connect(cloud_server)
                logger.debug(f"Connected to cloud server {cloud_server}")
                send_pickle(sckt, device_ID)
                break
            except ConnectionRefusedError:
                logger.debug(f"Reconnecting to server {cloud_server} (attempt {i})")
                time.sleep(random.random())
                continue
        else:
            raise Exception(f"Could not connect to cloud server {cloud_server}")
        
        epoch = 0
        update_collect = 0
        # Primary loop
        while True:
            assert para is not None

            # Receive cloud message
            server_para, z_n = recv_pickle(sckt)
            start_time = time.time()
            #logger.debug(f"Received server_para/z_n from {cloud_server}")

            stopping_iter = random.randint(1, STEP_NUM) # Number of iterations
            net.load_state_dict(para)
            iter_count = 0

            # Training phase for elastic/inelastic formulations
            if ELASTIC:
                if update_collect <= 4:
                    update_collect += 1
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data[0], data[1]
                        inputs = inputs.view(inputs.shape[0], -1)
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        fitting_loss = criterion(outputs, labels)
                        loss = fitting_loss
                        loss.backward()
                        optimizer.step()
                        iter_count += 1
                        if iter_count == stopping_iter:
                            break

                if server_para is not None:
                    update_collect = 0
                    para_tmp = copy.deepcopy(net.state_dict())
                    for i in range(stopping_iter):
                        para_tmp = sub_dict(para_tmp, mul_dict(sub_dict(para_tmp, server_para), ELASTIC_LR))
                    para = para_tmp
                    time.sleep(stopping_iter*np.random.exponential(scale=1/EDGE_DEVICE_COMPUTATION_RATE))
                    send_pickle(sckt, para)
            else: # INELASTIC
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data[0], data[1]
                    inputs = inputs.view(inputs.shape[0], -1)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    fitting_loss = criterion(outputs, labels)
                    penalty = None
                    for (Ww, Zz) in zip(net.parameters(), z_n):
                        if penalty is None:
                            penalty = torch.norm(Ww-Zz)**2
                        else:
                            penalty = penalty + torch.norm(Ww-Zz) ** 2
                    loss = fitting_loss + RHO * penalty
                    loss.backward()
                    optimizer.step()
                    iter_count += 1
                    if iter_count == stopping_iter:
                        break
                time.sleep(stopping_iter*np.random.exponential(scale=1/EDGE_DEVICE_COMPUTATION_RATE))
                para = copy.deepcopy(net.state_dict())
                send_pickle(sckt, para)
            stop_time = time.time()
            epoch += 1
    except BrokenPipeError as e:
        logger.warning("Broken pipe, shutting down")
    finally:
        sckt.shutdown(socket.SHUT_RDWR)
        sckt.close()
    logger.debug("Shutting down")


def test(net, testloader, para):
    ''' This tests the accuracy for a certain parameter set '''
    a = time.time()
    correct, total = 0, 0
    with torch.no_grad():
        net.load_state_dict(para)
        for data in testloader:
            if torch.cuda.device_count() != 0:
                images, labels = data[0].cuda(), data[1].cuda()
            else:
                images, labels = data[0], data[1]
            images = images.view(images.shape[0], -1)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return time.time() - a, correct, total
    logger.debug('[%d, %d] test accuracy: %.2f %%' %  (NUM_EPOCHS[SYNC] + 1, epoch + 1, 100 * float(correct) / total))


def start_parallel(ASYNC, ELASTIC):
    ''' This is a helper function to start up all the processes '''

    # Start by clearing existing redis keys
    r = redis.Redis(host=REDIS_SERVER, port=6379, db=0)
    for key in r.keys(REDIS_PREFIX.format(ASYNC=ASYNC, ELASTIC=ELASTIC) + '*'):
        logger.debug(f"Deleting redis key {key}")
        r.delete(key)

    import subprocess
    assert ASYNC <= SERVER_NUM

    coordinator_server = SERVERS[-1] # Place the coordinator on the last server
    cloud_servers      = [SERVERS[i*len(SERVERS)//SERVER_NUM] for i in range(SERVER_NUM)] # Evenly distribute the cloud servers over the servers
    edge_servers       = [SERVERS[i*len(SERVERS)//DEVICE_NUM] for i in range(DEVICE_NUM)] # Evenly distribute the edge devices over the servers

    script = os.path.join(os.getcwd(), __file__) # The python script location

    # Define subprocess.Popen arguments
    procs = []
    procs.append((coordinator_server, [sys.executable, script, 'coordinator', str(int(ASYNC)), str(int(ELASTIC)), str(SERVER_NUM)]))

    # Map edge devices to cloud servers
    cloud_mapping = {}
    for i in range(SERVER_NUM):
        for j in range(DEVICE_PER_SERVER):
            cloud_mapping[len(cloud_mapping)] = i

    for i in range(SERVER_NUM):
        # Arguments to start cloud process
        procs.append((cloud_servers[i], [sys.executable, script, 'cloud', str(int(ASYNC)), str(int(ELASTIC)), str(i), str(sum(int(x == i) for x in cloud_mapping.values())), coordinator_server]))
    for i in range(DEVICE_NUM):
        # Arguments to start edge process
        procs.append((edge_servers[i] , [sys.executable, script, 'edge',  str(int(ASYNC)), str(int(ELASTIC)), str(i), cloud_servers[cloud_mapping[i]], str(PORT_OFFSET + 1 + cloud_mapping[i])]))

    for p, (server, args) in enumerate(procs):
        time.sleep(1/10)
        # Start processes over ssh
        procs[p] = subprocess.Popen(['ssh', server] + args)

    # Wait for processes to finish
    for p in procs:
        p.wait()


def plot():
    ''' This function plots the results '''

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Choices to plots, first list is ASYNC values, second is ELASTIC values
    CHOICES = list(itertools.product([0, 3, 5, 7], [False, True]))

    r = redis.Redis(host=REDIS_SERVER, port=6379, db=0)

    # Cloud server redis keys
    cloud_key = {(ASYNC, ELASTIC): REDIS_PREFIX.format(ASYNC=ASYNC, ELASTIC=ELASTIC) + 'cloud' for ASYNC, ELASTIC in CHOICES}

    # Start time values
    start = {KEY: pickle.loads(r.get(REDIS_PREFIX.format(ASYNC=KEY[0], ELASTIC=KEY[1]) + 'start')) if r.exists(REDIS_PREFIX.format(ASYNC=KEY[0], ELASTIC=KEY[1]) + 'start') else None  for KEY in cloud_key.keys()}
    # Cloud key values
    cloud = {KEY: sorted([pickle.loads(x) for x in r.lrange(cloud_key[KEY], 0, r.llen(cloud_key[KEY]))], key=lambda x: x[0]) for KEY in cloud_key.keys()}

    # The number of cloud servers
    num_servers = max(x[3] for y in cloud.values() for x in y)+1

    cloud_fig, cloud_ax = plt.subplots(num_servers, dpi=300, figsize=(6,2*num_servers))
    colors = [colorsys.hls_to_rgb(i/len(CHOICES), 0.5, 0.5) for i in range(len(CHOICES))] # Colors for different solutions

    # Loop over each solution and plot
    for oi, (ASYNC, ELASTIC) in enumerate(CHOICES):
        # We create a subfigure for each cloud server
        for i in range(num_servers):
            _cloud = [(x[0], 100*x[7]/x[8]) for x in cloud[ASYNC, ELASTIC] if x[3] == i]
            if len(_cloud) == 0:
                continue
            x, y = list(zip(*_cloud))
            x = [(y - start[ASYNC, ELASTIC]).total_seconds() for y in x]

            for xx in x:
                cloud_ax[i].axvline(xx, linewidth=0.1, color=colors[oi])
            label = ('FedBCD-i' if ELASTIC else "FedBCD") + " " + ("Sync" if ASYNC == 0 else fr"Async ($\beta={ASYNC/10}$)")
            cloud_ax[i].plot(x, y, color=colors[oi], label=label)
            cloud_ax[i].set_ylabel('Accuracy [%]')
            cloud_ax[i].set_ylim(0,100)
            cloud_ax[i].set_title(f'Cloud #{i+1}')
            cloud_ax[i].set_xlim(0,2000)

            #ax[i].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: x.strftime("%H:%M")))
        cloud_ax[len(cloud_ax)-1].set_xlabel('Solving time [s]')
        cloud_ax[0].legend()

    cloud_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    cloud_fig.savefig(f'/home/kari/public_html/fedbcd/cloud.png')
    

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(processName)14s %(levelname)7s: %(message)s", level=logging.DEBUG, datefmt='%H:%M:%S')
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    fh = logging.FileHandler('/tmp/fedbcd.log')
    formatter = logging.Formatter("%(asctime)s %(processName)14s %(levelname)7s: %(message)s")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(fh)

    warnings.filterwarnings("ignore") # ignore annoying warnings from pytorch 

    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "parallel":
                start_parallel(int(sys.argv[2]), bool(int(sys.argv[3])))
                sys.exit(0)
            elif sys.argv[1] == "plot":
                plot()
                sys.exit(0)
            elif sys.argv[1] == "coordinator":
                multiprocessing.current_process().name = f"COORDINATOR"
                coordinator_process(int(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4]))
                sys.exit(0)
            elif sys.argv[1] == "cloud":
                multiprocessing.current_process().name = f"CLOUD {sys.argv[4]}"
                cloud_process(int(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
                sys.exit(0)
            elif sys.argv[1] == "edge":
                multiprocessing.current_process().name = f"EDGE {sys.argv[4]}"
                edge_process(int(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4]), (sys.argv[5], int(sys.argv[6])))
                sys.exit(0)
        print("This script needs to be started in one of the following ways:")
        print("  python fedbcd.py coordinator <ASYNC> <ELASTIC> <NUM_CLOUD_SERVERS>")
        print("  python fedbcd.py cloud <ASYNC> <ELASTIC> <SERVER_ID> <NUM_EDGES> <COORDINATOR IP>")
        print("  python fedbcd.py edge <ASYNC> <ELASTIC> <EDGE_ID> <CLOUD SERVER IP> <CLOUD SERVER PORT>")
        print("  python fedbcd.py parallel <ASYNC> <ELASTIC>")
        print("  python fedbcd.py plot")
    except Exception as e:
        logger.exception("Fatal error")
