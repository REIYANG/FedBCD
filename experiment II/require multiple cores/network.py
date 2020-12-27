import pickle
import socket
import logging

logger = logging.getLogger(__name__)


''' This functions here assist with the socket communications '''

class AsyncSocketWrapper(object):
    def __init__(self, sckt):
        logger.debug(f"Creating AsyncSocketWrapper({sckt})")
        self.sckt = sckt
        self.buffer = b''
        self.messages = []
        self.outgoing_buffer = b''

    def read(self):
        data = self.sckt.recv(10240)
        if data:
            #logger.debug(f"Read {len(data)} bytes")
            self.buffer += data
        else:
            return None

        while True:
            if len(self.buffer) >= 10:
                msglen = int(self.buffer[0:10].decode('iso-8859-1'), 16)
                if len(self.buffer) >= 10 + msglen:
                    # message is ready
                    self.messages.append(pickle.loads(self.buffer[10:(10+msglen)]))
                    self.buffer = self.buffer[(10+msglen):]
                    continue
            break
        return len(self.messages) > 0

    def shift(self):
        return self.messages.pop(0)

    def sendall(self, msg):
        data = pickle.dumps(msg)
        #logger.debug(f"Adding len {len(data)} message to buffer, size was {len(self.outgoing_buffer)}")
        data = "{0:010x}".format(len(data)).encode('iso-8859-1') + data
        self.outgoing_buffer += data

    def send(self):
        gone = self.sckt.send(self.outgoing_buffer[:10240])
        self.outgoing_buffer = self.outgoing_buffer[gone:] 
        #logger.debug(f"Sent {gone} bytes")
        return self.empty()

    def empty(self):
        return len(self.outgoing_buffer) == 0

    def close(self):
        self.sckt.close()

def recv_pickle(sckt):
    msglen = b''
    while len(msglen) < 10:
        m = sckt.recv(10 - len(msglen))
        if not m:
            logger.warning("Connection closed unexpectedly")
            raise BrokenPipeError("Connection closed unexpectedly")
        msglen += m

    msglen = int(msglen.decode('iso-8859-1'), 16)

    data = b''
    while len(data) < msglen:
        d = sckt.recv(min(1024, msglen - len(data)))
        if not d:
            logger.warning("Connection closed unexpectedly")
            raise BrokenPipeError("Connection closed unexpectedly")
        data += d

    return pickle.loads(data) # packet is complete - we received full pickle object

def send_pickle(sckt, data):
    data = pickle.dumps(data)
    data = "{0:010x}".format(len(data)).encode('iso-8859-1') + data
    sckt.sendall(data)
