import socket
import typing

import numpy as np


class UdpSender:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, pos: typing.Tuple[int, int, int]):
        tracked_pos_with_empty_rotation = np.zeros(6)
        tracked_pos_with_empty_rotation[0] = pos[0]
        tracked_pos_with_empty_rotation[1] = -pos[1]
        tracked_pos_with_empty_rotation[2] = pos[2]
        self.sock.sendto(tracked_pos_with_empty_rotation.tobytes(), (self.ip, self.port))
