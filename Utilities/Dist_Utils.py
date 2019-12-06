#!/usr/bin/env python
# coding: utf-8
"""
module: utilities for distributed training, including:
    find an available port on current node
"""
# __all__ = ()

def get_local_addr():
    import socket
    addr = socket.gethostbyname(socket.gethostname())
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return addr, port