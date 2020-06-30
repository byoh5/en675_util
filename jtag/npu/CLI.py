import socket
import binascii

def NetCon(host, port):

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((host,port))

    return client_socket

def NetClose(socket):
    socket.close()


def calc_checksum(string):
    '''
    Calculates checksum for sending commands to the ELKM1.
    Sums the ASCII character values mod256 and takes
    the Twos complement
    '''
    sum = 0

    for i in range(len(string)):
        sum = sum + ord(string[i])

    temp = sum % 256  # mod256
    rem = temp ^ 256  # inverse
    cc1 = hex(rem)
    cc = cc1.upper()
    p = len(cc)
    return cc[p - 2:p]


def getDataRSP(socket,addr,size):

    message = "m"+addr+","+"%08x" %(size)
    chksum = calc_checksum(message)
    new_msg = "$"+message+"#"+chksum
    # print("send:"+new_msg)
    socket.send(new_msg.encode())
    decoded_data =""

    while True:
        rdata = socket.recv(1024)
        decoded_data += rdata.decode()

        if (decoded_data.find('$') != -1) and (decoded_data.find('#') != -1):
            socket.send('+'.encode())
            tmp = decoded_data.split('$')
            tmp = tmp[1]
            tmp = tmp.split('#')
            tmp = tmp[0]
            print(tmp)
            break

    bi = binascii.a2b_hex(tmp)

    return bi


def setDataRSP(socket, addr, size, data):

    asc_data = binascii.b2a_hex(data)
    asc_data = str(asc_data)
    tmp = asc_data.split('\'')
    tmp = tmp[1]
    tmp = tmp.split('\'')
    asc_data = tmp[0]
    message = "M" + addr + "," + "%08x" %(size) +":"+ str(asc_data)
    chksum = calc_checksum(message)
    new_msg = "$" + message + "#" + chksum
    # print("send:" + new_msg)
    socket.send(new_msg.encode())
    decoded_data = ""

    while True:
        rdata = socket.recv(1024)
        decoded_data += rdata.decode()

        if (decoded_data.find('O') != -1) and (decoded_data.find('K') != -1):
            socket.send('+'.encode())
            print(decoded_data)
            break
    return decoded_data

# --example
# socket = NetCon('localhost',5557)
# dd =getDataRSP(socket,"93000000",100000)
# setDataRSP(socket,"93000000",100000, dd)



