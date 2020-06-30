import CLI

socket = CLI.NetCon('localhost',5557)
dd =CLI.getDataRSP(socket,"93000000",100000)
CLI.setDataRSP(socket,"93000000",100000, dd)
CLI.NetClose(socket)