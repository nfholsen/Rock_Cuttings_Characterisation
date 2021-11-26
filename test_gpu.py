import torch
from paramiko import SSHClient

if torch.cuda.is_available():
    
    number_of_devices = torch.cuda.device_count()

    print('Number of devices :', number_of_devices)

    for device in range(number_of_devices):
        print(torch.cuda.get_device_name(device))

else : 
    print('Cuda not available')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Connect
client = SSHClient()
client.load_system_host_keys()
client.connect('Tomo-graph2.intranet.epfl.ch', username='nolsen', password='')