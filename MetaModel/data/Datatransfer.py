import os
import socket

# Download location
download_path = os.getcwd()

# Server details for rsync command
server_username = os.environ.get('USER', os.environ.get('USERNAME', 'username'))

# Get hostname
def get_hostname():
    try:
        return socket.gethostname()
    except:
        return "server"

server_hostname = get_hostname()

# Local file path input
local_path = r"\Users\lhfor\AppData\Local\Programs\Python\Python38-32\Helix2TextDataRetreival\Dataset\protein_to_dna_sequence_descriptions.csv"
    
# Build the transfer command using SCP instead of rsync
scp_cmd = f"scp \"{local_path}\" {server_username}@{server_hostname}:{download_path}"

print("\nUpload data to server (Run on local machine):")
print(scp_cmd)



download_path2 = r"\Users\lhfor\AppData\Local\Programs\Python\Python38-32\Helix2TextDataRetreival\Dataset"

scp_cmd2 = f"scp {server_username}@{server_hostname}:\"/path/to/remote/file\" \"{download_path2}\""

print("\nDownload data to local (Run on Server):")
print(scp_cmd2)