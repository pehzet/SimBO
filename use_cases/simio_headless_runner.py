from use_case_runner import UseCaseRunner
import torch
import paramiko
from scp import SCPClient
import os
def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

class SimioHeadlessRunner(UseCaseRunner):

    def __init__(self) -> None:
        super().__init__()
    
    def init_scp_client(self, server=None, port=None, user=None, password=None):
        if server is None:
            server = os.getenv("server")
        if port is None:
            port = os.getenv("port")
        if user is None:
            user = os.getenv("user")
        if password is None:
            password = os.getenv("password")
        
        ssh = create_ssh_client(server, port, user, password)
        self.ssh = ssh
        scp = SCPClient(ssh.get_transport())
        self.scp = scp
        return ssh, scp
    
    def put_file_via_scp(self, local_path, remote_path, scp = None):
        if scp is None:
            scp = self.scp
        try:
            scp.put(local_path, remote_path)
        except Exception as e:
            print(e)
            print("Error while uploading file")

    def get_file_via_scp(self, remote_path, local_path, scp = None):
        if scp is None:
            scp = self.scp
        try:
            scp.get(remote_path, local_path)
        except Exception as e:
            print(e)
            print("Error while downloading file")
    
    def send_ssh_command(self, command, ssh = None):
        if ssh is None:
            ssh = self.ssh
        try:
            stdin, stdout, stderr = ssh.exec_command(command)
            return stdin, stdout, stderr
        except Exception as e:
            print(e)
            print("Error while executing command")
    

    def load_file_from_local_path(self, local_path):
        with open(local_path, 'r') as f:
            return f.read()
    
    def save_file_to_local_path(self, local_path, content):
        with open(local_path, 'w') as f:
            f.write(content)
    
    def load_file_from_remote_path(self, remote_path, scp = None):
        if scp is None:
            scp = self.scp
        self.get_file_via_scp(remote_path, remote_path, scp)
        return self.load_file_from_local_path(remote_path)
    
    def save_file_to_remote_path(self, local_path, remote_path, content, scp = None):
        if scp is None:
            scp = self.scp
        self.save_file_to_local_path(local_path, content)
        self.put_file_via_scp(local_path, remote_path, scp)
    

    def eval(self):
        pass