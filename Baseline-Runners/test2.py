import socket

ROBOT_IP = "192.168.0.119"  # IP of the host running URSim
PORT = 30002

script = """
def myProgram():
  textmsg("Hello from external script!")
  sleep(1.0)
end

myProgram()
"""

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((ROBOT_IP, PORT))
    s.sendall(script.encode("utf-8"))