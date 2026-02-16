import socket

def check_port(host, port):
    try:
        with socket.create_connection((host, port), timeout=5) as s:
            print(f"Connection to {host}:{port} successful")
    except Exception as e:
        print(f"Failed to connect to {host}:{port}: {e}")

if __name__ == "__main__":
    check_port("3.87.216.2", 7687)
