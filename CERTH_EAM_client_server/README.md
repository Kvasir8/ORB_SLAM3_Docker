# CERTH's server-client for EAM

The server consumes messages from multiple clients that produce frame messages from a realsense camera feed.
## Server side 

Go to CERTH_EAM_server folder and build the certh_server docker
```
sudo docker build -t certh_server .
```
There is docker-compose.yml to run it separately. You will need to give display privillages to the docker
```
xhost +local:docker
sudo docker-compose up
```

## Client side

Go to CERTH_EAM_client folder and build the certh_client docker
```
sudo docker build -t certh_client . 
```
There is docker-compose.yml to run it separately.   
```
sudo docker-compose up
```
## Server + Client

The docker-compose.yml file in the CERTH_EAM_client_server folder will run both dockers. You will need to give display privillages to the docker.
```
xhost +local:docker
sudo docker-compose up
```

###Remember to configure the IP and Port in the environment sections of the docker-compose.yml.
On the server set the IP address 0.0.0.0 so that it can receive messages from any client.
In the client set the IP of the server where the message should be sent. For example, if the IP toy server is 192.168.10.20, the client should know that the message will be sent there.
If the server and client are on different networks do not forget to configure the firewall and NAT respectively.

###Make sure to pass all the video devices on docker-compose.yaml file  (e.g /dev/videoX:/dev/videoX)
