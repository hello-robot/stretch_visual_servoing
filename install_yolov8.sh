#!/bin/bash

echo "Install ultralytics that includes YOLOv8..."
echo "pip3 install ultralytics"
pip3 install ultralytics
echo ""

echo "Install PyZMQ connecting to an off-robot machine with a large GPU..."
echo "pip3 install pyzmq"
pip3 install pyzmq
echo ""

echo "Install urchin URDF processor for off-robot fingertip detetion..."
echo "pip3 install urchin"
pip3 install urchin
echo ""

echo "Install pyusb for working with a webcamera..."
echo "pip3 install pyusb"
pip3 install pyusb
echo ""

echo "Open a TCP port in the firewall for connecting with an off-robot machine with a large GPU"
echo "sudo ufw allow 4405/tcp"
sudo ufw allow 4405/tcp
echo ""

echo "Open a TCP port in the firewall for connecting with an off-robot machine with a large GPU"
echo "sudo ufw allow 4010/tcp"
sudo ufw allow 4010/tcp
echo ""
