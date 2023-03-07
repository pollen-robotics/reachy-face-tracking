# Reachy face tracking

The face tracking project is an autonomous application in which Reachy will detect faces in its field of view and track the detected face that is closet to it.

This can be used as is to make demo with the robot, show the capabilities of the [Orbita actuator](http://localhost:1313/sdk/first-moves/head/#reachys-neck-orbita-actuator) or serve as a brick to build more complex and interactive applications.

## How to install and run the application

### Install

Clone the repository and install it on your Reachy with pip.

```bash
cd ~/dev
git clone https://github.com/pollen-robotics/reachy-face-tracking.git
cd reachy-face-tracking
pip3 install -e .
```

### Run the application
You can run the application directly with Python.
```bash
cd ~/dev/reachy-face-tracking
python3 -m reachy-face-tracking.face_tracking_launcher
```

## Documentation
Consult the [dedicated page](https://docs.pollen-robotics.com/sdk/application/face-tracking/) on Reachy's documentation to learn more on this application.
