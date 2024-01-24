# car
Commands to run the package after adding it to the work space:
    source /opt/ros/galactic/setup.bash
    colcon build 
    source install/local_setup.bash

To spawn the robot in gazebo:
    ros2 launch car gazebo.launch.py

To run teleop in gazebo:
    ros2 run car teleop.py
    ros2 run car open_loop_controller.py