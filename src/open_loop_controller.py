#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray



class OpenLoopController(Node):

    def __init__(self):
        super().__init__('open_loop_controller')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0.0


    def timer_callback(self):
        self.velocity_callback()
        self.i += 1.0


    def velocity_callback(self):
        wheel_velocities = Float64MultiArray() 
        if(self.i<10.0):
            linear_vel = 3.0
        elif(self.i> 10.0 and self.i <40.0 ):
            linear_vel = 0.0
        elif(self.i> 40.0 and self.i <50.0 ):
            linear_vel = 3.0
        else:
             self.get_logger().info('stoping car at: "%s"' % self.i)
             linear_vel = 0.0
        wheel_velocities.data = [linear_vel,-linear_vel,linear_vel,-linear_vel]
        self.publisher_.publish(wheel_velocities)
       

def main(args=None):
    rclpy.init(args=args)
    
    open_loop_controller = OpenLoopController()

    rclpy.spin(open_loop_controller)
    open_loop_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


       
