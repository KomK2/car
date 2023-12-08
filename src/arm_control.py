#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class OpenLoopController(Node):

    def __init__(self):
        super().__init__('open_loop_controller')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        # timer_period = 1  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0.0
        self.arm_angle_callback()
        self.final_x = 0.0
        self.final_y = 0.0
        self.final_z = 0.0


    # def timer_callback(self):
    #     self.arm_angle_callback()
    #     self.i += 1.0

    def transformationMatrix(self, a , alpha, d, t) :
        return Matrix([
            [cos(t),( -sin(t)*cos(alpha)), (sin(t)*sin(alpha)), (a*cos(t))],
            [sin(t),( cos(t)*cos(alpha)), -(cos(t)*sin(alpha)), (a*sin(t))],
            [0, sin(alpha), cos(alpha), d],
            [0,0,0,1]
        ])
    
    def jacobian(self,theta_1,theta_2, theta_3, theta_4, theta_5, theta_6 ):

        x0= self.transformationMatrix(0 , rad(0), 0, rad(0))
        x1= self.transformationMatrix( 0 , rad( -90), 128, theta_1)
        x2= self.transformationMatrix(-612.7, rad(180), 0, rad(90)+ theta_2)
        x3 = self.transformationMatrix(-571.6, rad(180), 0,theta_3)
        x4 = self.transformationMatrix(0, rad(90), 163.9, rad(-90)+theta_4)
        x5 = self.transformationMatrix(0, rad(-90), 115.7, theta_5)
        x6 = self.transformationMatrix(0, rad(0), 192.2, theta_6)
        x = x1*x2*x3*x4*x5*x6

        k = Matrix([[0], [0], [1]])

        z0 = x0[:3, :3]*k
        z1 = (x0*x1)[:3, :3]*k
        z2 = (x0*x1*x2)[:3, :3]*k
        z3 = (x0*x1*x2*x3)[:3, :3]*k
        z4 = (x0*x1*x2*x3*x4)[:3, :3]*k
        z5 = (x0*x1*x2*x3*x4*x5)[:3, :3]*k
        z6 = (x0*x1*x2*x3*x4*x5*x6)[:3, :3]*k


        o0 = x0[:3, -1]
        o1 = (x0*x1)[:3, -1]
        o2 = (x0*x1*x2)[:3, -1]
        o3 = (x0*x1*x2*x3)[:3, -1]
        o4 = (x0*x1*x2*x3*x4)[:3, -1]
        o5 = (x0*x1*x2*x3*x4*x5)[:3, -1]
        o6 = (x0*x1*x2*x3*x4*x5*x6)[:3, -1]


        j1 =Matrix([[z0.cross(o6-o0)],[z0]])
        j2 =Matrix([[z1.cross(o6-o1)],[z1]])
        j3 =Matrix([[z2.cross(o6-o2)],[z2]])
        j4 =Matrix([[z3.cross(o6-o3)],[z3]])
        j5 =Matrix([[z4.cross(o6-o4)],[z4]])
        j6 =Matrix([[z5.cross(o6-o5)],[z5]])

        j =  Matrix.hstack(j1,j2,j3,j4,j5,j6)
        return j
 
    def arm_angle_callback(self):
        x_coordinate = []
        y_coordinate = []
        z_coordinate = []


        x_1=0
        x_2= 800

        y_1=356.12
        y_2=500

        z_1=1428
        z_2=1300

        x_dot = (x_1-x_2)/20
        y_dot = (y_1-y_2)/20
        z_dot = (z_1-z_2)/20


        q = np.zeros((6,1))
        Ti = 0
        dt = 0.001
        time = 20
        j = self.jacobian(0.001,0.001,0.001,0.001,0.001,0.001)
        j_float = np.matrix(j).astype(np.float64)
        j_inv = np.linalg.pinv(j_float)


        while (Ti < time):
            # epsilon matrix
            e = np.matrix([[x_dot], [y_dot], [z_dot], [0], [0], [0]])

            q_dot = j_inv * e
            q = q + q_dot * dt

            [q1, q2, q3, q4, q5, q6] = [q[i].item() for i in range(6)]
            # [q1, q2, q3, q4, q5, q6] = [q[i].item() if any(q[i].item() != 0 for i in range(6)) else q[i]+0.01 for i in range(6)]
            

            temp = self.jacobian(q1,q2,q3,q4,q5,q6)
            temp_float = np.matrix(temp).astype(np.float64)
            det =np.linalg.det(temp_float)
            if(abs(det) <0.001):
                q1 = q1 + 0.01
                q2 = q2 + 0.01
                q3 = q3 + 0.01
                q4 = q4 + 0.01
                q5 = q5 + 0.01
                q6 = q6 + 0.01   
                temp = self.jacobian(q1,q2,q3,q4,q5,q6)
                j = temp
                temp_float = np.matrix(temp).astype(np.float64)
                j_float = temp_float
            else :
                j = temp 
                j_float = temp_float
            

            j_float = np.matrix(j).astype(np.float64)
            j_inv = np.linalg.inv(temp_float)
            
            to0= self.transformationMatrix(0 , rad(0), 0, rad(0))
            t1= self.transformationMatrix(0 ,  rad( -90), 128, q1)
            t2= self.transformationMatrix(-612.7, rad(180), 0, rad(90)+ q2 )
            t3 = self.transformationMatrix(-571.6, rad(180), 0, q3)
            t4 = self.transformationMatrix(0,rad(90), 163.9, rad(-90)+q4)
            t5 = self.transformationMatrix(0,rad(-90),115.7, q5)
            t6 = self.transformationMatrix(0, rad(0), 192.2, q5)

            t = to0*t1*t2*t3*t4*t5*t6

            pprint(t)
            print(t[1,3])


            x_coordinate.append(t[0,3])
            y_coordinate.append(t[1,3])
            z_coordinate.append(t[2,3])

            self.final_x = t[0,3]
            self.final_y = t[1,3]
            self.final_z = t[2,3]

            Ti = Ti + dt

            wheel_velocities = Float64MultiArray() 
            wheel_velocities.data = [0.0,0.0,q1,q2,q3,q4,q5,q6,0.0]
            # self.get_logger().info('publishing postions: "%s"' % self.i)
            self.publisher_.publish(wheel_velocities)
            
            

        print(f"x :{self.final_x}, y: {self.final_y}, z :{self.final_z}")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_coordinate, y_coordinate, z_coordinate, label='End Effector')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('UR10 drawing a circle')
        ax.legend()
        plt.show()
        

def main(args=None):
    rclpy.init(args=args)
    
    open_loop_controller = OpenLoopController()

    rclpy.spin(open_loop_controller)
    open_loop_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()