#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from numpy import cos, sin, pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sympy as sp



class OpenLoopController(Node):

    def __init__(self):
        super().__init__('open_loop_controller')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        
        
        self.DOF = 6

        self.q1, self.q2, self.q3, self.q4, self.q5, self.q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

        spi = sp.pi

        
        #DH table
        self.dh_parameters = []

        self.dh_parameters.append([128, self.q1, 0, -spi/2])
        self.dh_parameters.append([0, self.q2 -(spi/2),612.7, 2*spi])
        self.dh_parameters.append([0, self.q3, 571.6, -spi*2])
        self.dh_parameters.append([163.9, self.q4 +(spi/2), 0, spi/2])
        self.dh_parameters.append([115.7, self.q5, 0, -spi/2])
        self.dh_parameters.append([192.2, self.q6, 0, 0])
        self.dh_parameters.append([0, 0, 0, 0])

        self.arm_angle_callback()


    def transformation_matrix(self, params):
        d, theta, a, alpha = (params[0], params[1], params[2], params[3])

        mat = sp.Matrix([[sp.cos(theta), -1*sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha),    a*sp.cos(theta)],
                        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha),    -1*sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
                        [0,             sp.sin(alpha),                  sp.cos(alpha),                  d],
                        [0,             0,                              0,                              1]])

        return mat
    

    # list of transformations
    def joint_transforms(self, dh_parameters):
        transforms = []

        transforms.append(sp.eye(4)) #At origin

        for l in dh_parameters:

            transforms.append(self.transformation_matrix(l))

        return transforms
    

    # jacobian matrix to calculate 
    def jacobian_expr(self, dh_parameters):

        transforms = self.joint_transforms(dh_parameters)

        # base transformation
        final_transformation = transforms[0]

        for mat in transforms[1:]:

            final_transformation = final_transformation * mat

        end_effector_position = final_transformation[0:3,3]

        j = sp.zeros(6, self.DOF)

        for joint in range(self.DOF):

            trans_joint = transforms[0]

            for mat in transforms[1:joint+1]:

                trans_joint = trans_joint*mat

            z_axis = trans_joint[0:3,2]

            pos_joint = trans_joint[0:3,3]

            jv = z_axis.cross(end_effector_position - pos_joint)

            jw = z_axis

            j[0:3,joint] = jv
            j[3:6,joint] = jw

        j = sp.simplify(j)
        return j
    

    def jacobian_calculation(self, joints, jacobian_equation):
        
        if (isinstance(joints, np.ndarray)):
            joints = joints.flatten().tolist()

        jac = jacobian_equation

        jac = jac.subs(self.q1, joints[0])
        jac = jac.subs(self.q2, joints[1])
        jac = jac.subs(self.q3, joints[2])
        jac = jac.subs(self.q4, joints[3])
        jac = jac.subs(self.q5, joints[4])
        jac = jac.subs(self.q6, joints[5])

        return jac
    


    def transformation_end_effector(self, joints, dh_parameters):
        if (isinstance(joints, np.ndarray)):
            joints = joints.flatten().tolist()

        # gives list of  all transformations
        transforms = self.joint_transforms(dh_parameters)

        #Intializing variable with identity matrix
        transformation_of_end_effector = transforms[0]

        for mat in transforms[1:]:

            transformation_of_end_effector = transformation_of_end_effector * mat

        sub_transform = transformation_of_end_effector

        sub_transform = sub_transform.subs(self.q1, joints[0])
        sub_transform = sub_transform.subs(self.q2, joints[1])
        sub_transform = sub_transform.subs(self.q3, joints[2])
        sub_transform = sub_transform.subs(self.q4, joints[3])
        sub_transform = sub_transform.subs(self.q5, joints[4])
        sub_transform = sub_transform.subs(self.q6, joints[5])

        return sub_transform

    
    def ik_calculation(self, intial_joint_angles, target, dh_parameters, error_trace=True):
        
        joints = intial_joint_angles
        
        desired_postion = target[0:3,0:3] #target postion
        desired_orentation = target[0:3,3]    #target orientation
        
        x_dot_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        error_trace = []
        
        itr = 0
        
        jacobian_symbolic = self.jacobian_expr(dh_parameters)
        
        final_orientation = 0
        
        while(1):
            
            jac = self.jacobian_calculation(joints, jacobian_symbolic)
            
            jac = np.array(jac).astype(np.float64)
            
            trans_EF_new = self.transformation_end_effector(joints, dh_parameters) 

            print(joints)
            

            arm_angles = Float64MultiArray() 
            arm_angles.data = [0.0,0.0,joints[0][0],joints[1][0],joints[2][0],0.0,joints[3][0],joints[4][0],0.0,0.0]
            self.publisher_.publish(arm_angles)
                    
            trans_EF_new = np.array(trans_EF_new).astype(np.float64)
            
            
            new_postion = trans_EF_new[0:3,0:3] 
            new_orientation = trans_EF_new[0:3,3]
            
            final_orientation = new_orientation
                    
            xt_dot = desired_orentation - new_orientation
            
            
            R = desired_postion @ new_postion.T
            
            v = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1)/2)
            r = (0.5 * sin(v)) * np.array([[R[2,1]-R[1,2]],
                                        [R[0,2]-R[2,0]],
                                        [R[1,0]-R[0,1]]])
            
            
            rotation_dot = 200 * r * sin(v)
            
            xt_dot = xt_dot.reshape((3,1))
                    
            x_dot = np.vstack((xt_dot, rotation_dot))
                    
            err = np.linalg.norm(x_dot)
            
                    
            if (err > 25):
                
                x_dot /= (err/25)
                
            x_dot_change = np.linalg.norm(x_dot - x_dot_prev)

            if (x_dot_change < 0.005):
                
                break
                
            x_dot_prev = x_dot
                
            error_trace.append(err)
                
            lam = 13
            alpha = 1
                            
            joint_change = alpha * np.linalg.inv(jac.T@jac + lam**2*np.eye(self.DOF)) @ jac.T @ x_dot
            
            joints += joint_change
            

            print(joints)
            
            itr += 1
        
        
        print("Final position is:")
        print(final_orientation)
            
        return (joints, error_trace) if error_trace else joints
    

    
    def arm_angle_callback(self):
        joints = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

        target = np.array([[1, 0, 0, 1300.0],
                        [0, 1, 0, 356.0],
                        [0, 0, 1, 128.0],
                        [0, 0, 0, 1]])

        new_j, error_trace = self.ik_calculation(joints, target, self.dh_parameters, error_trace=True)
        print(f"joint values {new_j}")


def main(args=None):
    rclpy.init(args=args)
    
    open_loop_controller = OpenLoopController()

    rclpy.spin(open_loop_controller)
    open_loop_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()