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

        # Value of pi from the symbolic library for convenience
        spi = sp.pi

        # Define DH table

        self.DH_params = []

        # test
        # self.DH_params.append([128, self.q1, 0, -spi/2])
        # self.DH_params.append([0, self.q2+(spi/2),-612.7, 2*spi])
        # self.DH_params.append([0, self.q3, -571.6, spi*2])
        # self.DH_params.append([163.9, self.q4 - (spi/2), 0, spi/2])
        # self.DH_params.append([115.7, self.q5, 0, -spi/2])
        # self.DH_params.append([192.2, self.q6, 0, 0])
        # self.DH_params.append([0, 0, 0, 0])

        self.DH_params.append([128, self.q1, 0, -spi/2])
        self.DH_params.append([0, self.q2 -(spi/2),612.7, 2*spi])
        self.DH_params.append([0, self.q3, 571.6, -spi*2])
        self.DH_params.append([163.9, self.q4 +(spi/2), 0, spi/2])
        self.DH_params.append([115.7, self.q5, 0, -spi/2])
        self.DH_params.append([192.2, self.q6, 0, 0])
        self.DH_params.append([0, 0, 0, 0])

        self.arm_angle_callback()


    def DH_trans_matrix(self, params):

        d, theta, a, alpha = (params[0], params[1], params[2], params[3])

        mat = sp.Matrix([[sp.cos(theta), -1*sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha),    a*sp.cos(theta)],
                        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha),    -1*sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
                        [0,             sp.sin(alpha),                  sp.cos(alpha),                  d],
                        [0,             0,                              0,                              1]])

        return mat
    
    # list of transformations
    def joint_transforms(self, DH_params):
        transforms = []

        transforms.append(sp.eye(4)) #Assuming the first first joint is at the origin

        for el in DH_params:

            transforms.append(self.DH_trans_matrix(el))

        return transforms
    

    # jacobian matrix to calculate 
    def jacobian_expr(self, DH_params):

        transforms = self.joint_transforms(DH_params)

        # base transformation
        trans_EF = transforms[0]

        for mat in transforms[1:]:

            trans_EF = trans_EF * mat

        pos_EF = trans_EF[0:3,3]

        J = sp.zeros(6, self.DOF)

        for joint in range(self.DOF):

            trans_joint = transforms[0]

            for mat in transforms[1:joint+1]:

                trans_joint = trans_joint*mat

            z_axis = trans_joint[0:3,2]

            pos_joint = trans_joint[0:3,3]

            Jv = z_axis.cross(pos_EF - pos_joint)

            Jw = z_axis

            J[0:3,joint] = Jv
            J[3:6,joint] = Jw

        J = sp.simplify(J)
        return J
    

    def jacobian_subs(self, joints, jacobian_sym):

        # Convert to list if it's an ndarray
        if (isinstance(joints, np.ndarray)):
            joints = joints.flatten().tolist()

        J_l = jacobian_sym

        J_l = J_l.subs(self.q1, joints[0])
        J_l = J_l.subs(self.q2, joints[1])
        J_l = J_l.subs(self.q3, joints[2])
        J_l = J_l.subs(self.q4, joints[3])
        J_l = J_l.subs(self.q5, joints[4])
        J_l = J_l.subs(self.q6, joints[5])

        return J_l
    
    def trans_EF_eval(self, joints, DH_params):

    # Convert to list if it's an ndarray
        if (isinstance(joints, np.ndarray)):
            joints = joints.flatten().tolist()

        transforms = self.joint_transforms(DH_params)

        trans_EF = transforms[0]

        for mat in transforms[1:]:

            trans_EF = trans_EF * mat

        trans_EF_cur = trans_EF

        trans_EF_cur = trans_EF_cur.subs(self.q1, joints[0])
        trans_EF_cur = trans_EF_cur.subs(self.q2, joints[1])
        trans_EF_cur = trans_EF_cur.subs(self.q3, joints[2])
        trans_EF_cur = trans_EF_cur.subs(self.q4, joints[3])
        trans_EF_cur = trans_EF_cur.subs(self.q5, joints[4])
        trans_EF_cur = trans_EF_cur.subs(self.q6, joints[5])

        return trans_EF_cur
    
    def plot_pose(self, joints, DH_params):

    # Convert to list if it's an ndarray
        if (isinstance(joints, np.ndarray)):
            joints = joints.flatten().tolist()

        transforms = self.joint_transforms(DH_params)

        trans_EF = self.trans_EF_eval(joints, DH_params)

        pos_EF = trans_EF[0:3,3]

        xs = []
        ys = []
        zs = []

        J = sp.zeros(6, self.DOF)

        for joint in range(self.DOF):

            trans_joint = transforms[0]

            for mat in transforms[1:joint+1]:

                trans_joint = trans_joint*mat

            pos_joint = trans_joint[0:3,3]

            pos_joint = pos_joint.subs(self.q1, joints[0])
            pos_joint = pos_joint.subs(self.q2, joints[1])
            pos_joint = pos_joint.subs(self.q3, joints[2])
            pos_joint = pos_joint.subs(self.q4, joints[3])
            pos_joint = pos_joint.subs(self.q5, joints[4])
            pos_joint = pos_joint.subs(self.q6, joints[5])



            xs.append(pos_joint[0])
            ys.append(pos_joint[1])
            zs.append(pos_joint[2])

        xs.append(pos_EF[0])
        ys.append(pos_EF[1])
        zs.append(pos_EF[2])

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim3d(-60,60)
        ax.set_ylim3d(-60,60)
        ax.set_zlim3d(0, 120)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        ax.plot(xs, ys, zs)

    def joint_limits(self,joints):
            
        # Joint 1
        if (joints[0] < -2*pi/3):
            
            joints[0] = -2*pi/3
            
        elif (joints[0] > 2*pi/3):
            
            joints[0] = 2*pi/3
            
        
        # Joint 2
        if (joints[1] < -0.95*pi):
            
            joints[1] = -0.95*pi
            
        elif (joints[1] > 0):
            
            joints[1] = 0
            
        # Joint 3
        if (joints[2] < -0.463*pi):
            
            joints[2] = -0.463*pi
            
        elif (joints[2] > 0.48*pi):
            
            joints[2] = 0.48*pi
            
        # Joint 4
        if (joints[3] < -0.97*pi):
            
            joints[3] = -0.97*pi
            
        elif (joints[3] > 0.97*pi):
            
            joints[3] = 0.97*pi
                

        # Joint 5
        if (joints[4] < -3*pi/2):
            
            joints[4] = -3*pi/2
            
        elif (joints[4] > 3*pi/2):
            
            joints[4] = 3*pi/2
            
        # Joint 6
        if (joints[5] < -0.95*pi):
            
            joints[5] = -0.95*pi
            
        elif (joints[5] > 0.95*pi):
            
            joints[5] = 0.95*pi
                
        return joints
    

    # joints_init is the current joint values for the robot
    # target is the desired transformation matrix at the end effector
    # set no_rotation to true if you only care about end effector position, not rotation
    # set joint_lims to false if you want to allow the robot to ignore joint limits
    # This is currently super slow since it's using all symbolic math
    
    def i_kine(self, joints_init, target, DH_params, error_trace=True, no_rotation=False, joint_lims=False):
        
        joints = joints_init
        
        xr_desired = target[0:3,0:3] #target postion
        xt_desired = target[0:3,3]    #target orientation
        
        x_dot_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        e_trace = []
        
        iters = 0
        
        print("Finding symbolic jacobian")
        
        # We only do this once since it's computationally heavy
        jacobian_symbolic = self.jacobian_expr(DH_params)
        
        print("Starting IK loop")
        
        final_xt = 0
        
        while(1):
            
            jac = self.jacobian_subs(joints, jacobian_symbolic)
            
            jac = np.array(jac).astype(np.float64)
            
            trans_EF_cur = self.trans_EF_eval(joints, DH_params) 

            print(joints)
            # ------------------------------------------------------------
            

            arm_angles = Float64MultiArray() 
            arm_angles.data = [0.0,0.0,joints[0][0],joints[1][0],joints[2][0],joints[3][0],joints[4][0],0.0,0.0]
                # self.get_logger().info('publishing postions: "%s"' % self.i)
            self.publisher_.publish(arm_angles)
                    
            trans_EF_cur = np.array(trans_EF_cur).astype(np.float64)
            
            
            xr_cur = trans_EF_cur[0:3,0:3] 
            xt_cur = trans_EF_cur[0:3,3]
            
            final_xt = xt_cur
                    
            xt_dot = xt_desired - xt_cur
            
            
            # Find error rotation matrix
            R = xr_desired @ xr_cur.T
            
                                
            # convert to desired angular velocity
            v = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1)/2)
            r = (0.5 * sin(v)) * np.array([[R[2,1]-R[1,2]],
                                        [R[0,2]-R[2,0]],
                                        [R[1,0]-R[0,1]]])
            
            
            # The large constant just tells us how much to prioritize rotation
            xr_dot = 200 * r * sin(v)
            
            # use this if you only care about end effector position and not rotation
            if (no_rotation):
                
                xr_dot = 0 * r
            
            xt_dot = xt_dot.reshape((3,1))
                    
            x_dot = np.vstack((xt_dot, xr_dot))
                    
            x_dot_norm = np.linalg.norm(x_dot)
            
            #print(x_dot_norm)
                    
            if (x_dot_norm > 25):
                
                x_dot /= (x_dot_norm/25)
                
            x_dot_change = np.linalg.norm(x_dot - x_dot_prev)
                        
            # This loop now exits if the change in the desired movement stops changing
            # This is useful for moving close to unreachable points
            if (x_dot_change < 0.005):
                
                break
                
            x_dot_prev = x_dot
                
            e_trace.append(x_dot_norm)
                
            Lambda = 12
            Alpha = 1
                            
            joint_change = Alpha * np.linalg.inv(jac.T@jac + Lambda**2*np.eye(self.DOF)) @ jac.T @ x_dot
            
            joints += joint_change
            
            if (joint_lims): joints = self.joint_limits(joints)

            print(joints)

            # arm_angles = Float64MultiArray() 
            # arm_angles.data = [0.0,0.0,joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],0.0]
            #     # self.get_logger().info('publishing postions: "%s"' % self.i)
            # self.publisher_.publish(arm_angles)
            
            iters += 1
                    
        print("Done in {} iterations".format(iters))
        
        
        print("Final position is:")
        print("test1")
        print(final_xt)
            
        return (joints, e_trace) if error_trace else joints
    

    
    def arm_angle_callback(self):
        joints = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

        target = np.array([[1, 0, 0, 1300.0],
                        [0, 1, 0, 356.0],
                        [0, 0, 1, 128.0],
                        [0, 0, 0, 1]])

        new_j, e_trace = self.i_kine(joints, target, self.DH_params, error_trace=True)
        print(f"joint values {new_j}")

        self.plot_pose(new_j, self.DH_params)

        plt.figure(figsize=(8,8))
        plt.plot(e_trace)
        plt.title('Error Trace')
            
        # arm_angles = Float64MultiArray() 
        # arm_angles.data = [0.0,0.0,-2.0943951 , -1.60491838 , 1.10762422 , 1.94664517, 1.18110623, 0.53379638,0.0]
        # # arm_angles.data = [0.0,0.0,1.57 , 0.0 ,0.0 ,0.0, 0.0,0.0,0.0]
        #     # self.get_logger().info('publishing postions: "%s"' % self.i)
        # self.publisher_.publish(arm_angles)
            

def main(args=None):
    rclpy.init(args=args)
    
    open_loop_controller = OpenLoopController()

    rclpy.spin(open_loop_controller)
    open_loop_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()