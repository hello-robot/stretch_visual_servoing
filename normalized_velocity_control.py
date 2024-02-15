
import stretch_body.robot as rb
from stretch_body.hello_utils import *
from stretch_body.robot_params import RobotParams
import threading

import time


###################################################################
# Derived from gamepad_joints.py

class CommandBase:
    def __init__(self):
        """Base motion command class
        """
        self.params = RobotParams().get_params()[1]['base']
        self.dead_zone = 0.0001
        self._prev_set_vel_ts = None
        self.max_linear_vel = self.params['motion']['max']['vel_m']
        self.max_rotation_vel = 0.5 # rad/s
        #self.max_rotation_vel = 1.90241 # rad/s
        self.normal_linear_vel = self.params['motion']['default']['vel_m']
        self.normal_rotation_vel = self.max_rotation_vel*0.4
        self.precision_mode = False 
        self.fast_base_mode = False
        self.acc = self.params['motion']['max']['accel_m']

        # Precision mode params
        self.precision_max_linear_vel = 0.02 # m/s Very precise: 0.01
        self.precision_max_rot_vel = 0.08 # rad/s Very precise: 0.04
    
    def command_stick_to_motion(self, x, y, robot):
        """Convert a stick axis value to robot base's tank driving motion.

        Args:
            x (float): Range [-1.0,+1.0], control rotation speed
            y (float): Range [-1.0,+1.0], control linear speed
            robot (robot.Robot): Valid robot instance
        """
        if abs(x) < self.dead_zone:
            x = 0
        if abs(y) < self.dead_zone:
            y = 0
        # x = to_parabola_transform(x)
        # y = to_parabola_transform(y) 
        
        # Standard Mode
        if not self.precision_mode:
            self.start_pos = None
            self.start_theta = None
            v_m, w_r = self._process_stick_to_vel(x,y, robot)
            robot.base.set_velocity(v_m, w_r, a=self.acc)
            self._prev_set_vel_ts = time.time()
        else:
        # Precision Mode
            if self.start_pos is None:
                self.start_pos = [robot.base.status['x'],robot.base.status['y']]
                self.start_theta = robot.base.status['theta']
                self.target_position = self.start_pos
                self.target_theta = self.start_theta  
            yv = map_to_range(abs(y),0,self.precision_max_linear_vel)
            xv = map_to_range(abs(x),0,self.precision_max_rot_vel)
            if x<0:
                xv = -1*xv
            if y<0:
                yv = -1*yv
            if abs(x)>abs(y):
                self._step_precision_rotate(xv, robot)
            else:
                self._step_precision_translate(yv, robot)
            # Update the previous_time for the next iteration
            self._prev_set_vel_ts = time.time()
    
    
    def stop_motion(self, robot):
        """Stop the joint motion. To be used when ever the controller is idle/no-inputs
        to stop unnecessary robot motion.

        Args:
            robot (robot.Robot): Valid robot instance
        """
        robot.base.set_velocity(0, 0, a=self.acc)

    def is_fastbase_safe(self, robot):
        """Check if the base is fast navigation mode safe

        Args:
            robot (robot.Robot): Valid robot instance

        Returns:
            bool: True if fast navigation safe
        """
        arm = robot.arm.status['pos'] < (robot.get_stow_pos('arm') + 0.1) # check if arm pos under stow pose + 0.1 m
        lift = robot.lift.status['pos'] < (robot.get_stow_pos('lift') + 0.05) # check if lift pos is under stow pose + 0.05m
        return arm and lift

    def _process_stick_to_vel(self, x, y, robot):
        max_linear_vel = self.normal_linear_vel
        max_rotation_vel = self.normal_rotation_vel
        if self.fast_base_mode and self.is_fastbase_safe(robot):
            max_linear_vel =  self.max_linear_vel
            max_rotation_vel = self.max_rotation_vel
        v_m = map_to_range(abs(y), 0, max_linear_vel)
        if y<0:
            v_m = -1*v_m
        x = -1*x
        w_r = map_to_range(abs(x), 0, max_rotation_vel)
        if x<0:
            w_r = -1*w_r

        w_r = -1 * w_r
        return v_m, w_r
            
    def _step_precision_rotate(self, xv, robot):
        # Calculate the time elapsed since the last iteration
        current_time = time.time()
        elapsed_time = current_time - self._prev_set_vel_ts 

        # Calculate the desired change in position to achieve the desired velocity
        desired_theta_change = xv * elapsed_time

        # Set the control_effort as the position setpoint for the joint
        if abs(xv)>=0:
            robot.base.rotate_by(-1*desired_theta_change)

    def _step_precision_translate(self, yv, robot):
        
        # Calculate the time elapsed since the last iteration
        current_time = time.time()
        elapsed_time = current_time - self._prev_set_vel_ts 

        # Calculate the desired change in position to achieve the desired velocity
        desired_position_change = yv * elapsed_time

        # Set the control_effort as the position setpoint for the joint
        if abs(yv)>=0:
            robot.base.translate_by(desired_position_change)
            
class CommandLift:
    def __init__(self):
        """Lift motion command class.
        """
        self.params = RobotParams().get_params()[1]['lift']
        self.dead_zone = 0.0001
        self._prev_set_vel_ts = None
        self.max_linear_vel = self.params['motion']['max']['vel_m']
        self.precision_mode = False
        self.acc = self.params['motion']['max']['accel_m']
        
        # Precision mode params
        self.start_pos = None
        self.target_position = self.start_pos
        self.precision_kp = 0.5 # Very Precise: 0.5
        self.precision_max_vel = 0.04 # m/s Very Precise: 0.02 m/s
        self.stopped_for_prec = False 
    
    def command_stick_to_motion(self, x, robot):
        """Convert a stick axis value to robot lift motion.

        Args:
            x (float): Range [-1.0,+1.0], control lift speed
            robot (robot.Robot): Valid robot instance
        """
        if abs(x) < self.dead_zone:
            x = 0
        # x = to_parabola_transform(x)
        
        # Standard Mode
        if not self.precision_mode:
            self.start_pos = None
            v_m = self._process_stick_to_vel(x)
            robot.lift.set_velocity(v_m, a_m=self.acc)
            self._prev_set_vel_ts = time.time()
            # print(f"[CommandLift]  X: {x} || v_m: {self.safety_v_m}")
        else:
        # Precision Mode
            if self.start_pos is None:
                self.start_pos = robot.lift.status['pos']
                self.target_position = self.start_pos
                # TODO: Wait for the velocity to settle to zero
            
            r = self.precision_max_vel
            xv = map_to_range(abs(x),0,r)
            if x<0:
                xv = -1*xv
            self._step_precision_move(xv, robot)
    
    
    def stop_motion(self, robot):
        """Stop the joint motion. To be used when ever the controller is idle/no-inputs
        to stop unnecessary robot motion.

        Args:
            robot (robot.Robot): Valid robot instance
        """
        robot.lift.set_velocity(0, a_m=self.params['motion']['max']['accel_m'])

    def _process_stick_to_vel(self, x):
        v_m = map_to_range(abs(x), 0, self.max_linear_vel)
        if x<0:
            v_m = -1*v_m
        return v_m
    
    def _step_precision_move(self,xv, robot):
        # Read the current joint position
        current_position = robot.lift.status['pos']

        # Calculate the time elapsed since the last iteration
        current_time = time.time()
        elapsed_time = current_time - self._prev_set_vel_ts 

        # Calculate the desired change in position to achieve the desired velocity
        desired_position_change = xv * elapsed_time
        
        # Update the target position based on the desired position change
        self.target_position = self.target_position + desired_position_change

        # Calculate the position error
        position_error = self.target_position - current_position

        # Calculate the control effort (position control)
        x_des = self.precision_kp * position_error

        # Set the control_effort as the position setpoint for the joint
        robot.lift.move_to(self.start_pos + x_des)

        # Update the previous_time for the next iteration
        self._prev_set_vel_ts = time.time()

class CommandArm:
    def __init__(self):
        """Arm motion command class.
        """
        self.params = RobotParams().get_params()[1]['arm']
        self.dead_zone = 0.0001
        self._prev_set_vel_ts = None
        self.max_linear_vel = self.params['motion']['default']['vel_m']
        self.precision_mode = False
        self.acc = self.params['motion']['max']['accel_m']
        
        # Precision mode params
        self.start_pos = None
        self.target_position = self.start_pos
        self.precision_kp = 0.6 # Very Precise: 0.6
        self.precision_max_vel = 0.04 # m/s  Very Precise: 0.02 m/s

    def command_stick_to_motion(self, x, robot):
        """Convert a stick axis value to robot arm motion.

        Args:
            x (float): Range [-1.0,+1.0], control lift speed
            robot (robot.Robot): Valid robot instance
        """
        if abs(x) < self.dead_zone:
            x = 0
        # x = to_parabola_transform(x)
        if not self.precision_mode:
        # Standard Mode
            self.start_pos = None
            v_m = self._process_stick_to_vel(x)
            robot.arm.set_velocity(v_m,a_m=self.acc)
            self._prev_set_vel_ts = time.time()
        else:
        # Precision Mode
            if self.start_pos is None:
                self.start_pos = robot.arm.status['pos']
                self.target_position = self.start_pos
                # TODO: Wait for the velocity to settle to zero
                
            r = self.precision_max_vel
            xv = map_to_range(abs(x),0,r)
            if x<0:
                xv = -1*xv
            self._step_precision_move(xv, robot)

    def stop_motion(self, robot):
        """Stop the joint motion. To be used when ever the controller is idle/no-inputs
        to stop unnecessary robot motion.

        Args:
            robot (robot.Robot): Valid robot instance
        """
        robot.arm.set_velocity(0, a_m=self.params['motion']['max']['accel_m'])

    def _process_stick_to_vel(self, x):
        v_m = map_to_range(abs(x), 0, self.max_linear_vel)
        if x<0:
            v_m = -1*v_m
        return v_m
    
    def _step_precision_move(self,xv, robot):
        # Read the current joint position
        current_position = robot.arm.status['pos']

        # Calculate the time elapsed since the last iteration
        current_time = time.time()
        elapsed_time = current_time - self._prev_set_vel_ts 

        # Calculate the desired change in position to achieve the desired velocity
        desired_position_change = xv * elapsed_time
        
        # Update the target position based on the desired position change
        self.target_position = self.target_position + desired_position_change

        # Calculate the position error
        position_error = self.target_position - current_position

        # Calculate the control effort (position control)
        x_des = self.precision_kp * position_error

        # Set the control_effort as the position setpoint for the joint
        if abs(xv)>=0:
            robot.arm.move_to(self.start_pos + x_des)

        # Update the previous_time for the next iteration
        self._prev_set_vel_ts = time.time()

class CommandDxlJoint:
    """Abstract motion command class for Dynamixel joints
    """
    def __init__(self, name, max_vel=None, acc_type=None):
        """Initiate a Dynamixe joint either a head_* or wrist_* group.

        Args:
            name (str): Name of the device name
            max_vel (float, optional): Set a custom max velocity (rad/s)
            acc_type (str, optional): Set custom acceleration profile (fast,slow,default)
        """
        self.params = RobotParams().get_params()[1][name]
        self.name = name
        self.dead_zone = 0.001
        self._prev_set_vel_ts = None
        self.max_vel = max_vel if max_vel else self.params['motion']['default']['vel']
        self.precision_mode = False
        self.acc = None
        if acc_type:
            self.acc = self.params['motion'][acc_type]['accel']
        self.precision_scale_down = 0.05

        
    def command_stick_to_motion(self, x, robot):
        """Convert a stick axis value to dynamixel servo motion.

        Args:
            x (float): Range [-1.0,+1.0], control servo speed
            robot (robot.Robot): Valid robot instance
        """
        if 'wrist' in self.name:
            motor = robot.end_of_arm.get_joint(self.name)
        if 'head' in self.name:
            motor = robot.head.get_joint(self.name)
        if 'gripper' in self.name:
            motor = robot.end_of_arm.get_joint(self.name)
        if abs(x)<self.dead_zone:
            x = 0
        acc = self.acc
        # quick hack t try to increase gripper speed
        if (x==0) or ('gripper' in self.name):
            acc = self.params['motion']['max']['accel'] #Stop with Strong Acceleration
        v = self._process_stick_to_vel(x)
        if self.precision_mode:
            v = v*self.precision_scale_down

        #print('motor =', motor)
        #print(self.name + ' : ' + 'motor.set_velocity(' + str(v) + ', ' + str(self.acc) + ')')

        motor.set_velocity(v, acc)
        self._prev_set_vel_ts = time.time()

    
    def stop_motion(self, robot):
        """Stop the joint motion. To be used when ever the controller is idle/no-inputs
        to stop unnecessary robot motion.

        Args:
            robot (robot.Robot): Valid robot instance
        """
        if 'wrist' in self.name:
            motor = robot.end_of_arm.get_joint(self.name)
        if 'head' in self.name:
            motor = robot.head.get_joint(self.name)
        if 'gripper' in self.name:
            motor = robot.end_of_arm.get_joint(self.name)

        motor.set_velocity(0,self.params['motion']['max']['accel'])
    
    def _process_stick_to_vel(self, x):
        #x = -1*x
        v = map_to_range(abs(x), 0, self.max_vel)
        if x<0:
            v = -1*v
        return v

class CommandWristYaw(CommandDxlJoint):
    """Wrist Yaw motion command class for Dynamixel joints
    """
    def __init__(self, name='wrist_yaw', max_vel=1.5, acc_type='slow'):
        super().__init__(name, max_vel, acc_type)

class CommandWristPitch(CommandDxlJoint):
    """Wrist Pitch motion command class for Dynamixel joints
    """
    def __init__(self, name='wrist_pitch', max_vel=1, acc_type='slow'):
        super().__init__(name, max_vel, acc_type)

class CommandWristRoll(CommandDxlJoint):
    """Wrist Roll motion command class for Dynamixel joints
    """
    def __init__(self, name='wrist_roll', max_vel=None, acc_type='slow'):
        super().__init__(name, max_vel, acc_type)

class CommandHeadPan(CommandDxlJoint):
    """Head Pan motion command class for Dynamixel joints
    """
    def __init__(self, name='head_pan', max_vel=None, acc_type='slow'):
        super().__init__(name, max_vel, acc_type)

class CommandHeadTilt(CommandDxlJoint):
    """Head Tilt motion command class for Dynamixel joints
    """
    def __init__(self, name='head_tilt', max_vel=None, acc_type='slow'):
        super().__init__(name, max_vel, acc_type)

class CommandGripper(CommandDxlJoint):
    """Head Tilt motion command class for Dynamixel joints
    """
    def __init__(self, name='stretch_gripper', max_vel=None, acc_type='slow'):
        super().__init__(name, max_vel, acc_type)
    
        
###################################################################


def bound_norm_vel(vel):
    return max(-1.0, min(1.0, vel))




# All velocities are normalized to the range [-1.0, 1.0]
zero_vel = {
    'base_forward' : 0.0,
    'base_counterclockwise' : 0.0,
    'lift_up': 0.0,
    'arm_out': 0.0,
    'wrist_roll_counterclockwise' : 0.0,
    'wrist_pitch_up' : 0.0,
    'wrist_yaw_counterclockwise' : 0.0,
    'head_pan_counterclockwise' : 0.0,
    'head_tilt_up' : 0.0,
    'gripper_open' : 0.0
}


class NormalizedVelocityControl():
    def __init__(self, robot):
        self.robot = robot
        if not self.robot.is_homed():
            print('WARNING from NormalizedVelocityControl: Robot reporting it is not calibrated!')
        self.precision_mode = False
        self.fast_base_mode = False
        self.base_command = CommandBase()
        self.lift_command = CommandLift()
        self.arm_command = CommandArm()
        self.wirst_yaw_command = CommandWristYaw()
        self.head_pan_command = CommandHeadPan()
        self.head_tilt_command =  CommandHeadTilt()
        self.gripper = CommandGripper()
        #self.gripper = CommandGripperPosition()
        self.wrist_pitch_command = CommandWristPitch()
        self.wrist_roll_command = CommandWristRoll()
        self.wait_between_executions = 1.0/15.0
        self.stop_loop = False
        self.lock = threading.Lock()
        self._init_command()
        self.controller_thread = None
        self._start_controller()
        
    def _init_command(self):
        with self.lock:
            self.new_command_received = False
            self.command = {'num': 0, 'time': time.time(), 'cmd': None}

    def stop(self):
        with self.lock:
            self.stop_loop = True
            self.new_command_received = False
            self.command['num'] = self.command['num'] + 1
            self.command['time'] = time.time()
            self.command['cmd'] = zero_vel.copy()
            self._execute(self.command)
            
    def set_command(self, cmd):
        with self.lock:
            self.command['num'] = self.command['num'] + 1
            self.command['time'] = time.time()
            self.command['cmd'] = cmd.copy()
            self.new_command_received = True

    def reset_base_odometry(self):
        with self.lock:
            self.robot.base.reset_odometry()
            
    def get_joint_state(self):
        with self.lock:
            
            arm_pos = self.robot.arm.status['pos']
            arm_eff = self.robot.arm.motor.status['effort_pct']

            lift_pos = self.robot.lift.status['pos']
            lift_eff = self.robot.lift.motor.status['effort_pct']

            left_wheel_pos = self.robot.base.left_wheel.status['pos']
            left_wheel_eff = self.robot.base.left_wheel.status['effort_pct']

            right_wheel_pos = self.robot.base.right_wheel.status['pos']
            right_wheel_eff = self.robot.base.right_wheel.status['effort_pct']
          
            wrist_roll_pos = self.robot.end_of_arm.motors['wrist_roll'].status['pos']
            wrist_roll_eff = self.robot.end_of_arm.motors['wrist_roll'].status['effort']

            wrist_pitch_pos = self.robot.end_of_arm.motors['wrist_pitch'].status['pos']
            wrist_pitch_eff = self.robot.end_of_arm.motors['wrist_pitch'].status['effort']

            wrist_yaw_pos = self.robot.end_of_arm.motors['wrist_yaw'].status['pos']
            wrist_yaw_eff = self.robot.end_of_arm.motors['wrist_yaw'].status['effort']

            head_pan_pos = self.robot.head.status['head_pan']['pos']
            head_pan_eff = self.robot.head.status['head_pan']['effort']

            head_tilt_pos = self.robot.head.status['head_tilt']['pos']
            head_tilt_eff = self.robot.head.status['head_tilt']['effort']

            gripper_pos = self.robot.end_of_arm.motors['stretch_gripper'].status['pos']
            gripper_pos_pct = self.robot.end_of_arm.motors['stretch_gripper'].status['pos_pct']
            gripper_eff = self.robot.end_of_arm.motors['stretch_gripper'].status['effort']
          
            base_odom_x = self.robot.base.status['x']
            base_odom_y = self.robot.base.status['y']
            base_odom_theta = self.robot.base.status['theta']
              
            state = {
                'arm_pos' : arm_pos,
                'arm_eff' : arm_eff,
                'lift_pos' : lift_pos,
                'lift_eff' : lift_eff,
                'left_wheel_pos' : left_wheel_pos,
                'left_wheel_eff' : left_wheel_eff,
                'right_wheel_pos' : right_wheel_pos,
                'right_wheel_eff' : right_wheel_eff,
                'wrist_roll_pos' : wrist_roll_pos,
                'wrist_roll_eff' : wrist_roll_eff,
                'wrist_pitch_pos' : wrist_pitch_pos,
                'wrist_pitch_eff' : wrist_pitch_eff,
                'wrist_yaw_pos' : wrist_yaw_pos,
                'wrist_yaw_eff' : wrist_yaw_eff,
                'head_pan_pos' : head_pan_pos,
                'head_pan_eff' : head_pan_eff,
                'head_tilt_pos' : head_tilt_pos,
                'head_tilt_eff' : head_tilt_eff,
                'gripper_pos' : gripper_pos,
                'gripper_pos_pct' : gripper_pos_pct,
                'gripper_eff' : gripper_eff,
                'base_odom_x' : base_odom_x,
                'base_odom_y' : base_odom_y,
                'base_odom_theta' : base_odom_theta
            }

            return(state)
            
    def controller_loop(self):
        while True: 
            with self.lock:
                if self.stop_loop:
                    exit()
                if self.new_command_received:
                    self._execute(self.command)
                    self.new_command_received = False
            time.sleep(self.wait_between_executions)

    def _start_controller(self):
        self.controller_thread = threading.Thread(target=self.controller_loop, daemon=True)
        self.controller_thread.start()
        
        
    def _update_modes(self):
        self.arm_command.precision_mode = self.precision_mode
        self.lift_command.precision_mode = self.precision_mode
        self.base_command.precision_mode = self.precision_mode
        self.base_command.fast_base_mode = self.fast_base_mode
        self.wirst_yaw_command.precision_mode = self.precision_mode
        self.gripper.precision_mode = self.precision_mode
        self.wrist_pitch_command.precision_mode = self.precision_mode
        self.wrist_roll_command.precision_mode = self.precision_mode
        self.head_pan_command.precision_mode = self.precision_mode
        self.head_tilt_command.precision_mode = self.precision_mode
            
    def _safety_stop(self):
        self.wirst_yaw_command.command_stick_to_motion(0, self.robot)
        self.arm_command.command_stick_to_motion(0, self.robot)
        self.lift_command.command_stick_to_motion(0, self.robot)
        self.head_pan_command.command_stick_to_motion(0, self.robot)
        self.head_tilt_command.command_stick_to_motion(0, self.robot)
        self.base_command.command_stick_to_motion(0,0, self.robot)
        self.wrist_pitch_command.command_stick_to_motion(0, self.robot)
        self.wrist_roll_command.command_stick_to_motion(0, self.robot)

        
    def _execute(self, norm_vel_cmd):
        #print('_execute', norm_vel_cmd)
        cmd = norm_vel_cmd['cmd']
        if cmd is not None: 

            # Note: Coninuously commanding stop_motion()(set zero
            # velocities) to chained Dxls above 15 Hz might cause thread
            # blocking issues while used in multithreaded executors
            # (E.g. ROS2).

            # Mobile Base Control
            if ('base_forward' in cmd) or ('base_counterclockwise' in cmd): 
                vf = 0.0
                vcc = 0.0
                if 'base_forward' in cmd:
                    vf = bound_norm_vel(cmd['base_forward'])
                if 'base_counterclockwise' in cmd:
                    vcc = bound_norm_vel(cmd['base_counterclockwise'])        
                self.base_command.command_stick_to_motion(vcc,  vf, self.robot)

            # Lift Control
            if 'lift_up' in cmd:
                v = bound_norm_vel(cmd['lift_up'])        
                self.lift_command.command_stick_to_motion(v,self.robot)

            # Arm Control
            if 'arm_out' in cmd: 
                v = bound_norm_vel(cmd['arm_out'])        
                self.arm_command.command_stick_to_motion(v, self.robot)


            # Dex Wrist Control
            if 'wrist_roll_counterclockwise' in cmd: 
                v = bound_norm_vel(cmd['wrist_roll_counterclockwise'])
                self.wrist_roll_command.command_stick_to_motion(v, self.robot)

            if 'wrist_pitch_up' in cmd: 
                v = bound_norm_vel(cmd['wrist_pitch_up'])
                self.wrist_pitch_command.command_stick_to_motion(v, self.robot)

            if 'wrist_yaw_counterclockwise' in cmd: 
                v = bound_norm_vel(cmd['wrist_yaw_counterclockwise'])
                self.wirst_yaw_command.command_stick_to_motion(v, self.robot)


            # Head Control
            if 'head_tilt_up' in cmd: 
                v = bound_norm_vel(cmd['head_tilt_up'])
                self.head_tilt_command.command_stick_to_motion(v, self.robot)

            if 'head_pan_counterclockwise' in cmd:
                v = bound_norm_vel(cmd['head_pan_counterclockwise'])        
                self.head_pan_command.command_stick_to_motion(v, self.robot)


            # Gripper Control
            if 'gripper_open' in cmd: 
                v = bound_norm_vel(cmd['gripper_open'])
                self.gripper.command_stick_to_motion(v, self.robot)
                #self.gripper.gripper(v, self.robot)

            self.robot.push_command()



    
if __name__ == "__main__":

    # wrist yaw has been quickly rotating in the positive direction
    # (stowing direction) upon exit until it hits the joint limit is
    # this something to do with velocity mode and or ending without
    # properly stopping the robot?

    def wait_after_cmd():
        print('sleep for ' + str(sleep_time) + ' seconds')
        time.sleep(sleep_time)    

    def test_cmd(k, v, controller):
        cmd = {k : v}
        print(cmd)
        controller.set_command(cmd)
        wait_after_cmd()


    vel_keys = list(zero_vel.keys())

    dxl_vel_keys = ['wrist_roll_counterclockwise', 
                    'wrist_pitch_up',
                    'wrist_yaw_counterclockwise',
                    'head_pan_counterclockwise',
                    'head_tilt_up']

    print('velocity keys = ' + str(vel_keys))

    vel_mag = 0.2

    pos = vel_mag
    zero = 0
    neg = -1.0 * vel_mag

    sleep_time = 0.25
    
    try: 
        robot = rb.Robot()
        robot.startup()
        controller = NormalizedVelocityControl(robot)

        controller.set_command(zero_vel)
        joint_state = controller.get_joint_state()
        print(joint_state)
        
        time.sleep(0.2)
        
        for k in vel_keys:
            test_cmd(k, pos, controller)
            test_cmd(k, neg, controller)
            test_cmd(k, zero, controller)

        controller.set_command(zero_vel)

        controller.stop()
                    
        exit()

    except (ThreadServiceExit, KeyboardInterrupt, SystemExit):
        robot.stop()
