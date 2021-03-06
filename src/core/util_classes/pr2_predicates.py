from core.util_classes import robot_predicates
from sco.expr import Expr, AffExpr, EqExpr, LEqExpr
from core.util_classes.sampling import ee_reachable_resample, resample_bp_around_target
import core.util_classes.pr2_constants as const
from collections import OrderedDict
from openravepy import DOFAffine
import numpy as np

"""
This file Defines specific PR2 related predicates
"""

# Attributes used in pr2 domain. (Tuple to avoid changes to the attr_inds)
ATTRMAP = {"Robot": (("backHeight", np.array([0], dtype=np.int)),
                         ("lArmPose", np.array(range(7), dtype=np.int)),
                         ("lGripper", np.array([0], dtype=np.int)),
                         ("rArmPose", np.array(range(7), dtype=np.int)),
                         ("rGripper", np.array([0], dtype=np.int)),
                         ("pose", np.array([0,1,2], dtype=np.int))),
           "RobotPose": (("backHeight", np.array([0], dtype=np.int)),
                          ("lArmPose", np.array(range(7), dtype=np.int)),
                          ("lGripper", np.array([0], dtype=np.int)),
                          ("rArmPose", np.array(range(7), dtype=np.int)),
                          ("rGripper", np.array([0], dtype=np.int)),
                          ("value", np.array([0,1,2], dtype=np.int))),
           "Can": (("pose", np.array([0,1,2], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int))),
           "EEPose": (("value", np.array([0,1,2], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int))),
           "Target": (("value", np.array([0,1,2], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int))),
           "Table": (("pose", np.array([0,1,2], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int))),
           "Obstacle": (("pose", np.array([0,1,2], dtype=np.int)),
                    ("rotation", np.array([0,1,2], dtype=np.int)))
              }

class PR2At(robot_predicates.At):
    pass

class PR2RobotAt(robot_predicates.RobotAt):

    # RobotAt, Robot, RobotPose

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_dim = 20
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(PR2RobotAt, self).__init__(name, params, expected_param_types, env)

class PR2IsMP(robot_predicates.IsMP):

    # IsMP Robot (Just the Robot Base)

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type]))])
        super(PR2IsMP, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((const.JOINT_DIM,1))
        active_lb = lb_limit[dof_inds].reshape((const.JOINT_DIM,1))
        joint_move = (active_ub-active_lb)/const.JOINT_MOVE_FACTOR
        # Setup the Equation so that: Ax+b < val represents
        # |base_pose_next - base_pose| <= const.BASE_MOVE
        # |joint_next - joint| <= joint_movement_range/const.JOINT_MOVE_FACTOR
        val = np.vstack((joint_move, const.BASE_MOVE*np.ones((const.BASE_DIM, 1)), joint_move, const.BASE_MOVE*np.ones((const.BASE_DIM, 1))))
        A = np.eye(2*const.ROBOT_ATTR_DIM) - np.eye(2*const.ROBOT_ATTR_DIM, k=const.ROBOT_ATTR_DIM) - np.eye(2*const.ROBOT_ATTR_DIM, k=-const.ROBOT_ATTR_DIM)
        b = np.zeros((2*const.ROBOT_ATTR_DIM,1))
        robot_body._set_active_dof_inds(range(39))

        # Setting attributes for testing
        self.base_step = const.BASE_MOVE*np.ones((const.BASE_DIM, 1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class PR2WithinJointLimit(robot_predicates.WithinJointLimit):

    # WithinJointLimit Robot

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][:-1]))])
        super(PR2WithinJointLimit, self).__init__(name, params, expected_param_types, env, debug)

    def setup_mov_limit_check(self):
        # Get upper joint limit and lower joint limit
        robot_body = self._param_to_body[self.robot]
        robot = robot_body.env_body
        robot_body._set_active_dof_inds()
        dof_inds = robot.GetActiveDOFIndices()
        lb_limit, ub_limit = robot.GetDOFLimits()
        active_ub = ub_limit[dof_inds].reshape((const.JOINT_DIM,1))
        active_lb = lb_limit[dof_inds].reshape((const.JOINT_DIM,1))
        # Setup the Equation so that: Ax+b < val represents
        # lb_limit <= pose <= ub_limit
        val = np.vstack((-active_lb, active_ub))
        A_lb_limit = -np.eye(const.JOINT_DIM)
        A_up_limit = np.eye(const.JOINT_DIM)
        A = np.vstack((A_lb_limit, A_up_limit))
        b = np.zeros((2*const.JOINT_DIM,1))
        robot_body._set_active_dof_inds(range(39))

        joint_move = (active_ub-active_lb)/const.JOINT_MOVE_FACTOR
        self.base_step = const.BASE_MOVE*np.ones((3,1))
        self.joint_step = joint_move
        self.lower_limit = active_lb
        return A, b, val

class PR2Stationary(robot_predicates.Stationary):
    pass

class PR2StationaryBase(robot_predicates.StationaryBase):

    # StationaryBase, Robot (Only Robot Base)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][-1]])])
        self.attr_dim = const.BASE_DIM
        super(PR2StationaryBase, self).__init__(name, params, expected_param_types, env)

class PR2StationaryArms(robot_predicates.StationaryArms):

    # StationaryArms, Robot (Only Robot Arms)

    def __init__(self, name, params, expected_param_types, env=None):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type][1:-1]))])
        self.attr_dim = const.TWOARMDIM
        super(PR2StationaryArms, self).__init__(name, params, expected_param_types, env)

class PR2StationaryW(robot_predicates.StationaryW):
    pass

class PR2StationaryNEq(robot_predicates.StationaryNEq):
    pass

class PR2GraspValid(robot_predicates.GraspValid):
    pass

class PR2GraspValidPos(PR2GraspValid):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][0]]),(params[1], [ATTRMAP[params[1]._type][0]])])
        self.attr_dim = 3
        super(PR2GraspValidPos, self).__init__(name, params, expected_param_types, env, debug)

class PR2GraspValidRot(PR2GraspValid):

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][1]]),(params[1], [ATTRMAP[params[1]._type][1]])])
        self.attr_dim = 3
        super(PR2GraspValidRot, self).__init__(name, params, expected_param_types, env, debug)

class PR2InContact(robot_predicates.InContact):

    # InContact robot EEPose target

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        # Define constants
        self.GRIPPER_CLOSE = const.GRIPPER_CLOSE_VALUE
        self.GRIPPER_OPEN = const.GRIPPER_OPEN_VALUE
        self.attr_inds = OrderedDict([(params[0], [ATTRMAP[params[0]._type][4]])])
        super(PR2InContact, self).__init__(name, params, expected_param_types, env, debug)

class PR2InGripper(robot_predicates.InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(PR2InGripper, self).__init__(name, params, expected_param_types, env, debug)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {"backHeight": back_height,
                         "lArmPose": l_arm_pose,
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose,
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot_body.env_body.GetManipulator('rightarm').GetArmIndices()
        return robot_trans, arm_inds

class PR2InGripperPos(PR2InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.pos_check(x)[0]
        self.eval_grad = lambda x: self.pos_check(x)[1]
        super(PR2InGripperPos, self).__init__(name, params, expected_param_types, env, debug)

    # "Robot": (("backHeight", np.array([0], dtype=np.int)),
    #                          ("lArmPose", np.array(range(7), dtype=np.int)),
    #                          ("lGripper", np.array([0], dtype=np.int)),
    #                          ("rArmPose", np.array(range(7), dtype=np.int)),
    #                          ("rGripper", np.array([0], dtype=np.int)),
    #                          ("pose", np.array([0,1,2], dtype=np.int)))
    #
    #            "Can": (("pose", np.array([0,1,2], dtype=np.int)),
    #                     ("rotation", np.array([0,1,2], dtype=np.int)))

    def pos_error(self, obj_trans, robot_trans, axises, arm_joints):
        """
            This function calculates the value and the jacobian of the displacement between center of gripper and center of object

            obj_trans: object's rave_body transformation
            robot_trans: robot gripper's rave_body transformation
            axises: rotational axises of the object
            arm_joints: list of robot joints
        """
        gp = np.array([0,0,0])
        robot_pos = robot_trans[:3, 3]
        obj_pos = obj_trans[:3, 3]
        dist_val = (robot_pos.flatten() - obj_pos.flatten()).reshape((3,1))
        # Calculate the joint jacobian
        arm_jac = np.array([np.cross(joint.GetAxis(), robot_pos.flatten() - joint.GetAnchor()) for joint in arm_joints]).T.copy()
        # Calculate jacobian for the robot base
        base_jac = np.eye(3)
        base_jac[:,2] = np.cross(np.array([0, 0, 1]), robot_pos - self.x[17:20])
        # Calculate jacobian for the back hight
        torso_jac = np.array([[0],[0],[1]])
        # Calculate object jacobian
        obj_jac = -1*np.array([np.cross(axis, obj_pos - gp - obj_trans[:3,3].flatten()) for axis in axises]).T
        obj_jac = np.c_[-np.eye(3), obj_jac]
        # Create final 3x26 jacobian matrix -> (Gradient checked to be correct)
        dist_jac = np.hstack((torso_jac, np.zeros((3, 8)), arm_jac, np.zeros((3, 1)), base_jac, obj_jac))

        return (dist_val, dist_jac)

class PR2InGripperRot(PR2InGripper):

    # InGripper, Robot, Can

    def __init__(self, name, params, expected_param_types, env = None, debug = False):
        # Sets up constants
        self.coeff = const.IN_GRIPPER_COEFF
        self.opt_coeff = const.INGRIPPER_OPT_COEFF
        self.eval_f = lambda x: self.rot_check(x)[0]
        self.eval_grad = lambda x: self.rot_check(x)[1]
        super(PR2InGripperRot, self).__init__(name, params, expected_param_types, env, debug)

class PR2EEReachable(robot_predicates.EEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[2], list(ATTRMAP[params[2]._type]))])
        self.attr_dim = 26
        super(PR2EEReachable, self).__init__(name, params, expected_param_types, env, debug, steps)

    def resample(self, negated, t, plan):
        return ee_reachable_resample(self, negated, t, plan)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {"backHeight": back_height,
                         "lArmPose": l_arm_pose,
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose,
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def get_robot_info(self, robot_body):
        # Provide functionality of Obtaining Robot information
        tool_link = robot_body.env_body.GetLink("r_gripper_tool_frame")
        robot_trans = tool_link.GetTransform()
        arm_inds = robot_body.env_body.GetManipulator('rightarm').GetArmIndices()
        return robot_trans, arm_inds

    def get_rel_pt(self, rel_step):
        if rel_step <= 0:
            return rel_step*np.array([const.APPROACH_DIST, 0, 0])
        else:
            return rel_step*np.array([0, 0, const.RETREAT_DIST])

    def stacked_f(self, x):
        i = 0
        f_res = []
        start, end = self.active_range
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            f_res.append(self.ee_pose_check_rel_obj(x[i:i+self.attr_dim], rel_pt)[0])
            i += self.attr_dim
        return np.vstack(tuple(f_res))

    def stacked_grad(self, x):
        f_grad = []
        start, end = self.active_range
        t = (2*self._steps+1)
        k = 3

        grad = np.zeros((k*t, self.attr_dim*t))
        i = 0
        j = 0
        for s in range(start, end+1):
            rel_pt = self.get_rel_pt(s)
            grad[j:j+k, i:i+self.attr_dim] = self.ee_pose_check_rel_obj(x[i:i+self.attr_dim], rel_pt)[1]
            i += self.attr_dim
            j += k
        return grad

class PR2EEReachablePos(PR2EEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=const.EEREACHABLE_STEPS):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_OPT_COEFF
        self.eval_f = self.stacked_f
        self.eval_grad = self.stacked_grad
        self.attr_dim = 26
        super(PR2EEReachablePos, self).__init__(name, params, expected_param_types, env, debug, steps)

class PR2EEReachableRot(PR2EEReachable):

    # EEUnreachable Robot, StartPose, EEPose

    def __init__(self, name, params, expected_param_types, env=None, debug=False, steps=0):
        self.coeff = const.EEREACHABLE_COEFF
        self.opt_coeff = const.EEREACHABLE_ROT_OPT_COEFF
        self.eval_f = lambda x: self.ee_rot_check(x)[0]
        self.eval_grad = lambda x: self.ee_rot_check(x)[1]
        super(PR2EEReachableRot, self).__init__(name, params, expected_param_types, env, debug, steps)

class PR2Obstructs(robot_predicates.Obstructs):

    # Obstructs, Robot, RobotPose, RobotPose, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False, tol=const.COLLISION_TOL):
        self.attr_dim = 20
        self.dof_cache = None
        self.coeff = -1
        self.neg_coeff = 1
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[3], list(ATTRMAP[params[3]._type]))])
        super(PR2Obstructs, self).__init__(name, params, expected_param_types, env, debug, tol)

    def resample(self, negated, t, plan):
        target_pose = self.can.pose[:, t]
        return resample_bp_around_target(self, t, plan, target_pose, dist=const.OBJ_RING_SAMPLING_RADIUS)

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {"backHeight": back_height,
                         "lArmPose": l_arm_pose,
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose,
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            dof_inds = np.ndarray(0, dtype=np.int)
            dof_inds = np.r_[dof_inds, robot.GetJoint("torso_lift_joint").GetDOFIndex()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetGripperIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetGripperIndices()]
            robot.SetActiveDOFs(
                    dof_inds,
                    DOFAffine.X + DOFAffine.Y + DOFAffine.RotationAxis,
                    [0, 0, 1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

class PR2ObstructsHolding(robot_predicates.ObstructsHolding):

    # ObstructsHolding, Robot, RobotPose, RobotPose, Can, Can

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 20
        self.dof_cache = None
        self.coeff = -1
        self.neg_coeff = 1
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[3], list(ATTRMAP[params[3]._type])),
                                 (params[4], list(ATTRMAP[params[4]._type]))])
        self.OBSTRUCTS_OPT_COEFF = const.OBSTRUCTS_OPT_COEFF
        super(PR2ObstructsHolding, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.DIST_SAFE

    def resample(self, negated, t, plan):
        target_pose = self.obstruct.pose[:, t]
        return resample_bp_around_target(self, t, plan, target_pose, dist=const.OBJ_RING_SAMPLING_RADIUS)

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            dof_inds = np.ndarray(0, dtype=np.int)
            dof_inds = np.r_[dof_inds, robot.GetJoint("torso_lift_joint").GetDOFIndex()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetGripperIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetGripperIndices()]
            # dof_inds = [12]+ list(range(15, 22)) + [22]+ list(range(27, 34)) + [34]
            robot.SetActiveDOFs(
                    dof_inds,
                    DOFAffine.X + DOFAffine.Y + DOFAffine.RotationAxis,
                    [0, 0, 1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {"backHeight": back_height,
                         "lArmPose": l_arm_pose,
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose,
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)

class PR2Collides(robot_predicates.Collides):
    pass

class PR2RCollides(robot_predicates.RCollides):

    # RCollides Robot Obstacle

    def __init__(self, name, params, expected_param_types, env=None, debug=False):
        self.attr_dim = 20
        self.dof_cache = None
        self.coeff = -1
        self.neg_coeff = 1
        self.opt_coeff = const.RCOLLIDES_OPT_COEFF
        self.attr_inds = OrderedDict([(params[0], list(ATTRMAP[params[0]._type])),
                                 (params[1], list(ATTRMAP[params[1]._type]))])
        super(PR2RCollides, self).__init__(name, params, expected_param_types, env, debug)
        self.dsafe = const.RCOLLIDES_DSAFE

    def resample(self, negated, t, plan):
        target_pose = self.obstacle.pose[:, t]
        return resample_bp_around_target(self, t, plan, target_pose,
                                        dist=const.TABLE_SAMPLING_RADIUS)

    def set_active_dof_inds(self, robot_body, reset = False):
        robot = robot_body.env_body
        if reset == True and self.dof_cache != None:
            robot.SetActiveDOFs(self.dof_cache)
            self.dof_cache = None
        elif reset == False and self.dof_cache == None:
            self.dof_cache = robot.GetActiveDOFIndices()
            dof_inds = np.ndarray(0, dtype=np.int)
            dof_inds = np.r_[dof_inds, robot.GetJoint("torso_lift_joint").GetDOFIndex()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("leftarm").GetGripperIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetArmIndices()]
            dof_inds = np.r_[dof_inds, robot.GetManipulator("rightarm").GetGripperIndices()]
            robot.SetActiveDOFs(
                    dof_inds,
                    DOFAffine.X + DOFAffine.Y + DOFAffine.RotationAxis,
                    [0, 0, 1])
        else:
            raise PredicateException("Incorrect Active DOF Setting")

    def set_robot_poses(self, x, robot_body):
        # Provide functionality of setting robot poses
        back_height = x[0]
        l_arm_pose, l_gripper = x[1:8], x[8]
        r_arm_pose, r_gripper = x[9:16], x[16]
        base_pose = x[17:20]
        robot_body.set_pose(base_pose)
        dof_value_map = {"backHeight": back_height,
                         "lArmPose": l_arm_pose,
                         "lGripper": l_gripper,
                         "rArmPose": r_arm_pose,
                         "rGripper": r_gripper}
        robot_body.set_dof(dof_value_map)
