"""
This file contains all constants necessary for running motion planning in baxter domain. Every class that needs to use these constants should refference to this file.
"""

# Baxter dimension constant
BASE_DIM = 1
JOINT_DIM = 16
ROBOT_ATTR_DIM = 17
TWOARMDIM = 16

# Baxter Movement Constraints
BASE_MOVE = 1
JOINT_MOVE_FACTOR = 20

# EEReachable Constants
APPROACH_DIST = 0.025
RETREAT_DIST = 0.025
EEREACHABLE_STEPS = 3

# Collision Constants
DIST_SAFE = 5e-3
RCOLLIDES_DSAFE = 5e-3
COLLIDES_DSAFE = 1e-3

# Plan Coefficient
IN_GRIPPER_COEFF = 1.
EEREACHABLE_COEFF = 1e2
EEREACHABLE_OPT_COEFF = 1.3e3
EEREACHABLE_ROT_OPT_COEFF = 3e2
INGRIPPER_OPT_COEFF = 3e2
RCOLLIDES_OPT_COEFF = 1e2
OBSTRUCTS_COEEF = 1
OBSTRUCTS_OPT_COEFF = 1e2
GRASP_VALID_COEFF = 1e1

# Gripper Value
GRIPPER_OPEN_VALUE = 0.02
GRIPPER_CLOSE_VALUE = 0.015

# Tolerance
TOL = 1e-4

# Predicate Gradient Test Option
TEST_GRAD = True
