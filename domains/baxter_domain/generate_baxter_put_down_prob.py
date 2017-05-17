from IPython import embed as shell
import itertools
import numpy as np
import random

SEED = 1234
NUM_PROBS = 5
NUM_CANS = 1 # each can i starts at target i, so we must have NUM_CANS <= NUM_TARGETS
NUM_TARGETS = 1
filename = "baxter_probs/putdown"
assert NUM_CANS <= NUM_TARGETS
GOAL = "(BaxterRobotAt baxter robot_end_pose), (BaxterAt can1 target2)"

DIST_SAFE = 5e-3

CAN_ROTATION_INIT = [0,0,0]
CAN_RADIUS = 0.02
CAN_HEIGHT = 0.25
CAN_GEOM = [CAN_RADIUS, CAN_HEIGHT]
DIST_BETWEEN_CANS = 0.01
# init and end robot pose(only the base)
Baxter_INIT_POSE = [0]
Baxter_END_POSE = [0]
L_ARM_INIT = [0, 0, 0, 0, 0, 0, 0]
# R_ARM_INIT = [-np.pi/6, np.pi/4,np.pi/4,np.pi/10,np.pi/4,np.pi/4,0]
R_ARM_INIT = [0,0,0,0,0,0,0]
Baxter_END_LARM = [0,0,0,0,0,0,0]
Baxter_END_RARM = [np.pi/10,0,0,0,0,0,0]
INT_GRIPPER = [0.015]
END_GRIPPER = [0.02]

ROBOT_DIST_FROM_TABLE = 0.05
# rll table
# TABLE_DIM = [2.235, 0.94]
# TABLE_THICKNESS = [0.2]
# TABLE_LEG_DIM = [1.3, 0.6]
# TABLE_LEG_HEIGHT = [0.6]
# TABLE_BACK = [False]

# small table
TABLE_DIM = [0.65, 1.5]
TABLE_THICKNESS = 0.2
TABLE_LEG_DIM = [.15, 0.2]
TABLE_LEG_HEIGHT = 0.6
TABLE_BACK = False
# TABLE_GEOM = []
TABLE_GEOM = [.325, .75, 0.1]
# for info in [TABLE_DIM, [TABLE_THICKNESS], TABLE_LEG_DIM, [TABLE_LEG_HEIGHT], [TABLE_BACK]]:
#     TABLE_GEOM.extend(info)

class CollisionFreeTargetValueGenerator(object):
    def __init__(self):
        self.max_x = 0.75+TABLE_DIM[0]/2 - CAN_RADIUS
        self.min_x = 1.5-self.max_x
        self.max_y = 0.02+TABLE_DIM[1]/2 - CAN_RADIUS
        self.min_y = 0.04-self.max_y
        self._poses = []

    def __iter__(self):
        return self

    def next(self):
        collides = True
        while collides:
            collides = False
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            z = TABLE_LEG_HEIGHT + TABLE_THICKNESS + CAN_HEIGHT/2
            for pose in self._poses:
                diff = np.array([x - pose[0], y-pose[1]])
                if np.linalg.norm(diff) <= CAN_RADIUS*2 + DIST_BETWEEN_CANS:
                    collides = True
        pose = [x,y,z]
        self._poses.append(pose)
        return pose

    def reset(self):
        self._poses = []

def get_baxter_init_attrs_str(name, LArm = L_ARM_INIT, RArm = R_ARM_INIT, G = INT_GRIPPER):
    s = ""
    s += "(lArmPose {} {}), ".format(name, LArm)
    s += "(lGripper {} {}), ".format(name, G)
    s += "(rArmPose {} {}), ".format(name, RArm)
    s += "(rGripper {} {}), ".format(name, G)
    return s

def get_baxter_undefined_attrs_str(name):
    s = ""
    s += "(lArmPose {} undefined), ".format(name)
    s += "(lGripper {} undefined), ".format(name)
    s += "(rArmPose {} undefined), ".format(name)
    s += "(rGripper {} undefined), ".format(name)
    return s

def main():
    random.seed(SEED)
    target_gen = CollisionFreeTargetValueGenerator()
    for iteration in range(NUM_PROBS):
        target_gen.reset()
        s = "# AUTOGENERATED. DO NOT EDIT.\n# Configuration file for CAN problem instance. Blank lines and lines beginning with # are filtered out.\n\n"

        s += "# The values after each attribute name are the values that get passed into the __init__ method for that attribute's class defined in the domain configuration.\n"
        s += "Objects: "
        for i in range(NUM_TARGETS):
            s += "Target (name target{}); ".format(i)
            s += "EEPose (name ee_target{}); ".format(i)
            s += "RobotPose (name pdp_target{}); ".format(i)
            if i < NUM_CANS:
                s += "Can (name can{}); ".format(i)
                # s += "RobotPose (name gp_can{}); ".format(i)
        s += "Robot (name {}); ".format("baxter")
        s += "RobotPose (name {}); ".format("robot_init_pose")
        s += "RobotPose (name {}); ".format("robot_end_pose")
        s += "Obstacle (name {}) \n\n".format("table")

        s += "Init: "
        for i in range(NUM_TARGETS):
            target_pos = target_gen.next()
            s += "(geom target{} {} {}), ".format(i, CAN_GEOM[0], CAN_GEOM[1])
            s += "(value target{} {}), ".format(i, target_pos)
            s += "(rotation target{} {}),".format(i, CAN_ROTATION_INIT)
            s += "(value pdp_target{} undefined)".format(i)
            s += get_baxter_undefined_attrs_str("pdp_target{}".format(i))
            s += "(value ee_target{} undefined), ".format(i)
            s += "(rotation ee_target{} undefined), ".format(i)

            if i < NUM_CANS:
                s += "(geom can{} {} {}), ".format(i, CAN_GEOM[0], CAN_GEOM[1])
                s += "(pose can{} {}), ".format(i, target_pos)
                s += "(rotation can{} {}),".format(i, CAN_ROTATION_INIT)
                # s += "(value gp_can{} undefined), ".format(i)
        s += "(geom {}), ".format("baxter")
        # setting intial state of robot
        s += "(pose baxter {}), ".format(Baxter_INIT_POSE)
        s += get_baxter_init_attrs_str('baxter')

        s += "(value {} {}), ".format("robot_init_pose", Baxter_INIT_POSE)
        s += get_baxter_init_attrs_str('robot_init_pose')
        s += "(value {} {}), ".format("robot_end_pose", Baxter_END_POSE)
        s += get_baxter_init_attrs_str('robot_end_pose', LArm = Baxter_END_LARM, RArm=Baxter_END_RARM, G=END_GRIPPER)

        # table pose
        z = TABLE_THICKNESS/2 + TABLE_LEG_HEIGHT - DIST_SAFE
        s += "(pose {} [0.75, 0.02, {}]), ".format("table", z)
        s += "(rotation {} {}), ".format("table", CAN_ROTATION_INIT)
        s += "(geom {} {}); ".format("table", TABLE_GEOM)

        for i in range(NUM_CANS):
            s += "(BaxterAt can{} target{}), ".format(i, i)
            s += "(BaxterStationary can{}), ".format(i)
            for j in range(NUM_CANS):
                s += "(BaxterStationaryNEq can{} can{}), ".format(i, j)
            # s += "(InContact baxter gp_can{} target{}), ".format(i, i)
            # s += "(GraspValid gp_can{} target{} grasp0), ".format(i, i)
        for i in range(NUM_TARGETS):
            s += "(BaxterInContact baxter ee_target{} target{}), ".format(i, i)
            s += "(BaxterGraspValidPos ee_target{} target{}), ".format(i, i)
            s += "(BaxterGraspValidRot ee_target{} target{}), ".format(i, i)
            s += "(BaxterEEReachablePos baxter pdp_target{} ee_target{}), ".format(i, i)
            s += "(BaxterEEReachableRot baxter pdp_target{} ee_target{}), ".format(i, i)
        s += "(BaxterRobotAt baxter robot_init_pose), "
        # s += "(BaxterStationaryArms baxter), "
        s += "(BaxterStationaryBase baxter), "
        s += "(BaxterIsMP baxter), "
        s += "(BaxterWithinJointLimit baxter), "
        s += "(BaxterStationaryW table) \n\n"

        s += "Goal: {}".format(GOAL)

        with open(filename+"_{}_{}.prob".format(SEED, iteration), "w") as f:
            f.write(s)

if __name__ == "__main__":
    main()
