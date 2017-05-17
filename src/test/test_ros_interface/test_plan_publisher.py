from collections import OrderedDict
import json
import numpy as np
import unittest

from openravepy import Environment

import rospy
from std_msgs.msg import String

from sco import expr

from core.internal_repr.action import Action
from core.internal_repr.parameter import Object, Symbol
from core.internal_repr.plan import Plan
from core.internal_repr.predicate import Predicate
from core.util_classes.baxter_predicates import *
from core.util_classes.box import Box
from core.util_classes.can import Can, BlueCan, RedCan, GreenCan
from core.util_classes.circle import Circle, BlueCircle, GreenCircle, RedCircle
from core.util_classes.common_predicates import *
from core.util_classes.matrix import *
from core.util_classes.param_setup import ParamSetup
from core.util_classes.robots import Baxter
from core.util_classes.robot_predicates import *
from core.util_classes.table import Table
from core.util_classes.wall import Wall
from ros_interface.plan_publisher import PlanPublisher

from baxter_plan.msg import ActionMSG
from baxter_plan.msg import ActionPredMSG
from baxter_plan.msg import FloatArrayMSG
from baxter_plan.msg import GeomMSG
from baxter_plan.msg import ParameterMSG
from baxter_plan.msg import PlanMSG
from baxter_plan.msg import PredicateMSG

e1 = expr.Expr(lambda x: np.array([x]))
e2 = expr.Expr(lambda x: np.power(x, 2))

class TestPlanPublisher(unittest.TestCase):
	def setup(self):
		self.plan_publisher = PlanPublisher()


	def test_create_geom_msg(self):
		self.setup()
		geom = Baxter()
		geom_msg = self.plan_publisher.create_geom_msg(geom)
		self.assertEqual(geom_msg.type_name, 'Baxter')
		
		geom = GreenCan(3.14, 1.23)
		geom_msg = self.plan_publisher.create_geom_msg(geom)
		self.assertEqual(geom_msg.type_name, 'GreenCan')
		attr_str = json.dumps(geom.__dict__)
		self.assertIn('"radius": 3.14', geom_msg.attrs)


	def test_create_floatarray_msg(self):
		self.setup()
		arr = np.array([1.0, 2.3, 4.56, 7.890])
		arr_msg = self.plan_publisher.create_floatarray_msg(arr)
		self.assertListEqual(arr.tolist(), arr_msg.data)


	def test_create_parameter_msg(self):
		self.setup()
		param = ParamSetup.setup_green_can()
		param_msg = self.plan_publisher.create_parameter_msg(param)

		self.assertEqual(param_msg.name, param.name)
		self.assertEqual(param_msg.type_name, param.get_type())
		self.assertFalse(param_msg.is_symbol)
		self.assertIn('pose', param_msg.undefined_attrs)
		self.assertEqual(param_msg.geom.type_name, 'GreenCan')
		self.assertIn('"radius": 0.02', param_msg.geom.attrs)


	def test_create_predicate_msg(self):
		self.setup()

		# Baxter predicates
		robot = ParamSetup.setup_baxter()
		can = ParamSetup.setup_blue_can()
		test_env = ParamSetup.setup_env()
		pred = BaxterInGripperPos("InGripper", [robot, can], ["Robot", "Can"], test_env)
		pred2 = BaxterInGripperRot("InGripper_rot", [robot, can], ["Robot", "Can"], test_env)
		pred_msg = self.plan_publisher.create_predicate_msg(pred)
		self.assertEqual(eval(pred_msg.type_name), BaxterInGripperPos)
		self.assertEqual(pred_msg.name, "InGripper")
		self.assertEqual(pred_msg.params[0].name, "baxter")
		self.assertEqual(pred_msg.params[0].type_name, 'Robot')
		self.assertEqual(pred_msg.params[0].lArmPose[6].data, [0])

	def test_create_plan(self):
		self.setup()
		robot = ParamSetup.setup_baxter()
		rPose = ParamSetup.setup_baxter_pose()
		table = ParamSetup.setup_table()
		can = ParamSetup.setup_green_can()
		param_map = {"baxter": robot, "baxter_pose": rPose, "table": table, "can": can}
		robot.pose = np.array([[3, 4, 5, 3]])
		rPose.value = np.array([[3, 4, 2, 6]])
		robot.rGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
		robot.lGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
		rPose.rGripper = np.matrix([0.02, 0.4, 0.6, 0.02])
		robot.rArmPose = np.array([[0,0,0,0,0,0,0],
								   [1,2,3,4,5,6,7],
								   [7,6,5,4,3,2,1],
								   [0,0,0,0,0,0,0]]).T
		robot.lArmPose = np.array([[0,0,0,0,0,0,0],
		                           [1,2,3,4,5,6,7],
		                           [7,6,5,4,3,2,1],
		                           [0,0,0,0,0,0,0]]).T
		rPose.rArmPose = np.array([[0,0,0,0,0,0,0]]).T
		rPose.lArmPose = np.array([[0,0,0,0,0,0,0]]).T
		pred = BaxterRobotAt("testRobotAt", [robot, rPose], ["Robot", "RobotPose"])
		pred_list = [{'pred': pred, 'negated': False, 'hl_info': 'pre', 'active_timesteps': (0, 4)}]
		act = Action(2, 'test_action', (0,4), param_map.values(), pred_list)
		env = Environment()
		plan = Plan(param_map, [act], 5, env)
		plan_msg = self.plan_publisher.create_plan_msg(plan)
		self.assertEqual(plan_msg.actions[0].predicates[0].pred.type_name, "BaxterRobotAt")
		self.assertTrue(isinstance(plan_msg.actions[0], ActionMSG))
		self.assertTrue(isinstance(plan_msg.parameters[0], ParameterMSG))
		self.assertListEqual(plan_msg.actions[0].predicates[0].pred.params[0].pose[0].data, [3, 4, 5, 3])
		self.assertListEqual(plan_msg.parameters[0].lArmPose[0].data, [0, 1, 7, 0])

