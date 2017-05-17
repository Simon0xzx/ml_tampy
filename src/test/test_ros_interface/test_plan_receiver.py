from collections import OrderedDict
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
from ros_interface.plan_receiver import PlanReceiver

from baxter_plan.msg import ActionMSG
from baxter_plan.msg import ActionPredMSG
from baxter_plan.msg import FloatArrayMSG
from baxter_plan.msg import GeomMSG
from baxter_plan.msg import ParameterMSG
from baxter_plan.msg import PlanMSG
from baxter_plan.msg import PredicateMSG

e1 = expr.Expr(lambda x: np.array([x]))
e2 = expr.Expr(lambda x: np.power(x, 2))

class TestPlanReceiver(unittest.TestCase):
	def setup(self):
		self.env = Environment()
		self.plan_publisher = PlanPublisher()
		self.plan_receiver = PlanReceiver()


	def test_build_geom(self):
		self.setup()
		geom = Baxter()
		geom_msg = self.plan_publisher.create_geom_msg(geom)

		new_geom = self.plan_receiver._build_geom(geom_msg)
		self.assertEquals(type(geom), type(new_geom))
		
		geom = GreenCan(3.14, 1.23)
		geom_msg = self.plan_publisher.create_geom_msg(geom)

		new_geom = self.plan_receiver._build_geom(geom_msg)
		self.assertEquals(geom.radius, new_geom.radius)
		self.assertEquals(geom.height, new_geom.height)
		self.assertEquals(type(geom), type(new_geom))


	def test_build_2D_numpy_array(self):
		self.setup()
		arr = np.array([1.0, 2.3, 4.56, 7.890])
		arr_msg = self.plan_publisher.create_floatarray_msg(arr)
		arr_msgs = [arr_msg]
		new_arr = self.plan_receiver._build_2D_numpy_array(arr_msgs)
		self.assertTrue(np.array_equal(arr, new_arr[0]))


	def test_build_parameter(self):
		self.setup()
		param = ParamSetup.setup_green_can()
		param_msg = self.plan_publisher.create_parameter_msg(param)
		new_param = self.plan_receiver._build_param(param_msg)
		self.assertEqual(new_param.geom.radius, param.geom.radius)
		self.assertEqual(new_param.geom.height, param.geom.height)
		self.assertEqual(type(param), type(new_param))
		self.assertEqual(param.name, new_param.name)


	def test_build_predicate(self):
		self.setup()

		# Baxter predicates
		robot = ParamSetup.setup_baxter()
		can = ParamSetup.setup_blue_can()
		test_env = ParamSetup.setup_env()
		pred = BaxterInGripperPos("InGripper", [robot, can], ["Robot", "Can"], test_env)
		pred_msg = self.plan_publisher.create_predicate_msg(pred)
		self.assertEqual(eval(pred_msg.type_name), BaxterInGripperPos)
		self.assertEqual(pred_msg.name, "InGripper")
		self.assertEqual(pred_msg.params[0].name, "baxter")
		self.assertEqual(pred_msg.params[0].type_name, 'Robot')
		self.assertEqual(pred_msg.params[0].lArmPose[6].data, [0])

		new_pred = self.plan_receiver._build_pred(pred_msg, {}, self.env)
		self.assertEqual(pred.name, new_pred.name)
		self.assertEqual(pred.active_range, new_pred.active_range)
		self.assertListEqual(pred.params[0].pose.tolist(), new_pred.params[0].pose.tolist())

	def test_build_plan(self):
		self.setup()
		robot = ParamSetup.setup_baxter()
		rPose = ParamSetup.setup_baxter_pose()
		target = ParamSetup.setup_target()
		can = ParamSetup.setup_green_can()
		param_map = {"baxter": robot, "baxter_pose": rPose, "target": target, "can": can}
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
		plan = Plan(param_map, [act], 5, self.env)
		plan_msg = self.plan_publisher.create_plan_msg(plan)
		new_plan = self.plan_receiver._build_plan(plan_msg)
		self.assertEqual(type(new_plan), Plan)
		self.assertEqual(plan.actions[0].preds[0]['pred'].name, new_plan.actions[0].preds[0]['pred'].name)
		self.assertListEqual(plan.actions[0].params[0].lArmPose.tolist(), new_plan.actions[0].params[0].rArmPose.tolist())
		self.assertListEqual(pred.params[0].pose.tolist(), new_plan.get_preds(False)[0].params[0].pose.tolist())

