from core.util_classes.viewer import OpenRAVEViewer
from core.util_classes.openrave_body import OpenRAVEBody
from core.util_classes import robot_predicates
from openravepy import matrixFromAxisAngle, IkParameterization, IkParameterizationType, IkFilterOptions, Environment, Planner, RaveCreatePlanner, RaveCreateTrajectory, matrixFromAxisAngle, CollisionReport, RaveCreateCollisionChecker
import core.util_classes.baxter_constants as const
from collections import OrderedDict
from sco.expr import Expr
import math
import numpy as np
PI = np.pi

#These functions are helper functions that can be used by many robots
def get_random_dir():
    """
        This helper function generates a random 2d unit vectors
    """
    rand_dir = np.random.rand(2) - 0.5
    rand_dir = rand_dir/np.linalg.norm(rand_dir)
    return rand_dir

def get_random_theta():
    """
        This helper function generates a random angle between -PI to PI
    """
    theta =  2*PI*np.random.rand(1) - PI
    return theta[0]

def smaller_ang(x):
    """
        This helper function takes in an angle in radius, and returns smaller angle
        Ex. 5pi/2 -> PI/2
            8pi/3 -> 2pi/3
    """
    return (x + PI)%(2*PI) - PI

def closer_ang(x,a,dir=0):
    """
        find angle y (==x mod 2*PI) that is close to a
        dir == 0: minimize absolute value of difference
        dir == 1: y > x
        dir == 2: y < x
    """
    if dir == 0:
        return a + smaller_ang(x-a)
    elif dir == 1:
        return a + (x-a)%(2*PI)
    elif dir == -1:
        return a + (x-a)%(2*PI) - 2*PI

def get_ee_transform_from_pose(pose, rotation):
    """
        This helper function that returns the correct end effector rotation axis (perpendicular to gripper side)
    """
    ee_trans = OpenRAVEBody.transform_from_obj_pose(pose, rotation)
    #the rotation is to transform the tool frame into the end effector transform
    rot_mat = matrixFromAxisAngle([0, PI/2, 0])
    ee_rot_mat = ee_trans[:3, :3].dot(rot_mat[:3, :3])
    ee_trans[:3, :3] = ee_rot_mat
    return ee_trans

def closer_joint_angles(pos,seed):
    """
        This helper function cleans up the dof if any angle is greater than 2 PI
    """
    result = np.array(pos)
    for i in [2,4,6]:
        result[i] = closer_ang(pos[i],seed[i],0)
    return result

def get_ee_from_target(targ_pos, targ_rot):
    """
        This function samples all possible EE Poses around the target

        target_pos: position of target we want to sample ee_pose form
        target_rot: rotation of target we want to sample ee_pose form
        return: list of ee_pose tuple in the format of (ee_pos, ee_rot) around target axis
    """
    possible_ee_poses = []
    ee_pos = targ_pos.copy()
    target_trans = OpenRAVEBody.transform_from_obj_pose(targ_pos, targ_rot)
    # rotate can's local z-axis by the amount of linear spacing between 0 to 2pi
    angle_range = np.linspace(PI/3, PI/3 + PI*2, num=const.EE_ANGLE_SAMPLE_SIZE)
    for rot in angle_range:
        target_trans = OpenRAVEBody.transform_from_obj_pose(targ_pos, targ_rot)
        # rotate new ee_pose around can's rotation axis
        rot_mat = matrixFromAxisAngle([0, 0, rot])
        ee_trans = target_trans.dot(rot_mat)
        ee_rot = OpenRAVEBody.obj_pose_from_transform(ee_trans)[3:]
        possible_ee_poses.append((ee_pos, ee_rot))
    return possible_ee_poses

def closest_arm_pose(arm_poses, cur_arm_pose):
    """
        Given a list of possible arm poses, select the one with the least displacement from current arm pose
    """
    min_change = np.inf
    chosen_arm_pose = None
    for arm_pose in arm_poses:
        change = sum((arm_pose - cur_arm_pose)**2)
        if change < min_change:
            chosen_arm_pose = arm_pose
            min_change = change
    return chosen_arm_pose

def closest_base_poses(base_poses, robot_base):
    """
        Given a list of possible base poses, select the one with the least displacement from current base pose
    """
    val, chosen = np.inf, robot_base
    if len(base_poses) <= 0:
        return chosen
    for base_pose in base_poses:
        diff = base_pose - robot_base
        distance = reduce(lambda x, y: x**2 + y, diff, 0)
        if distance < val:
            chosen = base_pose
            val = distance
    return chosen

def lin_interp_traj(start, end, time_steps):
    """
    This helper function returns a linear trajectory from start pose to end pose
    """
    assert start.shape == end.shape
    if time_steps == 0:
        assert np.allclose(start, end)
        return start.copy()
    rows = start.shape[0]
    traj = np.zeros((rows, time_steps+1))

    for i in range(rows):
        traj_row = np.linspace(start[i], end[i], num=time_steps+1)
        traj[i, :] = traj_row
    return traj

def plot_transform(env, T, s=0.1):
    """
    Helper function mainly used for debugging purpose
    Plots transform T in openrave environment.
    S is the length of the axis markers.
    """
    h = []
    x = T[0:3,0]
    y = T[0:3,1]
    z = T[0:3,2]
    o = T[0:3,3]
    h.append(env.drawlinestrip(points=np.array([o, o+s*x]), linewidth=3.0, colors=np.array([(1,0,0),(1,0,0)])))
    h.append(env.drawlinestrip(points=np.array([o, o+s*y]), linewidth=3.0, colors=np.array(((0,1,0),(0,1,0)))))
    h.append(env.drawlinestrip(points=np.array([o, o+s*z]), linewidth=3.0, colors=np.array(((0,0,1),(0,0,1)))))
    return h

def get_expr_mult(coeff, expr):
    """
        Multiply expresions with coefficients
    """
    new_f = lambda x: coeff*expr.eval(x)
    new_grad = lambda x: coeff*expr.grad(x)
    return Expr(new_f, new_grad)

# Sample base values to face the target
def sample_base(target_pose, base_pose):
    vec = target_pose[:2] - np.zeros((2,))
    vec = vec / np.linalg.norm(vec)
    theta = math.atan2(vec[1], vec[0])
    return theta

# Resampling For IK
def get_ik_transform(pos, rot):
    trans = OpenRAVEBody.transform_from_obj_pose(pos, rot)
    # Openravepy flip the rotation axis by 90 degree, thus we need to change it back
    rot_mat = matrixFromAxisAngle([0, PI/2, 0])
    trans_mat = trans[:3, :3].dot(rot_mat[:3, :3])
    trans[:3, :3] = trans_mat
    return trans

def get_ik_from_pose(pos, rot, robot, manip_name):
    trans = get_ik_transform(pos, rot)
    solution = get_ik_solutions(robot, manip_name, trans)
    return solution

def get_ik_solutions(robot, manip_name, trans):
    manip = robot.GetManipulator(manip_name)
    iktype = IkParameterizationType.Transform6D
    solutions = manip.FindIKSolutions(IkParameterization(trans, iktype),IkFilterOptions.CheckEnvCollisions)
    if len(solutions) == 0:
        return None
    return closest_arm_pose(solutions, robot.GetActiveDOFValues()[manip.GetArmIndices()])




# Get RRT Planning Result
def get_rrt_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    active_dofs = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof) # set goal to all ones
    # # forces parabolic planning with 40 iterations
    # import ipdb; ipdb.set_trace()
    params.SetExtraParameters("""<_postprocessing planner="parabolicsmoother">
        <_nmaxiterations>20</_nmaxiterations>
    </_postprocessing>""")

    planner=RaveCreatePlanner(env,'birrt')
    planner.InitPlan(robot, params)

    traj = RaveCreateTrajectory(env,'')
    result = planner.PlanPath(traj)
    if result == False:
        robot.SetActiveDOFs(active_dofs)
        return None
    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data=traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(data,robot,robot.GetActiveDOFIndices())
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    robot.SetActiveDOFs(active_dofs)
    return np.array(traj_list)


def process_traj(raw_traj, timesteps):
    """
        Process raw_trajectory so that it's length is desired timesteps
        when len(raw_traj) > timesteps
            sample Trajectory by space to reduce trajectory size
        when len(raw_traj) < timesteps
            append last timestep pose util the size fits

        Note: result_traj includes init_dof and end_dof
    """
    result_traj = []
    if len(raw_traj) == timesteps:
        result_traj = raw_traj.copy()
    else:
        traj_arr = [0]
        result_traj.append(raw_traj[0])
        #calculate accumulative distance
        for i in range(len(raw_traj)-1):
            traj_arr.append(traj_arr[-1] + np.linalg.norm(raw_traj[i+1] - raw_traj[i]))
        step_dist = traj_arr[-1]/(timesteps - 1)
        process_dist, i = 0, 1
        while i < len(traj_arr)-1:
            if traj_arr[i] == process_dist + step_dist:
                result_traj.append(raw_traj[i])
                process_dist += step_dist
            elif traj_arr[i] < process_dist+step_dist < traj_arr[i+1]:
                dist = process_dist+step_dist - traj_arr[i]
                displacement = (raw_traj[i+1] - raw_traj[i])/(traj_arr[i+1]-traj_arr[i])*dist
                result_traj.append(raw_traj[i]+displacement)
                process_dist += step_dist
            else:
                i += 1
    result_traj.append(raw_traj[-1])
    return np.array(result_traj).T


def get_ompl_rrtconnect_traj(env, robot, active_dof, init_dof, end_dof):
    # assert body in env.GetRobot()
    dof_inds = robot.GetActiveDOFIndices()
    robot.SetActiveDOFs(active_dof)
    robot.SetActiveDOFValues(init_dof)

    params = Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(end_dof) # set goal to all ones
    # forces parabolic planning with 40 iterations
    planner=RaveCreatePlanner(env,'OMPL_RRTConnect')
    planner.InitPlan(robot, params)
    traj = RaveCreateTrajectory(env,'')
    planner.PlanPath(traj)

    traj_list = []
    for i in range(traj.GetNumWaypoints()):
        # get the waypoint values, this holds velocites, time stamps, etc
        data=traj.GetWaypoint(i)
        # extract the robot joint values only
        dofvalues = traj.GetConfigurationSpecification().ExtractJointValues(data,robot,robot.GetActiveDOFIndices())
        # raveLogInfo('waypint %d is %s'%(i,np.round(dofvalues, 3)))
        traj_list.append(np.round(dofvalues, 3))
    robot.SetActiveDOFs(dof_inds)
    return traj_list



def get_col_free_armPose(pred, negated, t, plan):
    robot = pred.robot
    body = pred._param_to_body[robot]
    arm_pose = None
    old_arm_pose = robot.rArmPose[:, t].copy()
    body.set_pose([0,0, robot.pose[0, t]])
    body.set_dof({'rArmPose': robot.rArmPose[:, t].flatten()})
    dof_inds = body.env_body.GetManipulator("right_arm").GetArmIndices()

    arm_pose = np.random.random_sample((len(dof_inds),))*1 - 0.5
    arm_pose = arm_pose + old_arm_pose
    return arm_pose

def resample_pred(pred, negated, t, plan):
    res, attr_inds = [], OrderedDict()
    # Determine which action failed first
    rs_action, ref_index = None, None
    for i in range(len(plan.actions)):
        active = plan.actions[i].active_timesteps
        if active[0] <= t <= active[1]:
            rs_action, ref_index = plan.actions[i], i
            break

    if rs_action.name == 'moveto' or rs_action.name == 'movetoholding':
        return resample_move(plan, t, pred, rs_action, ref_index)
    elif rs_action.name == 'grasp' or rs_action.name == 'putdown':
        return resample_pick_place(plan, t, pred, rs_action, ref_index)
    else:
        raise NotImplemented

def resample_move(plan, t, pred, rs_action, ref_index):
    res, attr_inds = [], OrderedDict()
    robot = rs_action.params[0]
    act_range = rs_action.active_timesteps
    body = robot.openrave_body.env_body
    manip_name = "right_arm"
    active_dof = body.GetManipulator(manip_name).GetArmIndices()
    active_dof = np.hstack([[0], active_dof])
    robot.openrave_body.set_dof({'rGripper': 0.02})

    # In pick place domain, action flow is natually:
    # moveto -> grasp -> movetoholding -> putdown
    sampling_trace = None
    #rs_param is pdp_target0
    rs_param = rs_action.params[2]
    if ref_index + 1 < len(plan.actions):
        # take next action's ee_pose and find it's ik value.
        # ref_param is ee_target0
        ref_action = plan.actions[ref_index + 1]
        ref_range = ref_action.active_timesteps
        ref_param = ref_action.params[4]
        event_timestep = (ref_range[1] - ref_range[0])/2
        pose = robot.pose[:, event_timestep]

        arm_pose = get_ik_from_pose(ref_param.value, ref_param.rotation, body, manip_name)
        # In the case ee_pose wasn't even feasible, resample other preds
        if arm_pose is None:
            return None, None

        sampling_trace = {'data': {rs_param.name: {'type': rs_param.get_type(), 'rArmPose': arm_pose, 'value': pose}}, 'timestep': t, 'pred': pred, 'action': rs_action.name}
        add_to_attr_inds_and_res(t, attr_inds, res, rs_param, [('rArmPose', arm_pose), ('value', pose)])

    else:
        arm_pose = rs_action.params[2].rArmPose[:,0].flatten()
        pose = rs_action.params[2].value[:,0]

    """Resample Trajectory by BiRRT"""
    init_arm_dof = rs_action.params[1].rArmPose[:,0].flatten()
    init_pose_dof = rs_action.params[1].value[:,0]
    init_dof = np.hstack([init_pose_dof, init_arm_dof])
    end_dof = np.hstack([pose, arm_pose])

    raw_traj = get_rrt_traj(plan.env, body, active_dof, init_dof, end_dof)
    if raw_traj == None and sampling_trace != None:
        # In the case resampled poses resulted infeasible rrt trajectory
        sampling_trace['reward'] = -1
        plan.sampling_trace.append(sampling_trace)
        return np.array(res), attr_inds
    elif raw_traj == None and sampling_trace == None:
        # In the case resample is just not possible, resample other preds
        return None, None
    # Restore dof0
    body.SetActiveDOFValues(np.hstack([[0], body.GetActiveDOFValues()[1:]]))
    # initailize feasible trajectory
    result_traj = process_traj(raw_traj, act_range[1] - act_range[0] + 2).T[1:-1]
    ts = 1
    for traj in result_traj:
        add_to_attr_inds_and_res(act_range[0] + ts, attr_inds, res, robot, [('rArmPose', traj[1:]), ('pose', traj[:1])])
        ts += 1
    # import ipdb; ipdb.set_trace()
    return np.array(res), attr_inds

def resample_pick_place(plan, t, pred, rs_action, ref_index):
    res, attr_inds = [], OrderedDict()
    robot = rs_action.params[0]
    act_range = rs_action.active_timesteps
    body = robot.openrave_body.env_body
    manip_name = "right_arm"
    active_dof = body.GetManipulator(manip_name).GetArmIndices()
    active_dof = np.hstack([[0], active_dof])
    # In pick place domain, action flow is natually:
    # moveto -> grasp -> movetoholding -> putdown
    #rs_param is ee_poses, ref_param is target
    rs_param = rs_action.params[4]
    ref_param = rs_action.params[2]
    ee_poses = get_ee_from_target(ref_param.value, ref_param.rotation)
    for samp_ee in ee_poses:
        arm_pose = get_ik_from_pose(samp_ee[0].flatten(), samp_ee[1].flatten(), body, manip_name)
        if arm_pose is not None:
            break

    if arm_pose is None:
        return None, None
    sampling_trace = {'data': {rs_param.name: {'type': rs_param.get_type(), 'value': samp_ee[0], 'rotation': samp_ee[1]}}, 'timestep': t, 'pred': pred, 'action': rs_action.name}
    add_to_attr_inds_and_res(t, attr_inds, res, rs_param, [('value', samp_ee[0].flatten()), ('rotation', samp_ee[1].flatten())])

    """Resample Trajectory by BiRRT"""
    if t < (act_range[1] - act_range[0])/2 and ref_index >= 1:
        # if resample time occured before grasp or putdown.
        # resample initial poses as well
        # ref_action is move
        ref_action = plan.actions[ref_index - 1]
        ref_range = ref_action.active_timesteps

        start_pose = ref_action.params[1]
        init_dof = start_pose.rArmPose[:,0].flatten()
        init_dof = np.hstack([start_pose.value[:,0], init_dof])
        end_dof = np.hstack([robot.pose[:,t], arm_pose])
        timesteps = act_range[1] - ref_range[0] + 2

        init_timestep = ref_range[0]

    else:
        start_pose = rs_action.params[3]

        init_dof = start_pose.rArmPose[:,0].flatten()
        init_dof = np.hstack([start_pose.value[:,0], init_dof])
        end_dof = np.hstack([robot.pose[:,t], arm_pose])
        timesteps = act_range[1] - act_range[0] + 2

        init_timestep = act_range[0]

    raw_traj = get_rrt_traj(plan.env, body, active_dof, init_dof, end_dof)
    if raw_traj == None:
        # In the case resampled poses resulted infeasible rrt trajectory
        plan.sampling_trace.append(sampling_trace)
        plan.sampling_trace[-1]['reward'] = -1
        return np.array(res), attr_inds

    # Restore dof0
    body.SetActiveDOFValues(np.hstack([[0], body.GetActiveDOFValues()[1:]]))
    # initailize feasible trajectory
    result_traj = process_traj(raw_traj, timesteps).T[1:-1]

    ts = 1
    for traj in result_traj:
        add_to_attr_inds_and_res(init_timestep + ts, attr_inds, res, robot, [('rArmPose', traj[1:]), ('pose', traj[:1])])
        ts += 1
    # import ipdb; ipdb.set_trace()

    if init_timestep != act_range[0]:
        sampling_trace['data'][start_pose.name] = {'type': start_pose.get_type(), 'rArmPose': robot.rArmPose[:, act_range[0]], 'value': robot.pose[:, act_range[0]]}
        add_to_attr_inds_and_res(init_timestep + ts, attr_inds, res, start_pose, [('rArmPose', sampling_trace['data'][start_pose.name]['rArmPose']), ('value', sampling_trace['data'][start_pose.name]['value'])])

    return np.array(res), attr_inds

def resample_eereachable_rrt(pred, negated, t, plan, inv = False):
    # Preparing the variables
    attr_inds, res = OrderedDict(), []
    robot, rave_body = pred.robot, pred.robot.openrave_body
    target_pos, target_rot = pred.ee_pose.value.flatten(), pred.ee_pose.rotation.flatten()
    body = rave_body.env_body
    manip_name = "right_arm"
    active_dof = body.GetManipulator(manip_name).GetArmIndices()
    active_dof = np.hstack([[0], active_dof])
    # Make sure baxter is well positioned in the env
    rave_body.set_pose([0,0,robot.pose[:,t]])
    rave_body.set_dof({'lArmPose': robot.lArmPose[:, t].flatten(),
                       'rArmPose': robot.rArmPose[:, t].flatten(),
                       "lGripper": np.array([0.02]), "rGripper": np.array([0.02])})
    for param in plan.params.values():
        if not param.is_symbol() and param != robot:
            param.openrave_body.set_pose(param.pose[:, t].flatten(), param.rotation[:, t].flatten())
    # Resample poses at grasping time
    grasp_arm_pose = get_ik_from_pose(target_pos, target_rot, body, manip_name)

    # When Ik infeasible
    if grasp_arm_pose is None:
        return None, None
    add_to_attr_inds_and_res(t, attr_inds, res, robot, [('rArmPose', grasp_arm_pose.copy()), ('pose', robot.pose[:,t])])
    # Store sampled pose
    plan.sampling_trace.append({'type': robot.get_type(), 'data':{'rArmPose': grasp_arm_pose}, 'timestep': t, 'pred': pred, 'action': "grasp"})
    # Prepare grasping direction and lifting direction
    manip_trans = body.GetManipulator("right_arm").GetTransform()
    pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
    manip_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
    if inv:
        # inverse resample_eereachable used in putdown action
        approach_dir = manip_trans[:3,:3].dot(np.array([0,0,-1]))
        retreat_dir = manip_trans[:3,:3].dot(np.array([-1,0,0]))
        approach_dir = approach_dir / np.linalg.norm(approach_dir) * const.APPROACH_DIST
        retreat_dir = -retreat_dir/np.linalg.norm(retreat_dir) * const.RETREAT_DIST
    else:
        # Normal resample eereachable used in grasp action
        approach_dir = manip_trans[:3,:3].dot(np.array([-1,0,0]))
        retreat_dir = manip_trans[:3,:3].dot(np.array([0,0,-1]))
        approach_dir = -approach_dir / np.linalg.norm(approach_dir) * const.APPROACH_DIST
        retreat_dir = retreat_dir/np.linalg.norm(retreat_dir) * const.RETREAT_DIST

    resample_failure = False
    # Resample entire approaching and retreating traj
    for i in range(const.EEREACHABLE_STEPS):
        approach_pos = target_pos + approach_dir * (3-i)
        approach_arm_pose = get_ik_from_pose(approach_pos, target_rot, body,
                                             'right_arm')
        retreat_pos = target_pos + retreat_dir * (i+1)
        retreat_arm_pose = get_ik_from_pose(retreat_pos, target_rot, body, 'right_arm')

        if approach_arm_pose is None or retreat_arm_pose is None:
            resample_failure = True
        add_to_attr_inds_and_res(t-3+i, attr_inds, res, robot,[('rArmPose',
                                 approach_arm_pose)])
        add_to_attr_inds_and_res(t+1+i, attr_inds, res, robot,[('rArmPose', retreat_arm_pose)])
    # Ik infeasible
    if resample_failure:
        plan.sampling_trace[-1]['reward'] = -1
        return None, None
    # lock the variables
    robot._free_attrs['rArmPose'][:, t-const.EEREACHABLE_STEPS: t+const.EEREACHABLE_STEPS+1] = 0
    robot._free_attrs['pose'][:, t-const.EEREACHABLE_STEPS: t+const.EEREACHABLE_STEPS+1] = 0
    # finding initial pose
    init_timestep, ref_index = 0, 0
    for i in range(len(plan.actions)):
        act_range = plan.actions[i].active_timesteps
        if act_range[0] <= t <= act_range[1]:
            init_timestep = act_range[0]
            ref_index = i

    if pred.ee_resample is True and ref_index > 0:
        init_timestep = plan.actions[ref_index - 1].active_timesteps[0]

    init_dof = robot.rArmPose[:, init_timestep].flatten()
    init_dof = np.hstack([robot.pose[:, init_timestep], init_dof])
    end_dof = robot.rArmPose[:, t - const.EEREACHABLE_STEPS].flatten()
    end_dof = np.hstack([robot.pose[:, t - const.EEREACHABLE_STEPS], end_dof])
    timesteps = t - const.EEREACHABLE_STEPS - init_timestep + 2

    raw_traj = get_rrt_traj(plan.env, body, active_dof, init_dof, end_dof)
    # Restore dof0
    dof = body.GetActiveDOFValues()
    dof[0] = 0
    body.SetActiveDOFValues(dof)
    # trajectory is infeasible
    if raw_traj == None:
        plan.sampling_trace[-1]['reward'] = -1
        return None, None
    # initailize feasible trajectory
    result_traj = process_traj(raw_traj, timesteps).T[1:-1]
    ts = 1
    for traj in result_traj:
        add_to_attr_inds_and_res(init_timestep + ts, attr_inds, res, robot, [('rArmPose', traj[1:]), ('pose', traj[:1])])
        ts += 1

    pred.ee_resample = True
    can = plan.params['can0']
    can.openrave_body.set_pose(can.pose[:, t], can.rotation[:, t])
    rave_body.set_dof({'rArmPose': robot.rArmPose[:, t]})
    return np.array(res), attr_inds

def resample_obstructs(pred, negated, t, plan):
    # Variable that needs to added to BoundExpr and latter pass to the planner
    attr_inds = OrderedDict()
    res = []
    robot = pred.robot
    body = pred._param_to_body[robot].env_body
    manip = body.GetManipulator("right_arm")
    arm_inds = manip.GetArmIndices()
    lb_limit, ub_limit = body.GetDOFLimits()
    joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds])/20.
    original_pose, arm_pose = robot.rArmPose[:, t], robot.rArmPose[:, t]

    obstacle_col_pred = [col_pred for col_pred in plan.get_preds(True) if isinstance(col_pred, robot_predicates.RCollides)]
    if len(obstacle_col_pred) == 0:
        obstacle_col_pred = None
    else:
        obstacle_col_pred = obstacle_col_pred[0]

    while not pred.test(t, negated) or (obstacle_col_pred is not None and not obstacle_col_pred.test(t, negated)):
        step_sign = np.ones(len(arm_inds))
        step_sign[np.random.choice(len(arm_inds), len(arm_inds)/2, replace=False)] = -1
        # Ask in collision pose to randomly move a step, hopefully out of collision
        arm_pose = original_pose + np.multiply(step_sign, joint_step)
        add_to_attr_inds_and_res(t, attr_inds, res, robot,[('rArmPose', arm_pose)])

    robot._free_attrs['rArmPose'][:, t] = 0
    return np.array(res), attr_inds

def resample_rcollides(pred, negated, t, plan):
    # Variable that needs to added to BoundExpr and latter pass to the planner
    JOINT_STEP = 20
    STEP_DECREASE_FACTOR = 1.5
    ATTEMPT_SIZE = 7
    LIN_SAMP_RANGE = 5

    attr_inds = OrderedDict()
    res = []
    robot, rave_body = pred.robot, pred._param_to_body[pred.robot]
    body = rave_body.env_body
    manip = body.GetManipulator("right_arm")
    arm_inds = manip.GetArmIndices()
    lb_limit, ub_limit = body.GetDOFLimits()
    step_factor = JOINT_STEP
    joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds])/ step_factor
    original_pose, arm_pose = robot.rArmPose[:, t].copy(), robot.rArmPose[:, t].copy()
    rave_body.set_pose([0,0,robot.pose[:, t]])
    rave_body.set_dof({"lArmPose": robot.lArmPose[:, t].flatten(),
                       "lGripper": robot.lGripper[:, t].flatten(),
                       "rArmPose": robot.rArmPose[:, t].flatten(),
                       "rGripper": robot.rGripper[:, t].flatten()})

    ## Determine the range we should resample
    pred_list = [act_pred['active_timesteps'] for act_pred in plan.actions[0].preds if act_pred['pred'].spacial_anchor == True]
    start, end = 0, plan.horizon-1
    for action in plan.actions:
        if action.active_timesteps[0] <= t and action.active_timesteps[1] > t:
            for act_pred in plan.actions[0].preds:
                if act_pred['pred'].spacial_anchor == True:
                    if act_pred['active_timesteps'][0] + act_pred['pred'].active_range[0] > t:
                        end = min(end, act_pred['active_timesteps'][0] + act_pred['pred'].active_range[0])
                    if act_pred['active_timesteps'][1] + act_pred['pred'].active_range[1] < t:
                        start = max(start, act_pred['active_timesteps'][1] + act_pred['pred'].active_range[1])

    desired_end_pose = robot.rArmPose[:, end]
    current_end_pose = robot.rArmPose[:, t]
    col_report = CollisionReport()
    collisionChecker = RaveCreateCollisionChecker(plan.env,'pqp')
    count = 1
    while (body.CheckSelfCollision() or
           collisionChecker.CheckCollision(body, report=col_report) or
           col_report.minDistance <= pred.dsafe):
        step_sign = np.ones(len(arm_inds))
        step_sign[np.random.choice(len(arm_inds), len(arm_inds)/2, replace=False)] = -1
        # Ask in collision pose to randomly move a step, hopefully out of collision
        arm_pose = original_pose + np.multiply(step_sign, joint_step)
        rave_body.set_dof({"rArmPose": arm_pose})
        # arm_pose = body.GetActiveDOFValues()[arm_inds]
        if not count % ATTEMPT_SIZE:
            step_factor = step_factor/STEP_DECREASE_FACTOR
            joint_step = (ub_limit[arm_inds] - lb_limit[arm_inds])/ step_factor
        count += 1

        # For Debug
        rave_body.set_pose([0,0,robot.pose[:, t]])
    add_to_attr_inds_and_res(t, attr_inds, res, robot,[('rArmPose', arm_pose)])
    robot._free_attrs['rArmPose'][:, t] = 0


    start, end = max(start, t-LIN_SAMP_RANGE), min(t+LIN_SAMP_RANGE, end)
    rcollides_traj = np.hstack([lin_interp_traj(robot.rArmPose[:, start], arm_pose, t-start), lin_interp_traj(arm_pose, robot.rArmPose[:, end], end - t)[:, 1:]]).T
    i = start + 1
    for traj in rcollides_traj[1:-1]:
        add_to_attr_inds_and_res(i, attr_inds, res, robot, [('rArmPose', traj)])
        i +=1


    return np.array(res), attr_inds

# Alternative approaches, frequently failed, Not used
def get_col_free_armPose_ik(pred, negated, t, plan):
    ee_pose = OpenRAVEBody.obj_pose_from_transform(body.env_body.GetManipulator('right_arm').GetTransform())
    pos, rot = ee_pose[:3], ee_pose[3:]
    while arm_pose is None and iteration < const.MAX_ITERATION_STEP:
    # for i in range(const.NUM_RESAMPLES):
        pos_bias = np.random.random_sample((3,))*const.BIAS_RADIUS*2 - const.BIAS_RADIUS
        rot_bias = np.random.random_sample((3,))*const.ROT_BIAS*2 - const.ROT_BIAS
        # print pos_bias, rot_bias, iteration
        print pos_bias, rot_bias
        iteration += 1
        arm_pose = get_ik_from_pose(pos + pos_bias, rot + rot_bias, body.env_body, 'right_arm')
        if arm_pose is not None:
            print iteration
            body.set_dof({'rArmPose': arm_pose})

def sample_arm_pose(robot_body, old_arm_pose=None):
    dof_inds = robot_body.GetManipulator("right_arm").GetArmIndices()
    lb_limit, ub_limit = robot_body.GetDOFLimits()
    active_ub = ub_limit[dof_inds].flatten()
    active_lb = lb_limit[dof_inds].flatten()
    if old_arm_pose is not None:
        arm_pose = np.random.random_sample((len(dof_inds),)) - 0.5
        arm_pose = np.multiply(arm_pose, (active_ub - active_lb)/5) + old_arm_pose
    else:
        arm_pose = np.random.random_sample((len(dof_inds),))
        arm_pose = np.multiply(arm_pose, active_ub - active_lb) + active_lb
    return arm_pose

def add_to_attr_inds_and_res(t, attr_inds, res, param, attr_name_val_tuples):
    param_attr_inds = []
    if param.is_symbol():
        t = 0
    for attr_name, val in attr_name_val_tuples:
        inds = np.where(param._free_attrs[attr_name][:, t])[0]
        getattr(param, attr_name)[inds, t] = val[inds]
        res.extend(val[inds].flatten().tolist())
        param_attr_inds.append((attr_name, inds, t))
    if param in attr_inds:
        attr_inds[param].extend(param_attr_inds)
    else:
        attr_inds[param] = param_attr_inds

def resample_eereachable(pred, negated, t, plan):
    attr_inds, res = OrderedDict(), []
    robot, rave_body = pred.robot, pred._param_to_body[pred.robot]
    target_pos, target_rot = pred.ee_pose.value.flatten(), pred.ee_pose.rotation.flatten()
    body = rave_body.env_body
    rave_body.set_pose([0,0,robot.pose[0, t]])
    # Resample poses at grasping time
    grasp_arm_pose = get_ik_from_pose(target_pos, target_rot, body, 'right_arm')
    add_to_attr_inds_and_res(t, attr_inds, res, robot, [('rArmPose', grasp_arm_pose.copy())])

    plan.sampling_trace.append({'type': robot.get_type(), 'data': {'rArmPose': grasp_arm_pose}, 'timestep': t, 'pred': pred, 'action': "grasp"})

    # Setting poses for environments to extract transform infos
    dof_value_map = {"lArmPose": robot.lArmPose[:,t].reshape((7,)),
                     "lGripper": 0.02,
                     "rArmPose": grasp_arm_pose,
                     "rGripper": 0.02}
    rave_body.set_dof(dof_value_map)
    # Prepare grasping direction and lifting direction
    manip_trans = body.GetManipulator("right_arm").GetTransform()
    pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
    manip_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
    gripper_direction = manip_trans[:3,:3].dot(np.array([-1,0,0]))
    lift_direction = manip_trans[:3,:3].dot(np.array([0,0,-1]))
    # Resample grasping and retreating traj
    for i in range(const.EEREACHABLE_STEPS):
        approach_pos = target_pos - gripper_direction / np.linalg.norm(gripper_direction) * const.APPROACH_DIST * (3-i)
        # rave_body.set_pose([0,0,robot.pose[0, t-3+i]])
        approach_arm_pose = get_ik_from_pose(approach_pos, target_rot, body,
                                             'right_arm')
        # rave_body.set_dof({"rArmPose": approach_arm_pose})
        add_to_attr_inds_and_res(t-3+i, attr_inds, res, robot,[('rArmPose',
                                 approach_arm_pose)])

        retreat_pos = target_pos + lift_direction/np.linalg.norm(lift_direction) * const.RETREAT_DIST * (i+1)
        # rave_body.set_pose([0,0,robot.pose[0, t+1+i]])
        retreat_arm_pose = get_ik_from_pose(retreat_pos, target_rot, body, 'right_arm')
        add_to_attr_inds_and_res(t+1+i, attr_inds, res, robot,[('rArmPose', retreat_arm_pose)])

    robot._free_attrs['rArmPose'][:, t-const.EEREACHABLE_STEPS: t+const.EEREACHABLE_STEPS+1] = 0
    robot._free_attrs['pose'][:, t-const.EEREACHABLE_STEPS: t+const.EEREACHABLE_STEPS+1] = 0
    return np.array(res), attr_inds

def resample_rrt_planner(pred, netgated, t, plan):
    startp, endp = pred.startp, pred.endp
    robot = pred.robot
    body = pred._param_to_body[robot].env_body
    manip_trans = body.GetManipulator("right_arm").GetTransform()
    pose = OpenRAVEBody.obj_pose_from_transform(manip_trans)
    manip_trans = OpenRAVEBody.get_ik_transform(pose[:3], pose[3:])
    gripper_direction = manip_trans[:3,:3].dot(np.array([-1,1,0]))
    lift_direction = manip_trans[:3,:3].dot(np.array([0,0,-1]))
    active_dof = body.GetManipulator("right_arm").GetArmIndices()
    attr_inds = OrderedDict()
    res = []
    pred_test = [not pred.test(k, negated) for k in range(20)]
    resample_ts = np.where(pred_test)[0]
    start, end = resample_ts[0]-1, resample_ts[-1]+1

    rave_body = pred._param_to_body[pred.robot]
    dof_value_map = {"lArmPose": pred.robot.lArmPose[:, start],
                     "lGripper": 0.02,
                     "rArmPose": pred.robot.rArmPose[:, start],
                     "rGripper": 0.02}
    rave_body.set_dof(dof_value_map)
    rave_body.set_pose([0,0,pred.robot.pose[:, start][0]])

    body = pred._param_to_body[pred.robot].env_body
    active_dof = body.GetManipulator('right_arm').GetArmIndices()
    r_arm = pred.robot.rArmPose
    traj = get_rrt_traj(plan.env, body, active_dof, r_arm[:, start], r_arm[:, end])
    result = process_traj(traj, end - start)
    body.SetActiveDOFs(range(18))
    for time in range(start+1, end):
        robot_attr_name_val_tuples = [('rArmPose', result[:, time - start-1])]
        add_to_attr_inds_and_res(time, attr_inds, res, pred.robot, robot_attr_name_val_tuples)
    return np.array(res), attr_inds
