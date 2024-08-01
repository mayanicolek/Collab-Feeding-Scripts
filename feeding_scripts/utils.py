import numpy as np
import time
import socket
from scipy.interpolate import interp1d


class Trajectory(object):

	def __init__(self, xi, T):
		""" create cublic interpolators between waypoints """
		self.xi = np.asarray(xi)
		self.T = T
		self.n_waypoints = xi.shape[0]
		timesteps = np.linspace(0, self.T, self.n_waypoints)
		self.f1 = interp1d(timesteps, self.xi[:,0], kind='cubic')
		self.f2 = interp1d(timesteps, self.xi[:,1], kind='cubic')
		self.f3 = interp1d(timesteps, self.xi[:,2], kind='cubic')
		self.f4 = interp1d(timesteps, self.xi[:,3], kind='cubic')
		self.f5 = interp1d(timesteps, self.xi[:,4], kind='cubic')
		self.f6 = interp1d(timesteps, self.xi[:,5], kind='cubic')
		# self.f7 = interp1d(timesteps, self.xi[:,6], kind='cubic')

	def get(self, t):
		""" get interpolated position """
		if t < 0:
			q = [self.f1(0), self.f2(0), self.f3(0), self.f4(0), self.f5(0), self.f6(0)]
		elif t < self.T:
			q = [self.f1(t), self.f2(t), self.f3(t), self.f4(t), self.f5(t), self.f6(t)]
		else:
			q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), self.f4(self.T), self.f5(self.T), self.f6(self.T)]
		return np.asarray(q)
	

class Robot():

	def __init__(self):
		# self.home = [0.00720989, -1.45442,  0.00506945, -2.21578, -3.280000e-04,  0.832898,  7.856910e-01]
		self.home = [0.166251, 0.359144,  0.317505, -1.75777, -0.0125136,  2.12252,  1.28472]
		# self.home = [0.00720989, -1.45442,  0.00506945, -2.21578, -3.280000e-04,  np.pi/4,  np.pi/4]

	def connect2robot(self, port):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(('172.16.0.3', port))
		s.listen()
		conn, addr = s.accept()
		return conn
	
	def send2robot(self, conn, qdot, control_mode, limit=1.0):
		qdot = np.asarray(qdot)
		scale = np.linalg.norm(qdot)
		if scale > limit:
			qdot *= limit/scale
		send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
		if send_msg == '0.,0.,0.,0.,0.,0.,0.':
			send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
		send_msg = "s," + send_msg + "," + control_mode + ","
		conn.send(send_msg.encode())

	def listen2robot(self, conn):
		state_length = 7 + 42
		message = str(conn.recv(2048))[2:-2]
		state_str = list(message.split(","))
		for idx in range(len(state_str)):
			if state_str[idx] == "s":
				state_str = state_str[idx+1:idx+1+state_length]
				break
		try:
			state_vector = [float(item) for item in state_str]
		except ValueError:
			return None
		if len(state_vector) is not state_length:
			return None
		state_vector = np.asarray(state_vector)
		state = {}
		state["q"] = state_vector[0:7]
		state["J"] = state_vector[7:49].reshape((7,6)).T
		xyz_lin, R = self.joint2pose(state_vector[0:7])
		beta = -np.arcsin(R[2, 0])
		alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
		gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
		xyz_ang = [alpha, beta, gamma]
		xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
		state["x"] = np.array(xyz)
		return state

	def readState(self, conn):
		while True:
			states = self.listen2robot(conn)
			if states is not None:
				break
		return states

	def xdot2qdot(self, xdot, states):
		J_inv = np.linalg.pinv(states["J"])
		return J_inv @ np.asarray(xdot)

	def joint2pose(self, q):
		def RotX(q):
			return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
		def RotZ(q):
			return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
		def TransX(q, x, y, z):
			return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
		def TransZ(q, x, y, z):
			return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
		H1 = TransZ(q[0], 0, 0, 0.333)
		H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
		H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
		H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
		H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
		H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
		H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
		H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
		T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
		R = T[:,:3][:3]
		xyz = T[:,3][:3]
		return xyz, R

	def go2position(self, conn, goal=False):
		if not goal:
			goal = self.home
		total_time = 20.0
		start_time = time.time()
		states = self.readState(conn)
		dist = np.linalg.norm(states["q"] - goal)
		elapsed_time = time.time() - start_time
		while dist > 0.05 and elapsed_time < total_time:
			qdot = np.clip(goal - states["q"], -0.3, 0.3)
			self.send2robot(conn, qdot, "v")
			states = self.readState(conn)
			dist = np.linalg.norm(states["q"] - goal)
			elapsed_time = time.time() - start_time

	def find_pos(self,conn):
		state = self.readState(conn)
		return state['x']
	
	def make_traj(self,start_pos, des_pos, val):
		traj = []
		for idx in range(val):
			traj.append(start_pos + (1/val) * idx * (des_pos - start_pos))
		traj.append(des_pos)
		return np.array(traj)

	def translate_robot(self, conn, target_pos, max_delta=0.01):
		obs = self.find_pos(conn)
		ee_euler = obs[3:]
		target_pos = np.concatenate((target_pos, ee_euler), axis=-1)
		traj = self.make_traj(obs,target_pos,5)
		self.run_traj(conn, traj,5)

	def rotate_robot(self, conn, target_euler, num_steps=30):
		obs = self.find_pos(conn)
		ee_pos = obs[:3]
		print(obs)
		target_pos = np.concatenate((ee_pos, target_euler), axis=-1)
		traj = self.make_traj(obs,target_pos,5)
		self.run_traj(conn, traj,5)

	def move_robot(self, conn, target_pos, target_euler, max_delta=0.005):
		obs = self.find_pos(conn)
		target_pos = np.concatenate((target_pos, target_euler), axis=-1)
		traj = self.make_traj(obs,target_pos,10)
		self.run_traj(conn, traj,10)

	def run_traj(self, conn, traj, val):
		traj = Trajectory(traj, val)
		start_t = time.time()
		while True:
			curr_t = time.time() - start_t
			x_des = traj.get(curr_t)
			state = self.readState(conn)
			x_curr = state['x']

			x_des[3] = self.wrap_angles(x_des[3])
			x_des[4] = self.wrap_angles(x_des[4])
			x_des[5] = self.wrap_angles(x_des[5])
			xdot = 1 * (x_des - x_curr)
			xdot[3] = self.wrap_angles(xdot[3])
			xdot[4] = self.wrap_angles(xdot[4])
			xdot[5] = self.wrap_angles(xdot[5])

			qdot = self.xdot2qdot(xdot, state)

			self.send2robot(conn, qdot, "v")
			time_elapsed = time.time() - start_t

			if time_elapsed >= val:
				break

	def wrap_angles(self, theta):
		if theta < -np.pi:
			theta += 2*np.pi
		elif theta > np.pi:
			theta -= 2*np.pi
		else:
			theta = theta
		return theta
