import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class SnakeEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def __init__(self):
		self.world = np.zeros((32,32))
		self.snake_head = np.array([5,5])
		self.snake_tail = self.snake_head + np.array([1,0])
		self.snake_indicies = [(self.snake_head[0],self.snake_head[1]),(self.snake_tail[0],self.snake_tail[1])]
		self.world[self.snake_head[0]][self.snake_head[1]] = 10
		self.world[self.snake_tail[0]][self.snake_tail[1]] = 1
		self.apple = np.random.randint(0,32,2)
		self.update_apple()
		self.action_space = spaces.Discrete(3)
		self.high = np.ones(32*32)
		self.observation_space = spaces.Box(-10*self.high, 10*self.high, dtype=np.float32)
		self.snake_orientation = 0
		self.state = None
		self.update_state()
		self.t_since_a = 0
		self.start_dist = np.linalg.norm(self.snake_head-self.apple,1)
		self.done = False
		self.alpha = .999999
		
	def get_index(x,y):
		return 32*x+y	

	def update_state(self):
		a = 0
		b = 0
		c = 0
		apa = 0 
		apr = 0
		apl = 0
		ta  = 0
		tr  = 0
		tl  = 0
		if self.snake_indicies[0][0]-self.snake_indicies[1][0] == -1 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == 0:
			if self.apple[1]-self.snake_indicies[0][1]<0:
				apl = 1
			if self.apple[1]-self.snake_indicies[0][1]>0:
				apr = 1
			if self.apple[0]-self.snake_indicies[0][0]<0:
				apa = 1
			if np.mean(self.snake_indicies,0)[1]-self.snake_indicies[0][1]<0:
				tl = 1
			if np.mean(self.snake_indicies,0)[1]-self.snake_indicies[0][1]>0:
				tr = 1
			if np.mean(self.snake_indicies,0)[0]-self.snake_indicies[0][0]<0:
				ta = 1
			if (self.snake_head+np.array([-1,0]))[0] not in range(32) or self.world[(self.snake_head+np.array([-1,0]))[0]%32][(self.snake_head+np.array([-1,0]))[1]%32]==1:
				a = 1
			if (self.snake_head+np.array([0,-1]))[1] not in range(32) or self.world[(self.snake_head+np.array([0,-1]))[0]%32][(self.snake_head+np.array([0,-1]))[1]%32]==1:
				b = 1
			if (self.snake_head+np.array([0,1]))[1] not in range(32) or self.world[(self.snake_head+np.array([0,1]))[0]%32][(self.snake_head+np.array([0,1]))[1]%32]==1:
				c = 1
		elif self.snake_indicies[0][0]-self.snake_indicies[1][0] == 1 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == 0:
			if self.apple[1]-self.snake_indicies[0][1]<0:
				apr = 1
			if self.apple[1]-self.snake_indicies[0][1]>0:
				apl = 1
			if self.apple[0]-self.snake_indicies[0][0]>0:
				apa = 1
			if np.mean(self.snake_indicies,0)[1]-self.snake_indicies[0][1]<0:
				tr = 1
			if np.mean(self.snake_indicies,0)[1]-self.snake_indicies[0][1]>0:
				tl = 1
			if np.mean(self.snake_indicies,0)[0]-self.snake_indicies[0][0]>0:
				ta = 1
			if (self.snake_head+np.array([1,0]))[0] not in range(32) or self.world[(self.snake_head+np.array([1,0]))[0]%32][(self.snake_head+np.array([1,0]))[1]%32]==1:				a = 1
			if (self.snake_head+np.array([0,1]))[1] not in range(32) or self.world[(self.snake_head+np.array([0,1]))[0]%32][(self.snake_head+np.array([0,1]))[1]%32]==1:
				b = 1
			if (self.snake_head+np.array([0,-1]))[1] not in range(32) or self.world[(self.snake_head+np.array([0,-1]))[0]%32][(self.snake_head+np.array([0,-1]))[1]%32]==1:
				c = 1
		elif self.snake_indicies[0][0]-self.snake_indicies[1][0] == 0 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == 1:
			if self.apple[1]-self.snake_indicies[0][1]>0:
				apa = 1
			if self.apple[0]-self.snake_indicies[0][0]>0:
				apr = 1
			if self.apple[0]-self.snake_indicies[0][0]<0:
				apl = 1
			if np.mean(self.snake_indicies,0)[1]-self.snake_indicies[0][1]>0:
				ta = 1
			if np.mean(self.snake_indicies,0)[0]-self.snake_indicies[0][0]>0:
				tr = 1
			if np.mean(self.snake_indicies,0)[0]-self.snake_indicies[0][0]<0:
				tl = 1
			if (self.snake_head+np.array([0,1]))[1] not in range(32) or self.world[(self.snake_head+np.array([0,1]))[0]%32][(self.snake_head+np.array([0,1]))[1]%32]==1:				a = 1
			if (self.snake_head+np.array([-1,0]))[0] not in range(32) or self.world[(self.snake_head+np.array([-1,0]))[0]%32][(self.snake_head+np.array([-1,0]))[1]%32]==1:
				b = 1
			if (self.snake_head+np.array([1,0]))[0] not in range(32) or self.world[(self.snake_head+np.array([1,0]))[0]%32][(self.snake_head+np.array([1,0]))[1]%32]==1:
				c = 1
		elif self.snake_indicies[0][0]-self.snake_indicies[1][0] == 0 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == -1:
			if self.apple[1]-self.snake_indicies[0][1]<0:
				apa = 1
			if self.apple[0]-self.snake_indicies[0][0]<0:
				apr = 1
			if self.apple[0]-self.snake_indicies[0][0]>0:
				apl = 1
			if np.mean(self.snake_indicies,0)[1]-self.snake_indicies[0][1]<0:
				ta = 1
			if np.mean(self.snake_indicies,0)[0]-self.snake_indicies[0][0]<0:
				tr = 1
			if np.mean(self.snake_indicies,0)[0]-self.snake_indicies[0][0]>0:
				tl = 1
			if (self.snake_head+np.array([0,-1]))[1] not in range(32) or self.world[(self.snake_head+np.array([0,-1]))[0]%32][(self.snake_head+np.array([0,-1]))[1]%32]==1:
				a = 1
			if (self.snake_head+np.array([1,0]))[0] not in range(32) or self.world[(self.snake_head+np.array([1,0]))[0]%32][(self.snake_head+np.array([1,0]))[1]%32]==1:
				b = 1
			if (self.snake_head+np.array([-1,0]))[0] not in range(32) or self.world[(self.snake_head+np.array([-1,0]))[0]%32][(self.snake_head+np.array([-1,0]))[1]%32]==1:
				c = 1
		
		self.state = np.asarray([a,b,c, apa, apr,apl,ta,tr,tl])

	def update_apple(self):
		while self.world[self.apple[0]][self.apple[1]] > 0 and len(self.snake_indicies)!=32*32:
			self.apple = np.random.randint(0,32,2)
		self.world[self.apple[0]][self.apple[1]] = -10

	def step(self, action):
		self.t_since_a +=1
		alpha = self.alpha
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		self.world[self.snake_head[0]%32][self.snake_head[1]%32] = 1 
		if self.snake_indicies[0][0]-self.snake_indicies[1][0] == -1 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == 0:
			if action == 0:
				self.snake_head+=np.array([-1,0])
			elif action == 1:
				self.snake_head+=np.array([0,-1])
			elif action == 2:
				self.snake_head+=np.array([0,1])
		elif self.snake_indicies[0][0]-self.snake_indicies[1][0] == 1 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == 0:
			if action == 0:
				self.snake_head+=np.array([1,0])
			elif action == 1:
				self.snake_head+=np.array([0,1])
			elif action == 2:
				self.snake_head+=np.array([0,-1])
		elif self.snake_indicies[0][0]-self.snake_indicies[1][0] == 0 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == 1:
			if action == 0:
				self.snake_head+=np.array([0,1])
			elif action == 1:
				self.snake_head+=np.array([-1,0])
			elif action == 2:
				self.snake_head+=np.array([1,0])
		elif self.snake_indicies[0][0]-self.snake_indicies[1][0] == 0 and self.snake_indicies[0][1]-self.snake_indicies[1][1] == -1:
			if action == 0:
				self.snake_head+=np.array([0,-1])
			elif action == 1:
				self.snake_head+=np.array([1,0])
			elif action == 2:
				self.snake_head+=np.array([-1,0])

		#print(self.snake_indicies)
		self.snake_indicies = [(self.snake_head[0],self.snake_head[1])]+self.snake_indicies	
		#print(self.snake_indicies)
		self.snake_tail[0], self.snake_tail[1] = self.snake_indicies.pop(-1)
		#print(self.snake_indicies)
		self.world[self.snake_tail[0]%32][self.snake_tail[1]%32] = 0

		if len(self.snake_indicies)==32*32:
			reward = 500
			self.done = True
		elif self.world[self.snake_head[0]%32][self.snake_head[1]%32]==-10:
			self.snake_indicies = self.snake_indicies+[(self.snake_tail[0],self.snake_tail[1])]
			self.world[self.snake_tail[0]%32][self.snake_tail[1]%32]=1
			self.t_since_a = 0
			self.world[self.snake_head[0]%32][self.snake_head[1]%32]=10
			self.update_apple()
			self.start_dist = np.linalg.norm(self.snake_head-self.apple,1)
			reward = 500
		
		elif self.world[self.snake_head[0]%32][self.snake_head[1]%32] == 1 or self.snake_head[0] not in range(32) or self.snake_head[1] not in range(32):
			self.world[self.snake_head[0]%32][self.snake_head[1]%32]=10
			reward = -100
			self.done = True
		else:
			self.world[self.snake_head[0]%32][self.snake_head[1]%32]=10
			reward = 0
		self.update_state()
		
		return self.state, reward, self.done, {}

	def render(self, mode = 'human',close='false'):
		plt.matshow(100*self.world)
		plt.show()
		
	
	def reset(self):
		self.world = np.zeros((32,32))
		self.snake_head = np.array([5,5])
		self.snake_tail = self.snake_head + np.array([1,0])
		self.snake_indicies = [(self.snake_head[0],self.snake_head[1]),(self.snake_tail[0],self.snake_tail[1])]
		self.world[self.snake_head[0]%32][self.snake_head[1]%32] = 10
		self.world[self.snake_tail[0]%32][self.snake_tail[1]%32] = 1
		self.apple = np.random.randint(0,32,2)
		self.update_apple()
		self.snake_orientation = 0
		self.update_state()
		self.done = False
		self.t_since_a = 0
		self.start_dist = np.linalg.norm(self.snake_head-self.apple,1)
		return self.state

		
		
		
			
					
			
		
		
			
		
		
		