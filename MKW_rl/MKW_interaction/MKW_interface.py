"""
This file attempts to make interfacing with the game easier by containing the important information about the game
within directly callable functions in leiu of the instance manager program handling the static objects
"""

from mkw_scripts.Modules import mkw_utils as mkw_utils
from mkw_scripts.Modules.mkw_classes import vec3 as hookvec3
from mkw_scripts.Modules.mkw_classes import quatf as hookquatf
from mkw_scripts.Modules.mkw_classes import ExactTimer, Timer
from mkw_scripts.Modules.mkw_classes import RaceManager, RaceManagerPlayer, RaceState
from mkw_scripts.Modules.mkw_classes import RaceConfig, RaceConfigScenario, RaceConfigSettings
from mkw_scripts.Modules.mkw_classes import KartObject, KartMove, KartSettings, KartBody
from mkw_scripts.Modules.mkw_classes import VehicleDynamics, VehiclePhysics, KartBoost, KartJump
from mkw_scripts.Modules.mkw_classes import KartState, KartCollide, KartInput, TimerManager

from MKW_rl.MKW_interaction.MKW_data_translate import *

class MKW_Interface():
	def __init__(self):
		"""
		All class objects can be initiated upon save state loading, when initialize_race_objects should be called
		All function calls return the current value, not the location in memory.
		"""
		self.race_mgr = RaceManager
		self.race_mgr_player: RaceManagerPlayer
		self.race_scenario: RaceConfigScenario
		self.race_settings: RaceConfigSettings

		self.kart_object: KartObject
		self.kart_state: KartState
		self.kart_move: KartMove
		self.kart_body: KartBody
		self.kart_boost: KartBoost
		self.kart_collide:KartCollide
		self.kart_jump: KartJump

		self.vehicle_dynamics: VehicleDynamics
		self.vehicle_physics: VehiclePhysics

		"""if self.kart_move.is_bike:
			text += f"Wheelie Length: {self.kart_move.wheelie_frames()}\n"
			text += f"Wheelie CD: {self.kart_move.wheelie_cooldown()} | "
		"""

	def initialize_race_objects(self):
		self.race_mgr = RaceManager()
		self.race_mgr_player = RaceManagerPlayer()
		self.race_scenario = RaceConfigScenario(addr=RaceConfig.race_scenario())
		self.race_settings = RaceConfigSettings(self.race_scenario.settings())

		self.kart_object = KartObject()
		self.kart_settings = KartSettings(addr=self.kart_object.kart_settings())
		self.kart_state = KartState(addr=self.kart_object.kart_state())
		self.kart_move = KartMove(addr=self.kart_object.kart_move())
		self.kart_body = KartBody(addr=self.kart_object.kart_body())
		self.kart_boost = KartBoost(addr=self.kart_move.kart_boost())
		self.kart_collide = KartCollide(addr=self.kart_object.kart_collide())
		self.kart_jump = KartJump(addr=self.kart_move.kart_jump())

		self.vehicle_dynamics = VehicleDynamics(addr=self.kart_body.vehicle_dynamics())
		self.vehicle_physics = VehiclePhysics(addr=self.vehicle_dynamics.vehicle_physics())
		self.timer_manager = TimerManager(addr=self.race_mgr.timer_manager())
		self.timer = Timer(self.timer_manager.timer())

	def get_start_boost_charge(self):
		return self.kart_state.start_boost_charge()
	
	def get_trickable_timer(self):
		return self.kart_state.trickable_timer()

	def get_kart_position_and_rotation(self):
		return {
			"position": self.vehicle_physics.position(),
			"rotation": self.kart_body.kart_part_rotation(),
			"angle": self.kart_body.angle()
		}
	
	def get_kart_velocities(self):
		return {
			"external_velocity": self.vehicle_physics.external_velocity(),
			"internal_velocity": self.vehicle_physics.internal_velocity(),
			"moving_road_velocity": self.vehicle_physics.moving_road_velocity(),
			"moving_water_velocity": self.vehicle_physics.moving_water_velocity()
		}

	def get_surface_properties(self):
		
		surface_properties = self.kart_collide.surface_properties()

		# Decode surface properties into a list of booleans
		is_wall = (surface_properties.value & SurfaceProperties.WALL) > 0
		is_solid_oob = (surface_properties.value & SurfaceProperties.SOLID_OOB) > 0
		is_boost_ramp = (surface_properties.value & SurfaceProperties.BOOST_RAMP) > 0
		is_offroad = (surface_properties.value & SurfaceProperties.OFFROAD) > 0
		is_boost_panel_or_ramp = (surface_properties.value & SurfaceProperties.BOOST_PANEL_OR_RAMP) > 0
		is_trickable = (surface_properties.value & SurfaceProperties.TRICKABLE) > 0
		"""
		WALL = 0x1
		SOLID_OOB = 0x2
		BOOST_RAMP = 0x10
		OFFROAD = 0x40
		BOOST_PANEL_OR_RAMP = 0x100
		TRICKABLE = 0x800
		"""
		return [is_wall, is_solid_oob, is_boost_ramp, is_offroad, is_boost_panel_or_ramp, is_trickable]
	
	def get_checkpoint_data(self):
		return {
			"lap_completion": self.race_mgr_player.lap_completion(),
			"race_completion": self.race_mgr_player.race_completion(),
			"race_completion_max": self.race_mgr_player.race_completion_max(),
			"checkpoint_id": self.race_mgr_player.checkpoint_id(),
			"current_key_checkpoint": self.race_mgr_player.current_kcp(),
			"max_key_checkpoint": self.race_mgr_player.max_kcp()
		}
	
	def get_respawn_point(self):
		return self.race_mgr_player.respawn()
	
	def get_driving_direction(self):
		"""
		DrivingDirection Enum:
		FORWARDS = 0
		BRAKING = 1
		WAITING_FOR_BACKWARDS = 2
		BACKWARDS = 3
		"""
		return self.kart_move.driving_direction().value # 1 and 3 likely deserve negative rewards
	
	def get_wheelie_cooldown(self):
		if self.kart_move.is_bike: # sanity check in case of dumb programmers (me)
			return self.kart_move.wheelie_cooldown()
		print("ERROR: Requested wheelie cooldown on a kart")
		return 0
	
	def get_trick_cooldown(self):
		return self.kart_jump.cooldown()
	
	def get_airtime(self):
		return self.kart_move.airtime()

	def get_glitchy_timers(self):
		# Likely to never be useful to the ai unless forced to find ultras
		return {
			"horizontal_wall_glitch_timer": self.kart_state.hwg_timer(),
			"solid_oob_timer": self.kart_collide.solid_oob_timer(),
			"offroad_invincibility": self.kart_move.offroad_invincibility()
		}
	
	def get_item_count(self):
		# Thanks to vαbol∂ and Blounard for helping locate and pythonically access item information addresses
		return mkw_utils.chase_pointer(0x809c3618, [0x14, 0x00*0x248 + 0x90], 's32')
	
	def get_item_type(self):
		# Thanks to vαbol∂ and Blounard for helping locate and pythonically access item information addresses
		return mkw_utils.chase_pointer(0x809c3618, [0x14, 0x00*0x248 + 0x8C], 's32')
	
	def convert_vec3(self, vector: hookvec3):
		return vec3(vector.x, vector.y, vector.z)
	
	def convert_quatf(self, valueS: hookquatf):
		return quatf(valueS.x, valueS.y, valueS.z, valueS.w)
	
	def get_boost_states(self) -> Boosts:
		boosts = Boosts()
		boosts["mt_charge"] = self.kart_move.mt_charge()
		boosts["smt_charge"] = self.kart_move.smt_charge()
		boosts["ssmt_charge"] = self.kart_move.ssmt_charge()
		boosts["mt_boost"] = self.kart_boost.all_mt_timer()
		boosts["trick_boost"] = self.kart_boost.trick_and_zipper_timer()
		boosts["shroom_boost"] = self.kart_boost.mushroom_and_boost_panel_timer()
		return boosts
	
	def get_kart_data(self) -> Kart_Data:
		kart_data = Kart_Data()
		kart_data["character"] = self.kart_settings.character().value
		kart_data["vehicle"] = self.kart_settings.vehicle().value
		kart_data["position"] = self.convert_vec3(self.vehicle_physics.position())
		kart_data["rotation"] = self.convert_quatf(self.vehicle_physics.main_rotation())
		kart_data["speed"] = self.vehicle_physics.speed_norm()
		kart_data["external_velocity"] = self.convert_vec3(self.vehicle_physics.external_velocity())
		kart_data["internal_velocity"] = self.convert_vec3(self.vehicle_physics.internal_velocity())
		kart_data["moving_road_velocity"] = self.convert_vec3(self.vehicle_physics.moving_road_velocity())
		kart_data["moving_water_velocity"] = self.convert_vec3(self.vehicle_physics.moving_water_velocity())
		if self.kart_move.is_bike:
			kart_data["wheelie_cooldown"] = self.kart_move.wheelie_cooldown()
		else:
			kart_data["wheelie_cooldown"] = 0
		kart_data["trick_cooldown"] = self.kart_jump.cooldown()
		kart_data["respawn_timer"] = self.kart_collide.time_before_respawn() # go figure, there's two different timers for the respawn. A waiting period, and a continuation period.
		return kart_data

	def get_race_data(self) -> Race_Data:
		race_data = Race_Data()
		race_data["lap_completion"] = self.race_mgr_player.lap_completion()
		race_data["race_completion"] = self.race_mgr_player.race_completion()
		race_data["race_completion_max"] = self.race_mgr_player.race_completion_max()
		race_data["checkpoint_id"] = self.race_mgr_player.checkpoint_id()
		race_data["current_key_checkpoint"] = self.race_mgr_player.current_kcp()
		race_data["max_key_checkpoint"] = self.race_mgr_player.max_kcp()
		# race_data["respawn_point"] = self.get_respawn_point()
		race_data["driving_direction"] = self.get_driving_direction()
		race_data["item_count"] = self.get_item_count()

		if self.race_mgr_player.race_completion_max() >= 4: # race finished
			self.timer = self.race_mgr_player.inst_race_finish_time()
		race_data["race_time"] = self.timer.minutes() * 60 + self.timer.seconds() + self.timer.milliseconds() / 1000
		race_data["state"] = self.race_mgr.state().value

		# race_data["item_type"] = self.get_item_type()
		return race_data

	def get_game_data_object(self) -> Game_Data:
		# Note that the order here and within the called functions is important, as that is the order in which they are flattened.
		# Thus, MKW_data_translate must match the ordering, especially in the float_input_mean and float_input_std variables.
		# If I find a way to remove those variables I will remove them.
		game_data = Game_Data()
		game_data["boost_data"] = self.get_boost_states()
		game_data["kart_data"] = self.get_kart_data()
		game_data["race_data"] = self.get_race_data()
		game_data["start_boost_charge"] = self.get_start_boost_charge()
		game_data["trickable_timer"] = self.get_trickable_timer()
		game_data["surface_properties"] = self.get_surface_properties()
		game_data["airtime"] = self.get_airtime()
		return game_data
