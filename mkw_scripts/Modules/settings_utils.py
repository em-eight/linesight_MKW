from config_files import user_config
from .mkw_classes import KartInput
from .mkw_classes import RaceManagerPlayer, RaceInputState
import configparser
import os


################# INFO DISPLAY CONFIG ########################
class InfoDisplayConfigInstance():
    def __init__(self, config : configparser.ConfigParser):
        self.debug = config['DEBUG'].getboolean('Debug')
        self.frame_count = config['INFO DISPLAY'].getboolean('Frame Count')
        self.rkg_buffer_size = config['INFO DISPLAY'].getboolean('RKG Buffer Size')
        self.lap_splits = config['INFO DISPLAY'].getboolean('Lap Splits')
        self.speed = config['INFO DISPLAY'].getboolean('Speed')
        self.speed_oriented = config['INFO DISPLAY'].getboolean('Oriented Speed')
        self.iv = config['INFO DISPLAY'].getboolean('Internal Velocity (X, Y, Z)')
        self.iv_oriented = config['INFO DISPLAY'].getboolean('Oriented Internal Velocity')
        self.iv_xyz = config['INFO DISPLAY'].getboolean('Internal Velocity (XYZ)')
        self.ev = config['INFO DISPLAY'].getboolean('External Velocity (X, Y, Z)')
        self.ev_oriented = config['INFO DISPLAY'].getboolean('Oriented External Velocity')
        self.ev_xyz = config['INFO DISPLAY'].getboolean('External Velocity (XYZ)')
        self.mrv = config['INFO DISPLAY'].getboolean('Moving Road Velocity (X, Y, Z)')
        self.mrv_oriented = config['INFO DISPLAY'].getboolean('Oriented Moving Road Velocity')
        self.mrv_xyz = config['INFO DISPLAY'].getboolean('Moving Road Velocity (XYZ)')
        self.mwv = config['INFO DISPLAY'].getboolean('Moving Water Velocity (X, Y, Z)')
        self.mwv_oriented = config['INFO DISPLAY'].getboolean('Oriented Moving Water Velocity')
        self.mwv_xyz = config['INFO DISPLAY'].getboolean('Moving Water Velocity (XYZ)')
        self.charges = config['INFO DISPLAY'].getboolean('Charges and Boosts')
        self.cps = config['INFO DISPLAY'].getboolean('Checkpoints and Completion')
        self.air = config['INFO DISPLAY'].getboolean('Airtime')
        self.misc = config['INFO DISPLAY'].getboolean('Miscellaneous')
        self.surfaces = config['INFO DISPLAY'].getboolean('Surface Properties')
        self.position = config['INFO DISPLAY'].getboolean('Position')
        self.rotation = config['INFO DISPLAY'].getboolean('Rotation')
        self.dpg = config['INFO DISPLAY'].getboolean('Distance Player-Ghost (X, Y, Z)')
        self.dpg_oriented = config['INFO DISPLAY'].getboolean('Oriented Distance Player-Ghost')
        self.dpg_xyz = config['INFO DISPLAY'].getboolean('Distance Player-Ghost (XYZ)')
        self.vd_spd = config['INFO DISPLAY'].getboolean('Player-Ghost Speed diff')
        self.vd_iv = config['INFO DISPLAY'].getboolean('Player-Ghost IV diff')
        self.vd_ev = config['INFO DISPLAY'].getboolean('Player-Ghost EV diff')
        self.rd_pitch = config['INFO DISPLAY'].getboolean('Player-Ghost Pitch diff')
        self.rd_yaw = config['INFO DISPLAY'].getboolean('Player-Ghost Facing Yaw diff')
        self.rd_movy = config['INFO DISPLAY'].getboolean('Player-Ghost Moving Yaw diff')
        self.rd_roll = config['INFO DISPLAY'].getboolean('Player-Ghost Roll diff')
        self.td_absolute = config['INFO DISPLAY'].getboolean('TimeDiff Absolute')
        self.td_relative = config['INFO DISPLAY'].getboolean('TimeDiff Relative')
        self.td_projected = config['INFO DISPLAY'].getboolean('TimeDiff Projected')
        self.td_crosspath = config['INFO DISPLAY'].getboolean('TimeDiff CrossPath')
        self.td_tofinish = config['INFO DISPLAY'].getboolean('TimeDiff ToFinish')
        self.td_racecomp = config['INFO DISPLAY'].getboolean('TimeDiff RaceComp')
        self.td_set = config['INFO DISPLAY']['TimeDiff Setting']
        self.td = self.td_absolute or self.td_relative or self.td_projected or self.td_crosspath or self.td_tofinish or self.td_racecomp
        self.stick = config['INFO DISPLAY'].getboolean('Stick')
        self.color = int(config['INFO DISPLAY']['Text Color (ARGB)'], 16)
        self.digits = min(7, config['INFO DISPLAY'].getint('Digits (to round to)'))
        self.history_size = config['INFO DISPLAY'].getint('History Size')

    def write_to_file(self):
        config = configparser.ConfigParser()
        config['DEBUG'] = {}
        config['DEBUG']['Debug'] = str(self.debug)
        
        config['INFO DISPLAY'] = {}
        config['INFO DISPLAY']["Frame Count"] = str(self.frame_count)
        config['INFO DISPLAY']["RKG Buffer Size"] = str(self.rkg_buffer_size)
        config['INFO DISPLAY']["Lap Splits"] = str(self.lap_splits)
        config['INFO DISPLAY']["Speed"] = str(self.speed)
        config['INFO DISPLAY']["Oriented Speed"] = str(self.speed_oriented)
        config['INFO DISPLAY']["Internal Velocity (X, Y, Z)"] = str(self.iv)
        config['INFO DISPLAY']["Oriented Internal Velocity"] = str(self.iv_oriented)
        config['INFO DISPLAY']["Internal Velocity (XYZ)"] = str(self.iv_xyz)
        config['INFO DISPLAY']["External Velocity (X, Y, Z)"] = str(self.ev)
        config['INFO DISPLAY']["Oriented External Velocity"] = str(self.ev_oriented)
        config['INFO DISPLAY']["External Velocity (XYZ)"] = str(self.ev_xyz)
        config['INFO DISPLAY']["Moving Road Velocity (X, Y, Z)"] = str(self.mrv)
        config['INFO DISPLAY']["Oriented Moving Road Velocity"] = str(self.mrv_oriented)
        config['INFO DISPLAY']["Moving Road Velocity (XYZ)"] = str(self.mrv_xyz)
        config['INFO DISPLAY']["Moving Water Velocity (X, Y, Z)"] = str(self.mwv)
        config['INFO DISPLAY']["Oriented Moving Water Velocity"] = str(self.mwv_oriented)
        config['INFO DISPLAY']["Moving Water Velocity (XYZ)"] = str(self.mwv_xyz)
        config['INFO DISPLAY']["Charges and Boosts"] = str(self.charges)
        config['INFO DISPLAY']["Checkpoints and Completion"] = str(self.cps)
        config['INFO DISPLAY']["Airtime"] = str(self.air)
        config['INFO DISPLAY']["Miscellaneous"] = str(self.misc)
        config['INFO DISPLAY']["Surface Properties"] = str(self.surfaces)
        config['INFO DISPLAY']["Position"] = str(self.position)
        config['INFO DISPLAY']["Rotation"] = str(self.rotation)        
        config['INFO DISPLAY']["Distance Player-Ghost (X, Y, Z)"] = str(self.dpg)
        config['INFO DISPLAY']["Oriented Distance Player-Ghost"] = str(self.dpg_oriented)
        config['INFO DISPLAY']["Distance Player-Ghost (XYZ)"] = str(self.dpg_xyz)
        config['INFO DISPLAY']['Player-Ghost Speed diff'] = str(self.vd_spd)
        config['INFO DISPLAY']['Player-Ghost IV diff'] = str(self.vd_iv)
        config['INFO DISPLAY']['Player-Ghost EV diff'] = str(self.vd_ev)
        config['INFO DISPLAY']['Player-Ghost Pitch diff'] = str(self.rd_pitch)
        config['INFO DISPLAY']['Player-Ghost Facing Yaw diff'] = str(self.rd_yaw)
        config['INFO DISPLAY']['Player-Ghost Moving Yaw diff'] = str(self.rd_movy)
        config['INFO DISPLAY']['Player-Ghost Roll diff'] = str(self.rd_roll)
        config['INFO DISPLAY']["TimeDiff Absolute"] = str(self.td_absolute)
        config['INFO DISPLAY']["TimeDiff Relative"] = str(self.td_relative)
        config['INFO DISPLAY']["TimeDiff Projected"] = str(self.td_projected)
        config['INFO DISPLAY']["TimeDiff CrossPath"] = str(self.td_crosspath)
        config['INFO DISPLAY']["TimeDiff ToFinish"] = str(self.td_tofinish)
        config['INFO DISPLAY']["TimeDiff RaceComp"] = str(self.td_racecomp)
        config['INFO DISPLAY']["TimeDiff Setting"] = self.td_set
        config['INFO DISPLAY']["Stick"] = str(self.stick)
        config['INFO DISPLAY']["Text Color (ARGB)"] = str(self.color)
        config['INFO DISPLAY']["Digits (to round to)"] = str(self.digits)
        config['INFO DISPLAY']["History Size"] = str(self.history_size)
        
        file_path = os.path.join(user_config.project_scripts_path, 'Settings', 'Infodisplay.ini')
        with open(file_path, 'w') as f:
            config.write(f)

def populate_default_config_infodisplay(file_path):
    config = configparser.ConfigParser()
    
    config['DEBUG'] = {}
    config['DEBUG']['Debug'] = "False"
    
    config['INFO DISPLAY'] = {}
    config['INFO DISPLAY']["Frame Count"] = "True"
    config['INFO DISPLAY']["RKG Buffer Size"] = "False"
    config['INFO DISPLAY']["Lap Splits"] = "False"
    config['INFO DISPLAY']["Speed"] = "True"
    config['INFO DISPLAY']["Oriented Speed"] = "False"
    config['INFO DISPLAY']["Internal Velocity (X, Y, Z)"] = "False"
    config['INFO DISPLAY']["Oriented Internal Velocity"] = "False"
    config['INFO DISPLAY']["Internal Velocity (XYZ)"] = "False"
    config['INFO DISPLAY']["External Velocity (X, Y, Z)"] = "False"
    config['INFO DISPLAY']["Oriented External Velocity"] = "False"
    config['INFO DISPLAY']["External Velocity (XYZ)"] = "True"
    config['INFO DISPLAY']["Moving Road Velocity (X, Y, Z)"] = "False"
    config['INFO DISPLAY']["Oriented Moving Road Velocity"] = "False"
    config['INFO DISPLAY']["Moving Road Velocity (XYZ)"] = "False"
    config['INFO DISPLAY']["Moving Water Velocity (X, Y, Z)"] = "False"
    config['INFO DISPLAY']["Oriented Moving Water Velocity"] = "False"
    config['INFO DISPLAY']["Moving Water Velocity (XYZ)"] = "False"
    config['INFO DISPLAY']["Charges and Boosts"] = "True"
    config['INFO DISPLAY']["Checkpoints and Completion"] = "True"
    config['INFO DISPLAY']["Airtime"] = "True"
    config['INFO DISPLAY']["Miscellaneous"] = "False"
    config['INFO DISPLAY']["Surface Properties"] = "False"
    config['INFO DISPLAY']["Position"] = "False"
    config['INFO DISPLAY']["Rotation"] = "True"
    config['INFO DISPLAY']["Distance Player-Ghost (X, Y, Z)"] = "False"
    config['INFO DISPLAY']["Oriented Distance Player-Ghost"] = "False"
    config['INFO DISPLAY']["Distance Player-Ghost (XYZ)"] = "False"
    config['INFO DISPLAY']['Player-Ghost Speed diff'] = "False"
    config['INFO DISPLAY']['Player-Ghost IV diff'] = "False"
    config['INFO DISPLAY']['Player-Ghost EV diff'] = "False"
    config['INFO DISPLAY']['Player-Ghost Pitch diff'] = "False"
    config['INFO DISPLAY']['Player-Ghost Facing Yaw diff'] = "False"
    config['INFO DISPLAY']['Player-Ghost Moving Yaw diff'] = "False"
    config['INFO DISPLAY']['Player-Ghost Roll diff'] = "False"
    config['INFO DISPLAY']["TimeDiff Absolute"] = "False"
    config['INFO DISPLAY']["TimeDiff Relative"] = "False"
    config['INFO DISPLAY']["TimeDiff Projected"] = "True"
    config['INFO DISPLAY']["TimeDiff CrossPath"] = "False"
    config['INFO DISPLAY']["TimeDiff ToFinish"] = "True"
    config['INFO DISPLAY']["TimeDiff RaceComp"] = "True"
    config['INFO DISPLAY']["TimeDiff Setting"] = "behind"
    config['INFO DISPLAY']["Stick"] = "True"
    config['INFO DISPLAY']["Text Color (ARGB)"] = "0xFFFFFFFF"
    config['INFO DISPLAY']["Digits (to round to)"] = "6"
    config['INFO DISPLAY']["History Size"] = "200"
    
    
    with open(file_path, 'w') as f:
        config.write(f)
        
    return config

def get_infodisplay_config():
    config = configparser.ConfigParser()
    file_path = os.path.join(user_config.project_scripts_path, 'Settings', 'Infodisplay.ini')
    config.read(file_path)
    if not config.sections():
        config = populate_default_config_infodisplay(file_path)
    return InfoDisplayConfigInstance(config)


################## AGC CONFIG ########################
class AGCConfigInstance():
    def __init__(self, config : configparser.ConfigParser):
        self.useFrames = config['DELAY'].getboolean('Delay unit in frame')
        self.ghost_delay = eval(config['DELAY']['Ghost delay'])
        self.player_delay = config['DELAY'].getfloat('Player delay')
        self.ghost_path = config['PATH'].get('Ghost .agc file path')
        self.player_path = config['PATH'].get('Player .agc file path')

def populate_default_config_agc(file_path):
    config = configparser.ConfigParser()
    
    config['DELAY'] = {}
    config['DELAY']['Delay unit in frame'] = "True"
    config['DELAY']['Ghost delay'] = "0"
    config['DELAY']['Player delay'] = "0"
    
    config['PATH'] = {}
    config['PATH']['Ghost .agc file path'] = "AGC_Data/ghost_data.agc"
    config['PATH']['Player .agc file path'] = "AGC_Data/player_data.agc"

    with open(file_path, 'w') as f:
        config.write(f)
        
    return config

def get_agc_config():
    config = configparser.ConfigParser()
    file_path = os.path.join(user_config.project_scripts_path, 'Settings', 'AGC.ini')
    config.read(file_path)
    if not config.sections():
        config = populate_default_config_agc(file_path)
    return AGCConfigInstance(config)


################## TTK CONFIG ########################
class TTKConfigInstance():
    def __init__(self, config : configparser.ConfigParser):
        #self.ttk_path = config['PATH'].get('TTK folder path')
        self.player_filename = config['PATH'].get('Player filename')
        self.ghost_filename = config['PATH'].get('Ghost filename')
        self.track_suffix = config['PATH'].getboolean('Use track suffix')
        self.ttk_backup = config['BACKUP'].getint('Backup Amount')

def populate_default_config_ttk(file_path):
    config = configparser.ConfigParser()

    config['PATH'] = {}
    #config['PATH']['TTK folder path'] = "MKW_Inputs/"
    config['PATH']['Player filename'] = "TTK_Player_Inputs"
    config['PATH']['Ghost filename'] = "TTK_Ghost_Inputs"
    config['PATH']['Use track suffix'] = 'True'

    config['BACKUP'] = {}
    config['BACKUP']['Backup Amount'] = '5'

    with open(file_path, 'w') as f:
        config.write(f)
        
    return config

def get_ttk_config():
    config = configparser.ConfigParser()
    file_path = os.path.join(user_config.project_scripts_path, 'Settings', 'TTK.ini')
    config.read(file_path)
    if not config.sections():
        config = populate_default_config_ttk(file_path)
    return TTKConfigInstance(config)

