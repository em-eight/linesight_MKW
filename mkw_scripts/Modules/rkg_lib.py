from dolphin import gui, memory # type: ignore
from math import floor
import os
import zlib

from mkw_scripts.Modules.mkw_classes import common
from .mkw_classes import RaceConfig, RaceConfigScenario, RaceConfigSettings
from .mkw_classes import RaceConfigPlayer, PlayerInput, KartInput, Controller
from .framesequence import Frame, FrameSequence
from .mkw_utils import chase_pointer
from .src import decompress, compress
from mkw_scripts.Modules import mkw_classes, mkw_config

def extract_bits(data, start_bit, length):
    byte_index = start_bit // 8
    bit_offset = start_bit % 8
    bits = 0

    for i in range(length):
        bit_position = bit_offset + i
        bits <<= 1
        bits |= (data[byte_index + (bit_position // 8)] >> (7 - (bit_position % 8))) & 1

    return bits

def crc16(i):
    '''Argument : bytearray
        Return : 16bit int'''
    #https://stackoverflow.com/questions/25239423/crc-ccitt-16-bit-python-manual-calculation
    def _update_crc(crc, c):
        def _initial(c):
            POLYNOMIAL = 0x1021
            PRESET = 0
            crc = 0
            c = c << 8
            for j in range(8):
                if (crc ^ c) & 0x8000:
                    crc = (crc << 1) ^ POLYNOMIAL
                else:
                    crc = crc << 1
                c = c << 1
            return crc
        POLYNOMIAL = 0x1021
        PRESET = 0
        _tab = [ _initial(i) for i in range(256) ]
        cc = 0xff & c
        tmp = (crc >> 8) ^ cc
        crc = (crc << 8) ^ _tab[tmp & 0xff]
        crc = crc & 0xffff
        return crc
    POLYNOMIAL = 0x1021
    PRESET = 0
    crc = PRESET
    for c in i:
        crc = _update_crc(crc, c)
    return crc


def convertTimer(bits: int):
    '''Convert a 24 bits int representing a Timer (m,s,ms), to a float in seconds'''
    ms = bits & 1023
    bits >>= 10
    s = bits & 127
    bits >>= 7
    m = bits & 127
    return m*60+s+ms/1000

def convertTimerBack(split: float):
    '''Convert a float in seconds, to a 24 bits int representing a Timer (m,s,ms)'''
    m = floor(split/60)
    s = floor(split%60)
    ms = floor(split*1000)%1000
    return ms + (s<<10) + (m <<17)

def add_bits(bit_list, number, size):
    ''' Add to a bit list another number, with a bit size'''
    for i in range(size):
        bit_list.append((number >> (size - i -1)) & 1)

def bits_to_bytearray(bit_list):
    ''' Convert an array of bits to an array of bytes (grouping bits by 8)'''
    b = bytearray()
    byte = 0
    for i in range(len(bit_list)):
        byte = (byte << 1) + bit_list[i]
        if i%8 == 7:
            b.append(byte)
            byte = 0
    if byte != 0:
        b.append(byte)
    return b

class RKGMetaData:
    def __init__(self, rkg_data, useDefault = False):
        if not useDefault:
            self.finish_time = convertTimer(extract_bits(rkg_data, 4*8+0, 24))
            self.track_id = extract_bits(rkg_data, 7*8+0, 6)
            self.character_id = extract_bits(rkg_data, 8*8+6, 6)
            self.vehicle_id = extract_bits(rkg_data, 8*8+0, 6)
            self.year = extract_bits(rkg_data, 9*8+4, 7)
            self.month = extract_bits(rkg_data, 0xA*8+3, 4)
            self.day = extract_bits(rkg_data, 0xA*8+7, 5)
            self.controller_id = extract_bits(rkg_data, 0xB*8+4, 4)
            self.compressed_flag = extract_bits(rkg_data, 0xC*8+4, 1)
            self.ghost_type = extract_bits(rkg_data, 0xC*8+7, 7)
            self.drift_id = extract_bits(rkg_data, 0xD*8+6, 1)
            self.input_data_length = extract_bits(rkg_data, 0xE*8+0, 16)
            self.lap_count = extract_bits(rkg_data, 0x10*8+0, 8)
            self.lap1_split = convertTimer(extract_bits(rkg_data, 0x11*8+0, 24))
            self.lap2_split = convertTimer(extract_bits(rkg_data, 0x14*8+0, 24))
            self.lap3_split = convertTimer(extract_bits(rkg_data, 0x17*8+0, 24))
            self.country_code = extract_bits(rkg_data, 0x34*8+0, 8)
            self.state_code = extract_bits(rkg_data, 0x35*8+0, 8)
            self.location_code = extract_bits(rkg_data, 0x36*8+0, 16)
        else:
            self.finish_time = 100
            self.track_id = 0
            self.character_id = 0
            self.vehicle_id = 0
            self.year = 25
            self.month = 1
            self.day = 1
            self.controller_id = 3
            self.compressed_flag = 1
            self.ghost_type = 38
            self.drift_id = 0
            self.input_data_length = 908
            self.lap_count = 3
            self.lap1_split = 0
            self.lap2_split = 5999.999
            self.lap3_split = 5999.999
            self.country_code = 255
            self.state_code = 255
            self.location_code = 65535

    def __iter__(self):
        return iter([self.finish_time,
            self.track_id,
            self.character_id,
            self.vehicle_id,
            self.year,
            self.month,
            self.day,
            self.controller_id,
            self.compressed_flag,
            self.ghost_type,
            self.drift_id,
            self.input_data_length,
            self.lap_count,
            self.lap1_split,
            self.lap2_split,
            self.lap3_split,
            self.country_code,
            self.state_code,
            self.location_code])

    def __str__(self):
        res = ""
        res += "finish_time = " + str(self.finish_time)+"\n"
        res += "track_id = " + str(self.track_id)+"\n"
        res += "character_id = " + str(self.character_id)+"\n"
        res += "vehicle_id = " + str(self.vehicle_id)+"\n"
        res += "year = " + str(self.year)+"\n"
        res += "month = " + str(self.month)+"\n"
        res += "day = " + str(self.day)+"\n"
        res += "controller_id = " + str(self.controller_id)+"\n"
        res += "compressed_flag = " + str(self.compressed_flag)+"\n"
        res += "ghost_type = " + str(self.ghost_type)+"\n"
        res += "drift_id = " + str(self.drift_id)+"\n"
        res += "input_data_length = " + str(self.input_data_length)+"\n"
        res += "lap_count = " + str(self.lap_count)+"\n"
        res += "lap1_split = " + str(self.lap1_split)+"\n"
        res += "lap2_split = " + str(self.lap2_split)+"\n"
        res += "lap3_split = " + str(self.lap3_split)+"\n"
        res += "country_code = " + str(self.country_code)+"\n"
        res += "state_code = " + str(self.state_code)+"\n"
        res += "location_code = " + str(self.location_code)
        return res
    
    @staticmethod
    def from_string(string):
        lines = string.split("\n")
        self = RKGMetaData(None, True)
        self.finish_time = float(lines[0].split('=')[1])
        self.track_id = int(lines[1].split('=')[1])
        self.character_id = int(lines[2].split('=')[1])
        self.vehicle_id = int(lines[3].split('=')[1])
        self.year = int(lines[4].split('=')[1])
        self.month = int(lines[5].split('=')[1])
        self.day = int(lines[6].split('=')[1])
        self.controller_id = int(lines[7].split('=')[1])
        self.compressed_flag = int(lines[8].split('=')[1])
        self.ghost_type = int(lines[9].split('=')[1])
        self.drift_id = int(lines[10].split('=')[1])
        self.input_data_length = int(lines[11].split('=')[1])
        self.lap_count = int(lines[12].split('=')[1])
        self.lap1_split = float(lines[13].split('=')[1])
        self.lap2_split = float(lines[14].split('=')[1])
        self.lap3_split = float(lines[15].split('=')[1])
        self.country_code = int(lines[16].split('=')[1])
        self.state_code = int(lines[17].split('=')[1])
        self.location_code = int(lines[18].split('=')[1])
        return self

    def to_bytes(self):
        bit_list = []
        add_bits(bit_list, convertTimerBack(self.finish_time), 24)
        add_bits(bit_list, self.track_id, 6)
        add_bits(bit_list, 0, 2) 
        add_bits(bit_list, self.vehicle_id, 6)
        add_bits(bit_list, self.character_id, 6)
        add_bits(bit_list, self.year, 7)
        add_bits(bit_list, self.month, 4)
        add_bits(bit_list, self.day, 5)
        add_bits(bit_list, self.controller_id, 4)
        add_bits(bit_list, 0, 4)
        add_bits(bit_list, self.compressed_flag, 1)
        add_bits(bit_list, 0, 2)
        add_bits(bit_list, self.ghost_type, 7)
        add_bits(bit_list, self.drift_id, 1)
        add_bits(bit_list, 0, 1)
        add_bits(bit_list, self.input_data_length, 8*2)
        add_bits(bit_list, self.lap_count, 8*1)
        add_bits(bit_list, convertTimerBack(self.lap1_split), 24)
        add_bits(bit_list, convertTimerBack(self.lap2_split), 24)
        add_bits(bit_list, convertTimerBack(self.lap3_split), 24)
        add_bits(bit_list, 0, 2*3*8)
        add_bits(bit_list, 0, 8*0x14)
        add_bits(bit_list, self.country_code, 8)
        add_bits(bit_list, self.state_code, 8)
        add_bits(bit_list, self.location_code, 8*2)
        add_bits(bit_list, 0, 8*4)
    
        return bytearray(b'RKGD') + bits_to_bytearray(bit_list)
    
    @staticmethod
    def from_current_race():
        metadata = RKGMetaData(None, True)
        race_config_scenario = RaceConfigScenario(addr=RaceConfig.race_scenario())
        race_config_settings = RaceConfigSettings(addr=race_config_scenario.settings())
        race_config_player = RaceConfigPlayer(addr=race_config_scenario.player())
        player_input = PlayerInput(player_idx=0)
        kart_input = KartInput(player_input.kart_input())
        controller = Controller(addr=kart_input.race_controller())
        
        metadata.track_id = race_config_settings.course_id().value
        metadata.vehicle_id = race_config_player.vehicle_id().value
        metadata.character_id = race_config_player.character_id().value
        metadata.drift_id = int(controller.drift_is_auto())

        return metadata


def decompress_ghost_input(ghost_src):
    if len(ghost_src) > 0x8F and ghost_src[0x8C:0x90] == b'Yaz1':
        uncompressed_size = (ghost_src[0x90] << 24) + (ghost_src[0x91] << 16) + (ghost_src[0x92] << 8) + ghost_src[0x93]
        return decode_Yaz1(ghost_src, 0x9c, uncompressed_size)
    else:
        return list(ghost_src[0x88:])

def decode_Yaz1(src, offset, uncompressed_size):
    src_pos = offset
    valid_bit_count = 0
    try:
        curr_code_byte = src[offset + src_pos]
    except:
         curr_code_byte = src[src_pos]

    dst = []
    
    while len(dst) < uncompressed_size:
        if valid_bit_count == 0:
            curr_code_byte = src[src_pos]
            src_pos += 1
            valid_bit_count = 8

        if (curr_code_byte & 0x80) != 0:
            dst.append(src[src_pos])
            src_pos += 1
        else:
            byte1 = src[src_pos]
            byte2 = src[src_pos + 1]
            src_pos += 2
            dist = ((byte1 & 0xF) << 8) | byte2
            copy_source = len(dst) - (dist + 1)
            num_bytes = byte1 >> 4
            if num_bytes == 0:
                num_bytes = src[src_pos] + 0x12
                src_pos += 1
            else:
                num_bytes += 2

            for _ in range(num_bytes):
                dst.append(dst[copy_source])
                copy_source += 1

        curr_code_byte <<= 1
        valid_bit_count -= 1

    return dst


def decode_rkg_inputs(rkg_data):
    raw_data = decompress_ghost_input(rkg_data)
    button_inputs = []
    analog_inputs = []
    trick_inputs = []
    cur_byte = 8
    nr_button_inputs = (raw_data[0] << 8) | raw_data[1]
    nr_analog_inputs = (raw_data[2] << 8) | raw_data[3]
    nr_trick_inputs = (raw_data[4] << 8) | raw_data[5]

    for _ in range(nr_button_inputs):
        if cur_byte + 1 < len(raw_data):
            inputs = raw_data[cur_byte]
            frames = raw_data[cur_byte + 1]
        else:
            inputs = 0
            frames = 0
        accelerator = inputs & 0x1
        drift = (inputs & 0x2) >> 1
        item = (inputs & 0x4) >> 2
        pseudoAB = (inputs & 0x8) >> 3
        breakdrift = (inputs & 0x10) >> 4
        button_inputs += [(accelerator, drift, item, pseudoAB, breakdrift)] * frames
        cur_byte += 2

    for _ in range(nr_analog_inputs):
        if cur_byte + 1 < len(raw_data):
            inputs = raw_data[cur_byte]
            frames = raw_data[cur_byte + 1]
        else:
            inputs = 0
            frames = 0
        horizontal = ((inputs >> 4) & 0xF) - 7
        vertical = (inputs & 0xF) - 7
        analog_inputs += [(horizontal, vertical)] * frames
        cur_byte += 2

    for _ in range(nr_trick_inputs):
        if cur_byte + 1 < len(raw_data):
            inputs = raw_data[cur_byte]
            frames = raw_data[cur_byte + 1]
        else:
            inputs = 0
            frames = 0
        trick = (inputs & 0x70) >> 4
        extra_frames = (inputs & 0x0F) << 8
        trick_inputs += [trick] * (frames + extra_frames)
        cur_byte += 2
    inputList = []
    for i in range(len(button_inputs)):
        if i >= len(analog_inputs):
            analog_inputs.append((0,0))
        if i >= len(trick_inputs):
            trick_inputs.append(0)
        inputList.append(Frame([button_inputs[i][0],
                          button_inputs[i][1],
                          button_inputs[i][2],
                          button_inputs[i][3],
                          button_inputs[i][4],
                          analog_inputs[i][0],
                          analog_inputs[i][1],
                          trick_inputs[i]]))
    res = FrameSequence()
    res.read_from_list_of_frames(inputList)
    return res


def encodeFaceButton(aButton, bButton, lButton, pabButton, bdButton):
    return aButton * 0x1 + bButton * 0x2 + lButton * 0x4 + pabButton * 0x8 + bdButton * 0x10

def encodeDirectionInput(horizontalInput, verticalInput):
    return ((horizontalInput+7) << 4) + verticalInput+7

def encodeTrickInput(trickInput):
    return trickInput << 4

def encodeRKGFaceButtonInput(inputList : 'FrameSequence'):
    data = bytearray()
    iL = inputList.frames
    fbBytes = 0
    if len(iL) == 0 :
        return data, fbBytes

    prevInput = encodeFaceButton(iL[0].accel, iL[0].brake, iL[0].item, iL[0].drift, iL[0].brakedrift) #Not sure if it should be 0 instead
    amountCurrentFrames = 0x0   
    for ipt in iL:
        currentInput = encodeFaceButton(ipt.accel, ipt.brake, ipt.item, ipt.drift, ipt.brakedrift)
        if prevInput != currentInput or amountCurrentFrames >= 0xFF:
            data.append(prevInput)
            data.append(amountCurrentFrames)  
            prevInput = currentInput
            amountCurrentFrames = 0x1
            fbBytes = fbBytes + 1
        else:
            amountCurrentFrames = amountCurrentFrames + 1
    data.append(prevInput)
    data.append(amountCurrentFrames)  
    fbBytes = fbBytes + 1
    return data, fbBytes

def encodeRKGDirectionInput(inputList : 'FrameSequence'):
    data = bytearray()
    iL = inputList.frames
    diBytes = 0
    if len(iL) == 0 :
        return data, diBytes

    prevInput = encodeDirectionInput(iL[0].stick_x, iL[0].stick_y)
    amountCurrentFrames = 0x0
    for ipt in iL:
        currentInput = encodeDirectionInput(ipt.stick_x, ipt.stick_y)
        if prevInput != currentInput or amountCurrentFrames >= 0xFF:
            data.append(prevInput)
            data.append(amountCurrentFrames)  
            prevInput = currentInput
            amountCurrentFrames = 0x1
            diBytes = diBytes + 1
        else:
            amountCurrentFrames = amountCurrentFrames + 1
    data.append(prevInput)
    data.append(amountCurrentFrames)  
    diBytes = diBytes + 1
    return data, diBytes

def encodeRKGTrickInput(inputList : 'FrameSequence'):
    data = bytearray()
    iL = inputList.frames
    tiBytes = 0
    if len(iL) == 0 :
        return data, tiBytes

    prevInput = encodeTrickInput(iL[0].dpad_raw())
    amountCurrentFrames = 0x0  
    for ipt in iL:
        currentInput = encodeTrickInput(ipt.dpad_raw())
        if prevInput != currentInput or amountCurrentFrames >= 0xFFF:
            data.append(prevInput + (amountCurrentFrames >> 8))
            data.append(amountCurrentFrames % 0x100)
            prevInput = currentInput
            amountCurrentFrames = 0x1
            tiBytes = tiBytes + 1
        else:
            amountCurrentFrames = amountCurrentFrames + 1

    data.append(prevInput + (amountCurrentFrames >> 8))
    data.append(amountCurrentFrames % 0x100)
    tiBytes = tiBytes + 1
    return data, tiBytes
    
def encodeRKGInput(inputList : 'FrameSequence'):
    #Encode face buttons inputs
    fbData, fbBytes = encodeRKGFaceButtonInput(inputList)

    #Encode joystick inputs
    diData, diBytes = encodeRKGDirectionInput(inputList)
    
    #Encode trick inputs
    tiData, tiBytes = encodeRKGTrickInput(inputList)
            
    return fbData+diData+tiData, fbBytes, diBytes, tiBytes


def decode_RKG(raw_data : bytearray):
    ''' Decode RKG data to 3 objetcs :
            - RKGMetaData
            - FrameSequence
            - Mii Raw data (bytearray, size 0x4A)'''
    if len(raw_data) < 0x88:
        print("decode_RKG can't decode raw_data : raw_data too small")
    else:
        metadata = RKGMetaData(raw_data)
        inputList = decode_rkg_inputs(raw_data)
        mii_data = raw_data[0x3C:0x86]
        return metadata, inputList, mii_data


def encode_RKG(metadata : 'RKGMetaData', inputList : 'FrameSequence', mii_data : bytearray):
    ''' Encode to a RKG raw data (bytearray) '''
    inputData, fbBytes, diBytes, tiBytes = encodeRKGInput(inputList)
    dataIndex = (fbBytes + diBytes + tiBytes) * 0x2
    metadata.input_data_length = dataIndex + 0x8
    crc16_int = crc16(mii_data)

    metadata_bytes = metadata.to_bytes()
    crc16_data = bytearray([crc16_int >> 8, crc16_int & 0xFF])
    inputHeader = bytearray([fbBytes >> 8, fbBytes & 0xFF, diBytes >> 8, diBytes & 0xFF, tiBytes >> 8, tiBytes & 0xFF, 0, 0])

    if metadata.compressed_flag:
        compressed_input_data = compress(inputHeader + inputData)
        rkg_data = metadata_bytes + mii_data + crc16_data + len(compressed_input_data).to_bytes(4, 'big') + compressed_input_data
    else:
        rkg_data = metadata_bytes + mii_data + crc16_data + inputHeader + inputData
    crc32 = zlib.crc32(rkg_data)
    arg1 = floor(crc32 / 0x1000000)
    arg2 = floor((crc32 & 0x00FF0000) / 0x10000)
    arg3 = floor((crc32 & 0x0000FF00) / 0x100)
    arg4 = floor(crc32 % 0x100)
       
    return rkg_data + bytearray([arg1, arg2, arg3, arg4])


def get_RKG_data_memory():
    ''' Return (is_available, rkg_data) : (bool, bytearray)
        rkg_data is the uncompressed rkg created by the game when the race ends
        It's only available after the text "A ghost has been created" is displayed.
        is_available is a boolean saying if the text has appeared yet'''
    region = mkw_config.game_id_string
    try:
        address = {"RMCE01": 0x809B8F88, "RMCP01": 0x809BD748,
                "RMCJ01": 0x809BC7A8, "RMCK01": 0x809ABD88}
        rkg_addr = chase_pointer(address[region], [0x18], 'u32')
    except KeyError:
        raise common.RegionError
    if not memory.read_u32(rkg_addr) == 0x524b4744 :
        return False, None
    else:
        return True, common.read_bytes(rkg_addr, 0x2800)

