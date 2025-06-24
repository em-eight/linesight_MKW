from typing import TypedDict

class GCInputs(TypedDict, total=False):
    """
    Dictionary describing the state of a GameCube controller
    Boolean keys (buttons): True means pressed, False means released.
    Float keys for triggers: 0 means fully released, 1 means fully pressed.
    Float keys for sticks: 0 means neutral, ranges from -1 to 1.
    """
    A: bool
    B: bool
    X: bool
    Y: bool
    Z: bool
    Start: bool
    Up: bool
    Down: bool
    Left: bool
    Right: bool
    L: bool
    R: bool
    StickX: float
    StickY: float
    CStickX: float
    CStickY: float
    TriggerLeft: float
    TriggerRight: float

"""
When creating a list of inputs, several considerations must be made to help reduce the amount of options available to the ai.
This algorithm uses discrete inputs. It only chooses one option out of the following list to perform for any given frame.
Thus, too many inputs will complicate the network and at some point slow down training in theory. Testing needs to be done to find the amount of slowdown, if any.
For now (e.g. until I get around to testing it), it is recommended to keep the number of input combinations to below 20.

Therefore, when selecting inputs, consider the following:
    1. Soft-drifting charges a mini-turbo at the fastest rate while turning the least amount. This value is -3 and 3 for left and right respectively and should be present as an option in the inputs for optimized times.
    2. Using an item rarely needs to use more than one input for Time Trials, as you can simply match what should be pressed when a given track's shroom strat happens.
    3. Only the inputs deviating from the default state need to be present (i.e the 'X' and 'Y' buttons are never used, so they are omitted), however all used buttons should be present in all states.
    4. No non-accelerating inputs are necessary as drift inputs do not contribute to the start boost charge.
    5. More stick options will likely give small improvements to times, although human WRs rarely use them, if at all. Remains to be tested.
        (Note that the base game simplifies down to 15 unique values for steering, ranging from -7 to 7)
    6. It may be useful on certain tracks to cancel wheelies with D-pad down, on tracks such as rPB, rSL, and rBC, but I forgot where I found this information.
"""

""" 
GCInputs type list
    A: bool
    B: bool
    X: bool
    Y: bool
    Z: bool
    Start: bool
    Up: bool
    Down: bool
    Left: bool
    Right: bool
    L: bool
    R: bool
    StickX: float
    StickY: float
    CStickX: float
    CStickY: float
    TriggerLeft: float
    TriggerRight: float
    
Stick value conversion for GCInputs:
    (14) 205-255 (+7) > 1 - Full Right
    (13) 197-204 (+6) > 0.57
    (12) 188-196 (+5) > 0.5
    (11) 179-187 (+4) > 0.43
    (10) 170-178 (+3) > 0.36 - Soft Right
    (9) 161-169 (+2) > 0.29
    (8) 152-160 (+1) > 0.22
    (7) 113-151 (+0) > 0 - Neutral
    (6) 105-112 (-1) > -0.156
    (5) 96-104 (-2) > -0.22
    (4) 87-95 (-3) > -0.29 - Soft Left
    (3) 78-86 (-4) > -0.36
    (2) 69-77 (-5) > -0.43
    (1) 60-68 (-6) > -0.5
    (0) 0-59 (-7) > -1 - Full Left

Note that all of the values must use the range of -1 to 1 for the inputs list. (I learned this the hard way lol)
Chart based on this document: https://docs.google.com/document/d/e/2PACX-1vSM96Kykn6ILXsJD42gD3T71GJ_tiUGtHE8afTjqXX-Y2sxrXfHWuSNPHplKPt0IEvv0BNsHemluEIS/pub#h.jhvf61tw73t8
"""

defaultInputState: GCInputs = {
    "A": False,
    "B": False,
    "Up": False,
    "StickX": 0,
    "StickY": 0,
    "TriggerLeft": 0,
    "TriggerRight": 0
}

# Adjust for individual tracks for item usage or other things
inputs = [
    {  # 0 Forward
        "A": True,
        "B": False,
        "Up": False,
        "StickX": 0,
        "StickY": 1,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 1 Drift full left
        "A": True,
        "StickX": -1,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 2 Drift full right
        "StickX": 1,
        "A": True,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 3 Drift slight left (-1)
        "A": True,
        "StickX": -0.156,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 4 Drift slight right (1)
        "A": True,
        "StickX": 0.22,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 5 Drift straight
        "A": True,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickX": 0,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 6 Drift full right item # Adjust for individual tracks based on item usage
        "StickX": 1,
        "A": True,
        "TriggerRight": 1,
        "TriggerLeft": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
    },
    {  # 7 Full left
        "StickX": -1,
        "A": True,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 8 Full right
        "StickX": 1,
        "A": True,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 9 Trick full right
        "Up": True,
        "A": True,
        "B": False,
        "StickX": 1,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 10 Trick straight
        "Up": True,
        "A": True,
        "B": False,
        "StickX": 0,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 11 Trick full left
        "StickX": -1,
        "Up": True,
        "A": True,
        "B": False,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 12 No accel full left (Start boost/start slide) # 
        "A": False,
        "B": False,
        "Up": False,
        "StickX": -1,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
]

"""
# Adjust for individual tracks for item usage or other things
inputs = [
    {  # 0 Forward
        "A": True,
        "B": False,
        "Up": False,
        "StickX": 0,
        "StickY": 1,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 1 Drift full left
        "A": True,
        "StickX": -1,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 2 Drift full right
        "StickX": 1,
        "A": True,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 3 Drift slight left
        "A": True,
        "StickX": -0.29,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0.43,
        "TriggerLeft": 0,
    },
    {  # 4 Drift slight right
        "A": True,
        "StickX": 0.36,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickY": 0.43,
        "TriggerLeft": 0,
    },
    {  # 5 Drift straight
        "A": True,
        "TriggerRight": 1,
        "B": False,
        "Up": False,
        "StickX": 0,
        "StickY": 0,
        "TriggerLeft": 0,
    },
    {  # 6 Drift full right item # Adjust for individual tracks based on item usage
        "StickX": 1,
        "A": True,
        "TriggerRight": 1,
        "TriggerLeft": 1,
        "B": False,
        "Up": False,
        "StickY": 0,
    },
    {  # 7 Full left
        "StickX": -1,
        "A": True,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 8 Full right
        "StickX": 1,
        "A": True,
        "B": False,
        "Up": False,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 9 Trick full right
        "Up": True,
        "A": True,
        "B": False,
        "StickX": 1,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 10 Trick straight
        "Up": True,
        "A": True,
        "B": False,
        "StickX": 0,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 11 Trick full left
        "StickX": -1,
        "Up": True,
        "A": True,
        "B": False,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
    {  # 12 No accel full right (Start boost/start slide) # 
        "A": False,
        "B": False,
        "Up": False,
        "StickX": 1,
        "StickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0
    },
]
"""

action_forward_idx = 0  # Accelerate forward, don't turn
action_backward_idx = 11  # Don't move, don't turn
