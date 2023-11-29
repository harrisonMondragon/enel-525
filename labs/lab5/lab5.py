import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Antecedent/Consequent objects hold universe variables and membership functions
speed = ctrl.Antecedent(np.arange(0, 81), 'speed')
incline = ctrl.Antecedent(np.arange(-15, 16), 'incline')
gear = ctrl.Consequent(np.arange(0, 5), 'gear', defuzzify_method="mom")

# Speed
speed['rolling'] = fuzz.trapmf(speed.universe, [0,0,10,11])
speed['slow'] = fuzz.trapmf(speed.universe, [10,11,25,26])
speed['medium'] = fuzz.trapmf(speed.universe, [25,26,50,51])
speed['fast'] = fuzz.trapmf(speed.universe, [50,51,65,66])
speed['speeding'] = fuzz.trapmf(speed.universe, [65,66,80,80])

# Incline
incline['steep'] = fuzz.trapmf(incline.universe, [-15,-15,-10,-9])
incline['slope'] = fuzz.trapmf(incline.universe, [-10,-9,-1,0])
incline['flat'] = fuzz.trapmf(incline.universe, [-1,0,0,1])
incline['up'] = fuzz.trapmf(incline.universe, [0,1,10,11])
incline['climb'] = fuzz.trapmf(incline.universe, [10,11,15,15])

# Gear
gear['first'] = fuzz.trimf(gear.universe, [0,0,1])
gear['second'] = fuzz.trimf(gear.universe, [0,1,2])
gear['third'] = fuzz.trimf(gear.universe, [1,2,3])
gear['fourth'] = fuzz.trimf(gear.universe, [2,3,4])
gear['fifth'] = fuzz.trimf(gear.universe, [3,4,4])

rule_list = []

# Rolling rules
rule_list.append(ctrl.Rule(speed['rolling'] & incline['steep'], gear['second']))
rule_list.append(ctrl.Rule(speed['rolling'] & incline['slope'], gear['first']))
rule_list.append(ctrl.Rule(speed['rolling'] & incline['flat'], gear['first']))
rule_list.append(ctrl.Rule(speed['rolling'] & incline['up'], gear['first']))
rule_list.append(ctrl.Rule(speed['rolling'] & incline['climb'], gear['first']))

# Slow rules
rule_list.append(ctrl.Rule(speed['slow'] & incline['steep'], gear['third']))
rule_list.append(ctrl.Rule(speed['slow'] & incline['slope'], gear['second']))
rule_list.append(ctrl.Rule(speed['slow'] & incline['flat'], gear['second']))
rule_list.append(ctrl.Rule(speed['slow'] & incline['up'], gear['first']))
rule_list.append(ctrl.Rule(speed['slow'] & incline['climb'], gear['first']))

# Medium rules
rule_list.append(ctrl.Rule(speed['medium'] & incline['steep'], gear['fourth']))
rule_list.append(ctrl.Rule(speed['medium'] & incline['slope'], gear['fourth']))
rule_list.append(ctrl.Rule(speed['medium'] & incline['flat'], gear['third']))
rule_list.append(ctrl.Rule(speed['medium'] & incline['up'], gear['third']))
rule_list.append(ctrl.Rule(speed['medium'] & incline['climb'], gear['second']))

# Fast rules
rule_list.append(ctrl.Rule(speed['fast'] & incline['steep'], gear['fifth']))
rule_list.append(ctrl.Rule(speed['fast'] & incline['slope'], gear['fifth']))
rule_list.append(ctrl.Rule(speed['fast'] & incline['flat'], gear['fourth']))
rule_list.append(ctrl.Rule(speed['fast'] & incline['up'], gear['fourth']))
rule_list.append(ctrl.Rule(speed['fast'] & incline['climb'], gear['fourth']))

# Speeding rules
rule_list.append(ctrl.Rule(speed['speeding'] & incline['steep'], gear['fifth']))
rule_list.append(ctrl.Rule(speed['speeding'] & incline['slope'], gear['fifth']))
rule_list.append(ctrl.Rule(speed['speeding'] & incline['flat'], gear['fifth']))
rule_list.append(ctrl.Rule(speed['speeding'] & incline['up'], gear['fourth']))
rule_list.append(ctrl.Rule(speed['speeding'] & incline['climb'], gear['fourth']))

gear_ctrl = ctrl.ControlSystem(rule_list)
gear_selector = ctrl.ControlSystemSimulation(gear_ctrl)

# Test inputs
gear_selector.input['speed'] = 15
gear_selector.input['incline'] = -12
gear_selector.compute()
print(gear_selector.output['gear'])
