import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Antecedent/Consequent objects hold universe variables and membership functions
speed = ctrl.Antecedent(np.arange(0, 81), 'speed')
incline = ctrl.Antecedent(np.arange(-15, 16), 'incline')
gear = ctrl.Consequent(np.arange(0, 5), 'gear', defuzzify_method="mom")

speed['rolling'] = fuzz.trimf(speed.universe, [0, 0, 13])
speed['slow'] = fuzz.trimf(speed.universe, [9, 20, 27])
speed['medium'] = fuzz.trimf(speed.universe, [24, 39, 52])

speed.view()
plt.show()



