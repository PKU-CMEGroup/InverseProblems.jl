import numpy as np

#--------------------------------------
# numerical constants
#--------------------------------------
__logZero = -700.0 # (used for communicating that a probability weight is zero)
__wtSumThresh = np.exp(__logZero+1.0)