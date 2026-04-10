import numpy as np
from PIL import Image



pgm = Image.open("samples/yes_sample_features_int8.pgm")
input = np.array(pgm)
with open("mfcc.txt", "w") as f:
    for x in input.flatten():
        f.write(f"{int(x)}\n")
np.savetxt("mfcc.csv", input, delimiter="  ", fmt="%d")