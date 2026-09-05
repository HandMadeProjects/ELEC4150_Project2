import numpy as np, sys
sys.path.insert(0, r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2")
import os; os.chdir(r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2")
from part1.audio_io import load_audio
audio, fs = load_audio("audio_1.wav")
L = audio[:,0]; R = audio[:,1]
corr = float(np.corrcoef(L,R)[0,1])
diff_power = float(np.mean((L-R)**2))
sum_power  = float(np.mean((L+R)**2))
side_ratio = diff_power / (sum_power + 1e-10)
print(f"L-R correlation   : {corr:.4f}")
print(f"Side power / Centre power : {side_ratio:.4f}  ({10*np.log10(side_ratio+1e-10):.1f} dB)")
if corr > 0.99:
    print("=> Audio type: MONO or near-mono (stereo width very small)")
elif corr > 0.85:
    print("=> Audio type: Lightly stereo (vocals likely somewhat spread)")
else:
    print("=> Audio type: Strongly stereo (distinct L/R content)")
