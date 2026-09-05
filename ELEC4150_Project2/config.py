# ============================================================
# ELEC4150/8150 PROJECT 2
# Global Project Configuration
# ============================================================

# Part 1
AUDIO_FILE       = "audio_3-GANGNAM_STYLE.wav"  # primary audio file
TARGET_FS        = 60000    # Required sampling rate: 60 kHz
AUDIO_MAX_DURATION = 30     # Recommended < 30 seconds

# Part 2
BIT_RATE = 64000           # Required compression rate: 64 kbps
CARRIER_FREQ = 250000      # Required carrier: 250 kHz
MAX_BANDWIDTH = 4000       # Maximum allowed bandwidth: 4 kHz
CHANNEL_SNR_DB = 10        # Required AWGN SNR: 10 dB

# Part 3
SPEED_OF_SOUND = 343       # m/s
MIC_RADIUS = 1.0           # microphone circle radius: 1 m

# General
RANDOM_SEED = 42