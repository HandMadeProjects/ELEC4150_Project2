
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import queue
import time


def play_with_realtime_spectrogram(audio, fs, duration=10):
    """
    Play audio continuously while displaying a real-time spectrogram.

    Audio playback is handled by the sounddevice callback.
    Spectrogram processing is performed separately so that
    plotting does not interrupt audio playback.
    """

    # ========================================================
    # 1. Limit audio to requested demonstration duration
    # ========================================================

    max_samples = int(duration * fs)

    audio = audio[:max_samples].astype(np.float32)

    actual_duration = len(audio) / fs

    print("\n========================================")
    print("REAL-TIME AUDIO + SPECTROGRAM")
    print("========================================")
    print(f"Sampling rate : {fs} Hz")
    print(f"Duration      : {actual_duration:.2f} seconds")

    # ========================================================
    # 2. Queue for communication between audio callback
    #    and spectrogram processing
    # ========================================================

    audio_queue = queue.Queue(maxsize=20)

    # ========================================================
    # 3. Playback variables
    # ========================================================

    current_position = 0

    # ========================================================
    # 4. Audio callback
    # ========================================================

    def audio_callback(outdata, frames, time_info, status):

        nonlocal current_position

        if status:
            print("Audio status:", status)

        start = current_position
        end = start + frames

        # ----------------------------------------------------
        # If there is still audio remaining
        # ----------------------------------------------------

        if start < len(audio):

            block = audio[start:end]

            # Number of samples actually available
            samples_available = len(block)

            # Fill output buffer with zeros first
            outdata.fill(0)

            # Copy audio into output buffer
            outdata[:samples_available, 0] = block

            # ------------------------------------------------
            # Send a copy to spectrogram queue
            # ------------------------------------------------

            try:
                audio_queue.put_nowait(block.copy())
            except queue.Full:
                pass

            current_position = end

        else:

            outdata.fill(0)

            raise sd.CallbackStop

    # ========================================================
    # 5. Create spectrogram figure
    # ========================================================

    plt.ion()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title(
        "Real-Time Spectrogram - 60 kHz Audio"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # ========================================================
    # 6. Spectrogram parameters
    # ========================================================

    n_fft = 2048

    window = np.hanning(n_fft)

    # Store spectrogram history
    spectrogram_history = []

    # Frequency axis
    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1 / fs
    )

    # Only display up to 10 kHz
    max_frequency = 10000

    frequency_indices = (
        frequencies <= max_frequency
    )

    display_frequencies = frequencies[
        frequency_indices
    ]

    # Initial image
    initial_data = np.full(
        (len(display_frequencies), 1),
        -100.0
    )

    image = ax.imshow(
        initial_data,
        aspect="auto",
        origin="lower",
        extent=[
            0,
            actual_duration,
            0,
            max_frequency
        ],
        vmin=-80,
        vmax=0,
        cmap="magma"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Magnitude (dB)"
    )

    ax.set_xlim(
        0,
        actual_duration
    )

    ax.set_ylim(
        0,
        max_frequency
    )

    # ========================================================
    # 7. Start audio stream
    # ========================================================

    print("\nStarting audio playback...")
    print("Spectrogram is running simultaneously.")
    print("Press Ctrl+C in the terminal to stop.")

    try:

        with sd.OutputStream(
            samplerate=fs,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=audio_callback
        ):

            start_time = time.time()

            # ------------------------------------------------
            # Main display loop
            # ------------------------------------------------

            while (
                current_position < len(audio)
            ):

                # --------------------------------------------
                # Get audio blocks from queue
                # --------------------------------------------

                while not audio_queue.empty():

                    block = audio_queue.get()

                    # ----------------------------------------
                    # Zero-pad short block
                    # ----------------------------------------

                    if len(block) < n_fft:

                        padded_block = np.zeros(
                            n_fft,
                            dtype=np.float32
                        )

                        padded_block[
                            :len(block)
                        ] = block

                        block = padded_block

                    else:

                        block = block[:n_fft]

                    # ----------------------------------------
                    # Apply Hann window
                    # ----------------------------------------

                    windowed = block * window

                    # ----------------------------------------
                    # FFT
                    # ----------------------------------------

                    spectrum = np.fft.rfft(
                        windowed
                    )

                    magnitude = np.abs(
                        spectrum
                    )

                    # ----------------------------------------
                    # Convert to dB
                    # ----------------------------------------

                    magnitude_db = (
                        20
                        * np.log10(
                            magnitude + 1e-10
                        )
                    )

                    # ----------------------------------------
                    # Keep frequencies up to 10 kHz
                    # ----------------------------------------

                    magnitude_db = magnitude_db[
                        frequency_indices
                    ]

                    spectrogram_history.append(
                        magnitude_db
                    )

                # --------------------------------------------
                # Update spectrogram
                # --------------------------------------------

                if len(spectrogram_history) > 0:

                    data = np.array(
                        spectrogram_history
                    ).T

                    image.set_data(data)

                    image.set_extent(
                        [
                            0,
                            len(spectrogram_history)
                            * 1024
                            / fs,
                            0,
                            max_frequency
                        ]
                    )

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                # --------------------------------------------
                # Small pause to allow GUI updates
                # --------------------------------------------

                plt.pause(0.01)

        print("\nPlayback finished.")

    except KeyboardInterrupt:

        print("\nPlayback stopped by user.")

        sd.stop()

    finally:

        plt.ioff()
        plt.show()

