"""

Implementation of the pipeline from Meng et al. (2019)
Reference: https://www.ijbs.com/v15p1921.htm

- Type-I Chebyshev band-pass filtering
- modified coif2 wavelet thresholding
- LMS adaptive heart-sound reduction using a reference estimated from the input signal.

"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import librosa
import numpy as np
import pywt
from scipy.signal import cheby1, filtfilt


class Denoiser:
    """Serial integrated denoiser for respiratory sounds."""

    DEFAULT_THRESHOLDS = {
        "CD1": 0.01,
        "CD2": 0.05,
        "CD3": 0.05,
        "CD4": 0.001,
        "CD5": 0.002,
        "CD6": 0.005,
        "CD9": 0.08,
        "CA9": 0.3,
    }

    def __init__(
        self,
        *,
        low_hz: float = 50.0,
        high_hz: float = 2000.0,
        order: int = 4,
        ripple_db: float = 0.5,
        standard_power: float | None = None,
        standard_signal: np.ndarray | None = None,
        wavelet: str = "coif2",
        wavelet_level: int = 9,
        thresholds: Mapping[str, float] | None = None,
        wavelet_mode: str = "periodization",
        lms_order: int = 32,
        lms_mu: float = 0.001,
        epoch_sec: float = 0.8,
        peak_factor: float = 1.5,
        nt_sec: float = 0.15,
        heart_half_sec: float = 0.08,
        skip_short_signal: bool = True,  # skip or not (if too short)
    ) -> None:
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order
        self.ripple_db = ripple_db
        self.standard_power = self._resolve_standard_power(standard_power, standard_signal)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.thresholds = dict(self.DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        self.wavelet_mode = wavelet_mode
        self.lms_order = lms_order
        self.lms_mu = lms_mu
        self.epoch_sec = epoch_sec
        self.peak_factor = peak_factor
        self.nt_sec = nt_sec
        self.heart_half_sec = heart_half_sec
        self.skip_short_signal = skip_short_signal

    # def run(self, input_path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray | None, int]:
    #     x, sr = librosa.load(input_path, sr=target_sr, mono=True)
    #     return self.denoise_signal(x, sr), sr

    def run(self, signal: np.ndarray, sr: int) -> np.ndarray | None:
        y = self.bandpass(signal, sr)
        y = self.normalize_power(y)
        y = self.wavelet_threshold_denoise(y)
        if y is None:
            return None

        reference = self.build_heart_reference(y, sr)
        y = self.lms_adaptive_filter(y, reference)

        return y

    # Step 1: Bandpass filtering
    def bandpass(self, signal: np.ndarray, sr: int) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)

        nyquist = sr * 0.5
        low = self.low_hz / nyquist
        high = self.high_hz / nyquist

        b, a = cheby1(self.order, self.ripple_db, [low, high], btype='bandpass')
        filtered = filtfilt(b, a, signal)  # zero-phase filtering
        return filtered

    # Step 2. Wavelet thresholding
    @staticmethod
    def _signal_power(signal: np.ndarray, eps: float = 1e-12) -> float | None:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.size == 0:
            return None
        power = float(np.mean(signal * signal))
        return power if power > eps else None

    def _resolve_standard_power(
        self,
        standard_power: float | None,
        standard_signal: np.ndarray | None,
    ) -> float | None:
        if standard_power is not None and standard_signal is not None:
            raise ValueError("Use either standard_power or standard_signal, not both.")
        if standard_signal is not None:
            return self._signal_power(standard_signal)
        if standard_power is None:
            return None
        if standard_power <= 0:
            raise ValueError("standard_power must be positive.")
        return float(standard_power)

    def normalize_power(self, signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        standard_power = self.standard_power
        if standard_power is None:
            return signal

        power = self._signal_power(signal, eps=eps)
        if power is None:
            return signal
        return (signal * np.sqrt(standard_power / power)).astype(np.float32)

    def wavelet_threshold_denoise(self, signal: np.ndarray) -> np.ndarray | None:
        """decomposition using coif2 9-level wavelet"""
        signal = np.asarray(signal, dtype=np.float32)

        wavelet_obj = pywt.Wavelet(self.wavelet)  # wavelet: "coif2"

        # skip if too short to decompose a signal into 9 levels
        max_level = pywt.dwt_max_level(signal.size, wavelet_obj.dec_len)
        if max_level < self.wavelet_level:
            message = (
                f"Signal is too short for {int(self.wavelet_level)}-level {self.wavelet} DWT: "
                f"len={signal.size}, max_level={max_level}"
            )
            if self.skip_short_signal:
                print(f"[skip] {message}")
                return None
            raise ValueError(message)


        coeffs = pywt.wavedec(signal, wavelet_obj, mode=self.wavelet_mode, level=self.wavelet_level)
        ca_threshold = self.thresholds.get(f"CA{self.wavelet_level}")
        if ca_threshold is not None:
            approx = pywt.threshold(coeffs[0], float(ca_threshold), mode="soft")
        else:
            approx = coeffs[0]
        denoised = [approx]

        for idx, coeff in enumerate(coeffs[1:], start=1):
            detail_level = self.wavelet_level - idx + 1
            key = f"CD{detail_level}"
            if detail_level in (1, 2):
                kind = "hard"
            elif detail_level in (3, 4, 5, 6, 9):
                kind = "soft"
            else:
                kind = None

            threshold = self.thresholds.get(key)
            if threshold is not None and kind is not None:
                coeff = pywt.threshold(coeff, float(threshold), mode=kind)
            denoised.append(coeff)

        restored = pywt.waverec(denoised, wavelet_obj, mode=self.wavelet_mode)  # inverse DWT

        return restored[: signal.size].astype(np.float32)

    # Step 3. LMS adaptive filter to reduce heart sound
    def build_heart_reference(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Estimate the LMS reference signal by copying likely heart-beat windows."""
        signal = np.asarray(signal, dtype=np.float32)
        reference = np.zeros_like(signal, dtype=np.float32)
        if signal.size == 0:
            return reference

        epoch = max(1, int(round(self.epoch_sec * sr)))
        nt = max(1, int(round(self.nt_sec * sr)))
        half = max(1, int(round(self.heart_half_sec * sr)))

        candidates = []
        for start in range(0, signal.size, epoch):
            stop = min(signal.size, start + epoch)
            window = signal[start:stop]
            if window.size == 0:
                continue
            max_idx = int(np.argmax(window))
            min_idx = int(np.argmin(window))
            peak = max(abs(float(window[max_idx])), abs(float(window[min_idx])))
            candidates.append((peak, start, max_idx, min_idx))

        if not candidates:
            return reference

        mean_peak = float(np.mean([candidate[0] for candidate in candidates]))
        for peak, start, max_idx, min_idx in candidates:
            if peak < self.peak_factor * mean_peak or abs(max_idx - min_idx) > nt:
                continue
            center = start + max_idx
            left = max(0, center - half)
            right = min(signal.size, center + half + 1)
            reference[left:right] = signal[left:right]
        return reference

    def lms_adaptive_filter(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Standard LMS stage; output is signal minus estimated heart-sound noise."""
        signal = np.asarray(signal, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if signal.size == 0 or reference.size != signal.size or not np.any(reference):
            return signal

        weights = np.zeros(int(self.lms_order), dtype=np.float32)
        xbuf = np.zeros_like(weights)
        output = np.zeros_like(signal, dtype=np.float32)

        for i, x in enumerate(reference):
            xbuf[1:] = xbuf[:-1]
            xbuf[0] = x

            estimated_noise = float(np.dot(weights, xbuf))
            error = signal[i] - estimated_noise

            weights += 2.0 * float(self.lms_mu) * error * xbuf
            output[i] = error

        return output

    def visualize_spectrogram(
        self,
        before: np.ndarray,
        after: np.ndarray,
        sr: int,
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 50,
        fmax: float = 2000,
        show: bool = True,
    ):
        import librosa.display
        import matplotlib.pyplot as plt

        def mel_spectrogram(x: np.ndarray) -> np.ndarray:
            return librosa.feature.melspectrogram(
                y=x,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
            )

        before = np.asarray(before, dtype=np.float32)
        after_raw = np.asarray(after, dtype=np.float32)

        before_mel = mel_spectrogram(before)
        after_mel = mel_spectrogram(after_raw)

        shared_ref = max(float(before_mel.max()), float(after_mel.max()), 1e-12)
        before_db = librosa.power_to_db(before_mel, ref=shared_ref)
        after_db = librosa.power_to_db(after_mel, ref=shared_ref)

        power_vmin = min(float(before_db.min()), float(after_db.min()))
        power_vmax = max(float(before_db.max()), float(after_db.max()))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        img = librosa.display.specshow(
            before_db,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=fmin,
            fmax=fmax,
            cmap='magma',
            vmin=power_vmin,
            vmax=power_vmax,
            ax=axes[0],
        )
        axes[0].set_title('Before denoising')

        librosa.display.specshow(
            after_db,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=fmin,
            fmax=fmax,
            cmap='magma',
            vmin=power_vmin,
            vmax=power_vmax,
            ax=axes[1],
        )
        axes[1].set_title('After denoising')

        for ax in axes:
            ax.set_ylabel('Mel frequency (Hz)')
            ax.tick_params(labelleft=True)

        cbar = fig.colorbar(img, ax=axes, format='%+2.0f dB')
        cbar.set_label('Power (dB)')

        fig.suptitle('Mel Spectrogram Comparison')
        if show:
            plt.show()
        return fig

    def visualize_signal(self, before: np.ndarray, after: np.ndarray, sr: int, *, show: bool = True):
        import matplotlib.pyplot as plt

        def rms(x: np.ndarray, eps: float = 1e-12) -> float:
            return max(float(np.sqrt(np.mean(x * x))), eps)

        before = np.asarray(before, dtype=np.float32)
        after_raw = np.asarray(after, dtype=np.float32)

        t = np.arange(before.size) / sr
        rms_gain_db = 20.0 * np.log10(rms(after_raw) / rms(before))

        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.plot(t, before, linewidth=0.8, label='Before')
        ax.plot(t, after_raw, linewidth=0.8, alpha=0.75, label='After')
        ax.set_title(f'Amplitude comparison (after gain={rms_gain_db:+.2f} dB)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')

        print(f'RMS before={rms(before):.6f}, after_raw={rms(after_raw):.6f}, gain={rms_gain_db:+.2f} dB')

        plt.tight_layout()
        if show:
            plt.show()
        return fig


def peak_normalize_for_wav(signal: np.ndarray, peak: float = 0.99, eps: float = 1e-8) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs <= eps:
        return signal
    if max_abs > peak:
        signal = signal * (peak / max_abs)
    return signal.astype(np.float32)


def make_mel_spectrogram(
    signal: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 50,
    fmax: float = 2000,
) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=np.asarray(signal, dtype=np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    ).astype(np.float32)


def save_mel_png(
    mel_power: np.ndarray,
    sr: int,
    out_path: str | Path,
    *,
    hop_length: int = 512,
    fmin: float = 50,
    fmax: float = 2000,
) -> None:
    import librosa.display
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Denoised mel spectrogram")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("Power (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_denoised_sample(
    input_path: str | Path,
    output_root: str | Path,
    *,
    target_sr: int | None = None,
    denoiser: Denoiser | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 50,
    fmax: float = 2000,
    save_comparison_plots: bool = True,
) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    input_path = Path(input_path)
    output_root = Path(output_root)
    denoiser = denoiser if denoiser is not None else Denoiser()

    signal, sr = librosa.load(input_path, sr=target_sr, mono=True)
    denoised = denoiser.run(signal, sr)
    if denoised is None:
        print(f"[skip] denoising failed or signal too short: {input_path}")
        return None

    stem = input_path.stem
    wav_path = output_root / "wav" / f"{stem}_denoised.wav"
    mel_npy_path = output_root / "mel_npy" / f"{stem}_denoised_mel.npy"
    mel_png_path = output_root / "mel_png" / f"{stem}_denoised_mel.png"

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    mel_npy_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(wav_path, peak_normalize_for_wav(denoised), sr)

    mel = make_mel_spectrogram(
        denoised,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    np.save(mel_npy_path, mel)
    save_mel_png(mel, sr, mel_png_path, hop_length=hop_length, fmin=fmin, fmax=fmax)

    if save_comparison_plots:
        plot_dir = output_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig = denoiser.visualize_signal(signal, denoised, sr, show=False)
        fig.savefig(plot_dir / f"{stem}_signal_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = denoiser.visualize_spectrogram(
            signal,
            denoised,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            show=False,
        )
        fig.savefig(plot_dir / f"{stem}_mel_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[saved] wav: {wav_path}")
    print(f"[saved] mel npy: {mel_npy_path}")
    print(f"[saved] mel png: {mel_png_path}")
    return wav_path


def save_denoised_directory(
    input_dir: str | Path,
    output_root: str | Path,
    *,
    pattern: str = "*.wav",
    target_sr: int | None = None,
    recursive: bool = True,
) -> None:
    input_dir = Path(input_dir)
    wav_paths = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    denoiser = Denoiser()
    print(f"found wav files: {len(wav_paths)}")
    for wav_path in wav_paths:
        save_denoised_sample(
            wav_path,
            output_root,
            target_sr=target_sr,
            denoiser=denoiser,
        )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/coder/workspace/data/hylee/snuch/gt/6s_repeat/wav/241219/241219_1318_s11.wav",
        help="Input wav file or directory of segment wav files.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/coder/workspace/data/hylee/denoising/denoised_samples",
        help="Directory for denoised wav, mel npy/png, and comparison plots.",
    )
    parser.add_argument("--target_sr", type=int, default=None)
    parser.add_argument("--no_recursive", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        save_denoised_directory(
            input_path,
            args.output_dir,
            target_sr=args.target_sr,
            recursive=not args.no_recursive,
        )
    else:
        save_denoised_sample(input_path, args.output_dir, target_sr=args.target_sr)
