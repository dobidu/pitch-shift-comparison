#!/usr/bin/env python3
"""
Enhanced Comparative Script for Pitch Shifting and Autotune Libraries
====================================================================

Tests and compares different Python libraries for pitch shifting and autotune,
including new libraries like pyrubberband and autotune implementations.

Included improvements:
- Support for pyrubberband (wrapper for Rubberband)
- Custom autotune implementations
- Enhanced quality metrics
- Latency and memory usage analysis
- Robustness tests with different audio types

Author: Enhanced Audio Library Comparison
Version: 2.0.0
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
import warnings
import psutil
import threading
import json
from collections import defaultdict

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class TestResult:
    """Result of a pitch shifting test."""
    library_name: str
    algorithm_name: str
    semitones: float
    processing_time: float
    output_audio: Optional[np.ndarray]
    error: Optional[str]
    quality_metrics: Dict[str, float]
    memory_usage: float = 0.0
    latency: float = 0.0
    cpu_usage: float = 0.0

@dataclass
class LibraryInfo:
    """Information about a library."""
    name: str
    version: str
    available: bool
    import_error: Optional[str]
    algorithms: List[str]
    installation_notes: str = ""
    requires_external: bool = False

@dataclass
class AudioTestCase:
    """Audio test case."""
    name: str
    audio: np.ndarray
    sr: int
    description: str
    fundamental_freq: float = 440.0

class AdvancedAudioQualityAnalyzer:
    """Advanced audio quality analyzer."""
    
    @staticmethod
    def calculate_snr(original: np.ndarray, processed: np.ndarray) -> float:
        """Calculates Signal-to-Noise Ratio."""
        if len(original) != len(processed):
            min_len = min(len(original), len(processed))
            original = original[:min_len]
            processed = processed[:min_len]
        
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((processed - original) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def calculate_thd(audio: np.ndarray, sr: int, fundamental_freq: float) -> float:
        """Calculates Total Harmonic Distortion."""
        try:
            # FFT of signal
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/sr)
            magnitude = np.abs(fft)
            
            # Find fundamental peak
            fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
            fund_power = magnitude[fund_idx] ** 2
            
            # Calculate harmonic power
            harmonic_power = 0
            for h in range(2, 6):  # 2nd to 5th harmonic
                harm_freq = fundamental_freq * h
                harm_idx = np.argmin(np.abs(freqs - harm_freq))
                if harm_idx < len(magnitude):
                    harmonic_power += magnitude[harm_idx] ** 2
            
            if fund_power == 0:
                return float('inf')
            
            return 10 * np.log10(harmonic_power / fund_power)
            
        except Exception:
            return float('nan')
    
    @staticmethod
    def calculate_spectral_centroid_stability(audio: np.ndarray, sr: int) -> float:
        """Calculates spectral centroid stability."""
        try:
            import librosa
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            return float(np.std(centroid) / (np.mean(centroid) + 1e-10))
        except Exception:
            return float('nan')
    
    @staticmethod
    def calculate_pitch_accuracy(original_pitch: float, processed_audio: np.ndarray, sr: int) -> float:
        """Calculates pitch accuracy using automatic detection."""
        try:
            import librosa
            pitches, magnitudes = librosa.piptrack(y=processed_audio, sr=sr)
            
            # Find predominant pitch
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return float('nan')
            
            detected_pitch = np.median(pitch_values)
            error_cents = 1200 * np.log2(detected_pitch / original_pitch)
            return float(abs(error_cents))
            
        except Exception:
            return float('nan')
    
    @staticmethod
    def calculate_phase_coherence(original: np.ndarray, processed: np.ndarray) -> float:
        """Calculates phase coherence between signals."""
        try:
            if len(original) != len(processed):
                min_len = min(len(original), len(processed))
                original = original[:min_len]
                processed = processed[:min_len]
            
            # FFT of signals
            fft_orig = np.fft.fft(original)
            fft_proc = np.fft.fft(processed)
            
            # Calculate phase coherence
            cross_spectrum = fft_orig * np.conj(fft_proc)
            coherence = np.abs(cross_spectrum) / (np.abs(fft_orig) * np.abs(fft_proc) + 1e-10)
            
            return float(np.mean(coherence))
            
        except Exception:
            return float('nan')
    
    def calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculates all quality metrics."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['snr'] = self.calculate_snr(original, processed)
            metrics['thd'] = self.calculate_thd(processed, sr, 440.0)  # Assume 440Hz as fundamental
            metrics['spectral_stability'] = self.calculate_spectral_centroid_stability(processed, sr)
            metrics['phase_coherence'] = self.calculate_phase_coherence(original, processed)
            
            # Additional metrics
            metrics['rms_difference'] = float(np.sqrt(np.mean((processed - original) ** 2)))
            metrics['dynamic_range'] = float(20 * np.log10(np.max(np.abs(processed)) / (np.mean(np.abs(processed)) + 1e-10)))
            metrics['zero_crossing_rate'] = float(np.mean(np.diff(np.sign(processed)) != 0))
            
            # Pitch accuracy (if possible)
            try:
                target_pitch = 440.0 * (2 ** (0 / 12))  # Target pitch based on semitone
                metrics['pitch_accuracy'] = self.calculate_pitch_accuracy(target_pitch, processed, sr)
            except:
                metrics['pitch_accuracy'] = float('nan')
                
        except Exception as e:
            # In case of error, return empty metrics
            metrics = {
                'snr': float('nan'),
                'thd': float('nan'),
                'spectral_stability': float('nan'),
                'phase_coherence': float('nan'),
                'rms_difference': float('nan'),
                'dynamic_range': float('nan'),
                'zero_crossing_rate': float('nan'),
                'pitch_accuracy': float('nan')
            }
        
        return metrics

class EnhancedPitchShiftingTester:
    """Enhanced pitch shifting library tester."""
    
    def __init__(self):
        """Initializes the tester."""
        self.analyzer = AdvancedAudioQualityAnalyzer()
        self.libraries = {}
        self.test_results = []
        self.test_cases = []
        
        # Detect available libraries
        self._detect_libraries()
        self._create_test_cases()
    
    def _detect_libraries(self):
        """Detects which libraries are available."""
        self.libraries = {}
        
        # LibROSA
        try:
            import librosa
            self.libraries['librosa'] = LibraryInfo(
                name='LibROSA',
                version=librosa.__version__,
                available=True,
                import_error=None,
                algorithms=['pitch_shift', 'phase_vocoder', 'pitch_shift_hifi'],
                installation_notes="pip install librosa soundfile"
            )
        except ImportError as e:
            self.libraries['librosa'] = LibraryInfo(
                name='LibROSA',
                version='N/A',
                available=False,
                import_error=str(e),
                algorithms=[],
                installation_notes="pip install librosa soundfile"
            )
        
        # PyDub
        try:
            import pydub
            self.libraries['pydub'] = LibraryInfo(
                name='PyDub',
                version=getattr(pydub, '__version__', 'Unknown'),
                available=True,
                import_error=None,
                algorithms=['speed_change'],
                installation_notes="pip install pydub"
            )
        except ImportError as e:
            self.libraries['pydub'] = LibraryInfo(
                name='PyDub',
                version='N/A',
                available=False,
                import_error=str(e),
                algorithms=[],
                installation_notes="pip install pydub"
            )
        
        # Praat-parselmouth
        try:
            import parselmouth
            self.libraries['parselmouth'] = LibraryInfo(
                name='Praat-parselmouth',
                version=parselmouth.__version__,
                available=True,
                import_error=None,
                algorithms=['psola', 'change_gender'],
                installation_notes="pip install praat-parselmouth"
            )
        except ImportError as e:
            self.libraries['parselmouth'] = LibraryInfo(
                name='Praat-parselmouth',
                version='N/A',
                available=False,
                import_error=str(e),
                algorithms=[],
                installation_notes="pip install praat-parselmouth"
            )
        
        # pedalboard (Spotify)
        try:
            import pedalboard
            self.libraries['pedalboard'] = LibraryInfo(
                name='pedalboard (Spotify)',
                version=pedalboard.__version__,
                available=True,
                import_error=None,
                algorithms=['pitch_shift'],
                installation_notes="pip install pedalboard"
            )
        except ImportError as e:
            self.libraries['pedalboard'] = LibraryInfo(
                name='pedalboard (Spotify)',
                version='N/A',
                available=False,
                import_error=str(e),
                algorithms=[],
                installation_notes="pip install pedalboard"
            )
        
        # pyrubberband
        try:
            import pyrubberband
            self.libraries['pyrubberband'] = LibraryInfo(
                name='pyrubberband',
                version=getattr(pyrubberband, '__version__', 'Unknown'),
                available=True,
                import_error=None,
                algorithms=['pitch_shift', 'time_stretch'],
                installation_notes="pip install pyrubberband (requires rubberband-cli installed)",
                requires_external=True
            )
        except ImportError as e:
            self.libraries['pyrubberband'] = LibraryInfo(
                name='pyrubberband',
                version='N/A',
                available=False,
                import_error=str(e),
                algorithms=[],
                installation_notes="pip install pyrubberband (requires rubberband-cli installed)",
                requires_external=True
            )
        
        # TimeSide - REMOVED (very complex dependencies)
        # self.libraries['timeside'] = LibraryInfo(...)
        
        # aubio - REMOVED (compatibility issues)
        # self.libraries['aubio'] = LibraryInfo(...)
        
        # Essentia - REMOVED (build issues)
        # self.libraries['essentia'] = LibraryInfo(...)
        
        # SciPy (manual implementation)
        try:
            import scipy
            self.libraries['scipy'] = LibraryInfo(
                name='SciPy (manual)',
                version=scipy.__version__,
                available=True,
                import_error=None,
                algorithms=['resampling', 'phase_vocoder_manual', 'autotune_manual'],
                installation_notes="pip install scipy"
            )
        except ImportError as e:
            self.libraries['scipy'] = LibraryInfo(
                name='SciPy (manual)',
                version='N/A',
                available=False,
                import_error=str(e),
                algorithms=[],
                installation_notes="pip install scipy"
            )
    
    def _create_test_cases(self):
        """Creates different audio test cases."""
        sr = 44100
        duration = 2.0
        
        # Case 1: Simple pure tone
        t = np.linspace(0, duration, int(sr * duration))
        pure_tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        self.test_cases.append(AudioTestCase(
            name="pure_tone",
            audio=pure_tone.astype(np.float32),
            sr=sr,
            description="Pure 440Hz tone",
            fundamental_freq=440.0
        ))
        
        # Case 2: Tone with harmonics
        harmonic_tone = (0.6 * np.sin(2 * np.pi * 440 * t) +
                        0.3 * np.sin(2 * np.pi * 880 * t) +
                        0.1 * np.sin(2 * np.pi * 1320 * t))
        envelope = np.exp(-t * 0.5)  # Decay envelope
        harmonic_tone *= envelope
        self.test_cases.append(AudioTestCase(
            name="harmonic_tone",
            audio=harmonic_tone.astype(np.float32),
            sr=sr,
            description="Tone with harmonics and envelope",
            fundamental_freq=440.0
        ))
        
        # Case 3: Complex chord
        chord = (0.4 * np.sin(2 * np.pi * 261.63 * t) +  # C4
                0.4 * np.sin(2 * np.pi * 329.63 * t) +   # E4
                0.4 * np.sin(2 * np.pi * 392.00 * t))    # G4
        self.test_cases.append(AudioTestCase(
            name="chord",
            audio=chord.astype(np.float32),
            sr=sr,
            description="C major chord",
            fundamental_freq=261.63
        ))
        
        # Case 4: Signal with noise
        noisy_tone = pure_tone + 0.05 * np.random.normal(0, 1, len(pure_tone))
        self.test_cases.append(AudioTestCase(
            name="noisy_tone",
            audio=noisy_tone.astype(np.float32),
            sr=sr,
            description="Pure tone with noise",
            fundamental_freq=440.0
        ))
        
        # Case 5: Frequency sweep
        freq_sweep = np.sin(2 * np.pi * (440 + 200 * t) * t)
        self.test_cases.append(AudioTestCase(
            name="freq_sweep",
            audio=freq_sweep.astype(np.float32),
            sr=sr,
            description="Frequency sweep 440-640Hz",
            fundamental_freq=540.0  # Average frequency
        ))
    
    def _monitor_resources(self, duration: float = 1.0) -> Dict[str, float]:
        """Monitors resource usage during execution."""
        import psutil
        import threading
        import time
        
        process = psutil.Process()
        
        # Initial measurements
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_samples = []
        memory_samples = []
        
        # Flag to stop monitoring
        stop_monitoring = threading.Event()
        
        def collect_samples():
            while not stop_monitoring.is_set():
                try:
                    cpu_samples.append(process.cpu_percent(interval=None))
                    memory_samples.append(process.memory_info().rss / 1024 / 1024)
                    time.sleep(0.05)  # Sample every 50ms
                except:
                    break
        
        # Start collection in separate thread
        monitor_thread = threading.Thread(target=collect_samples)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for specified duration
        time.sleep(duration)
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join(timeout=0.5)
        
        # Calculate metrics
        if memory_samples:
            final_memory = memory_samples[-1]
            max_memory = max(memory_samples)
            memory_delta = max_memory - initial_memory
        else:
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = final_memory - initial_memory
        
        if cpu_samples:
            # Remove NaN and zero values
            valid_cpu = [c for c in cpu_samples if not np.isnan(c) and c >= 0]
            if valid_cpu:
                cpu_avg = np.mean(valid_cpu)
                cpu_max = max(valid_cpu)
            else:
                cpu_avg = 0.0
                cpu_max = 0.0
        else:
            cpu_avg = 0.0
            cpu_max = 0.0
        
        return {
            'memory_delta': memory_delta,
            'cpu_avg': cpu_avg,
            'cpu_max': cpu_max
        }
    
    # === EXISTING PITCH SHIFTING IMPLEMENTATIONS ===
    
    def pitch_shift_librosa(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Pitch shifting using LibROSA."""
        import librosa
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=semitones, bins_per_octave=12)
    
    def pitch_shift_librosa_hifi(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """LibROSA pitch shifting with high quality settings."""
        import librosa
        return librosa.effects.pitch_shift(
            y=audio, sr=sr, n_steps=semitones, 
            bins_per_octave=12, res_type='kaiser_best'
        )
    
    def pitch_shift_pydub(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Pitch shifting using PyDub."""
        from pydub import AudioSegment
        
        # Convert to PyDub
        audio_int = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int.tobytes()
        segment = AudioSegment(audio_bytes, frame_rate=sr, sample_width=2, channels=1)
        
        # Calculate speed change
        speed_factor = 2 ** (semitones / 12)
        
        # Apply speed change (changes pitch and duration)
        shifted_segment = segment._spawn(
            segment.raw_data, overrides={'frame_rate': int(sr * speed_factor)}
        ).set_frame_rate(sr)
        
        # Convert back
        shifted_bytes = shifted_segment.raw_data
        shifted_audio = np.frombuffer(shifted_bytes, dtype=np.int16)
        
        return shifted_audio.astype(np.float32) / 32767
    
    def pitch_shift_parselmouth(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Pitch shifting using Praat-parselmouth."""
        import parselmouth
        
        # Create Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Calculate pitch factor
        pitch_factor = 2 ** (semitones / 12)
        
        # Apply pitch change using PSOLA
        # The lengthen() method needs specific parameters
        manipulated = sound.lengthen(
            minimum_pitch=75.0,
            maximum_pitch=600.0,
            factor=pitch_factor
        )
        
        # Adjust duration back if necessary
        if len(manipulated.values[0]) != len(audio):
            # Resample to maintain original duration
            manipulated = manipulated.resample(sr)
            
        return manipulated.values[0]
    
    def pitch_shift_pedalboard(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Pitch shifting using pedalboard."""
        import pedalboard
        
        # Create pitch shift effect
        pitch_shifter = pedalboard.PitchShift(semitones=semitones)
        
        # Apply effect
        return pitch_shifter(audio, sample_rate=sr)
    
    # === NEW IMPLEMENTATIONS ===
    
    def pitch_shift_pyrubberband(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Pitch shifting using pyrubberband."""
        try:
            import pyrubberband as pyrb
            return pyrb.pitch_shift(audio, sr, n_steps=semitones)
        except Exception as e:
            if "rubberband" in str(e).lower():
                raise RuntimeError("pyrubberband requires rubberband-cli installed. "
                                 "Ubuntu/Debian: sudo apt-get install rubberband-cli")
            raise
    
    # === TIMESIDE METHOD REMOVED ===
    # TimeSide was removed due to very complex dependencies
    # Use LibROSA or other libraries for pitch shifting.
    
    # === REMOVED METHODS ===
    # aubio and Essentia were removed due to compatibility issues
    # and installation difficulties. Use LibROSA for pitch detection.
    
    def _manual_pitch_shift(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Manual pitch shift implementation using phase vocoder."""
        from scipy import signal
        
        # Parameters
        hop_length = 512
        n_fft = 2048
        
        # Pitch factor
        pitch_factor = 2 ** (semitones / 12)
        
        # STFT
        f, t, Zxx = signal.stft(audio, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        
        # Frequency shift
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Interpolate magnitude for new frequencies
        new_freqs = f * pitch_factor
        valid_mask = new_freqs < f[-1]
        
        shifted_magnitude = np.zeros_like(magnitude)
        for i, new_f in enumerate(new_freqs):
            if valid_mask[i]:
                # Linear interpolation
                idx = np.searchsorted(f, new_f)
                if idx < len(f) - 1:
                    alpha = (new_f - f[idx]) / (f[idx+1] - f[idx])
                    shifted_magnitude[i] = (1-alpha) * magnitude[idx] + alpha * magnitude[idx+1]
        
        # Reconstruct with new magnitude
        shifted_Zxx = shifted_magnitude * np.exp(1j * phase)
        
        # ISTFT
        _, shifted_audio = signal.istft(shifted_Zxx, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        
        return shifted_audio.astype(np.float32)
    
    def autotune_manual_scipy(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """Manual autotune implementation using SciPy."""
        # Detect pitch using autocorrelation
        def detect_pitch_autocorr(signal, sr):
            # Autocorrelation
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find main peak (excluding lag 0)
            min_period = int(sr / 800)  # 800 Hz maximum
            max_period = int(sr / 80)   # 80 Hz minimum
            
            if max_period < len(autocorr):
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                return sr / peak_idx
            return 0
        
        # Detect current pitch
        current_pitch = detect_pitch_autocorr(audio, sr)
        
        if current_pitch == 0:
            return audio
        
        # Calculate necessary correction
        target_pitch = current_pitch * (2 ** (semitones / 12))
        correction_factor = target_pitch / current_pitch
        correction_semitones = 12 * np.log2(correction_factor)
        
        # Apply correction
        return self._manual_pitch_shift(audio, sr, correction_semitones)
    
    def test_pitch_shift_method(self, method: Callable, method_name: str,
                               audio: np.ndarray, sr: int, 
                               semitones: float, test_case_name: str = "default") -> TestResult:
        """Tests a specific pitch shifting method with advanced monitoring."""
        
        # Monitor resources during execution
        resources = {'memory_delta': 0, 'cpu_avg': 0, 'cpu_max': 0}
        
        try:
            # Start resource monitoring
            start_time = time.time()
            
            # Apply pitch shifting
            output_audio = method(audio, sr, semitones)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Monitor resources for a time proportional to processing
            monitor_duration = max(0.1, min(processing_time * 2, 1.0))
            resources = self._monitor_resources(duration=monitor_duration)
            
            # Check if result is valid
            if output_audio is None or len(output_audio) == 0:
                raise ValueError("Method returned empty audio")
            
            # Adjust size if necessary
            if len(output_audio) != len(audio):
                if len(output_audio) > len(audio):
                    output_audio = output_audio[:len(audio)]
                else:
                    output_audio = np.pad(output_audio, (0, len(audio) - len(output_audio)), 'constant')
            
            # Calculate quality metrics
            quality_metrics = self.analyzer.calculate_quality_metrics(
                original=audio, 
                processed=output_audio, 
                sr=sr
            )
            
            error = None
            
        except Exception as e:
            processing_time = time.time() - start_time
            error = str(e)
            output_audio = None
            quality_metrics = {}
            
        finally:
            # Stop monitoring if still running
            pass
        
        return TestResult(
            library_name=method_name.split('_')[1] if '_' in method_name and len(method_name.split('_')) > 1 else method_name,
            algorithm_name=method_name,
            semitones=semitones,
            processing_time=processing_time,
            output_audio=output_audio,
            error=error,
            quality_metrics=quality_metrics,
            memory_usage=resources['memory_delta'],
            latency=processing_time,
            cpu_usage=resources['cpu_avg']
        )
    
    def run_comprehensive_test(self, test_semitones: List[float] = None, 
                             test_case_names: List[str] = None):
        """Runs comprehensive test with multiple test cases."""
        
        if test_semitones is None:
            test_semitones = [-12, -6, -3, -1, 0, 1, 3, 6, 12]
        
        if test_case_names is None:
            test_case_names = [case.name for case in self.test_cases]
        
        print("🎵 STARTING ENHANCED COMPARATIVE PITCH SHIFTING TEST")
        print("=" * 70)
        
        # Show available libraries
        print("📚 Detected libraries:")
        for lib_id, lib_info in self.libraries.items():
            status = "✅" if lib_info.available else "❌"
            external = " (requires external software)" if lib_info.requires_external else ""
            print(f"   {status} {lib_info.name} {lib_info.version}{external}")
            if not lib_info.available:
                print(f"      Error: {lib_info.import_error}")
                print(f"      Installation: {lib_info.installation_notes}")
        
        print(f"\n🎼 Available test cases:")
        for case in self.test_cases:
            if case.name in test_case_names:
                print(f"   ✅ {case.name}: {case.description}")
        
        # Define methods to test
        test_methods = []
        
        if self.libraries['librosa'].available:
            test_methods.extend([
                (self.pitch_shift_librosa, "librosa_standard"),
                (self.pitch_shift_librosa_hifi, "librosa_hifi")
            ])
        
        if self.libraries['pydub'].available:
            test_methods.append((self.pitch_shift_pydub, "pydub_speed"))
        
        if self.libraries['parselmouth'].available:
            test_methods.append((self.pitch_shift_parselmouth, "parselmouth_psola"))
        
        if self.libraries['pedalboard'].available:
            test_methods.append((self.pitch_shift_pedalboard, "pedalboard_shift"))
        
        if self.libraries['pyrubberband'].available:
            test_methods.append((self.pitch_shift_pyrubberband, "pyrubberband_shift"))
        
        # TimeSide removed
        # if self.libraries['timeside'].available:
        #     test_methods.append((self.pitch_shift_timeside, "timeside_shift"))
        
        # Removed methods: aubio and essentia
        # if self.libraries['aubio'].available:
        #     test_methods.append((self.autotune_aubio_basic, "aubio_autotune"))
        
        # if self.libraries['essentia'].available:
        #     test_methods.append((self.pitch_correct_essentia, "essentia_melodia"))
        
        if self.libraries['scipy'].available:
            test_methods.extend([
                (self._manual_pitch_shift, "scipy_manual"),
                (self.autotune_manual_scipy, "scipy_autotune")
            ])
        
        # Run tests
        total_tests = len(test_methods) * len(test_semitones) * len(test_case_names)
        current_test = 0
        
        print(f"\n🧪 Running {total_tests} tests...")
        print(f"   Methods: {len(test_methods)}")
        print(f"   Pitch values: {len(test_semitones)}")
        print(f"   Test cases: {len(test_case_names)}")
        
        for case in self.test_cases:
            if case.name not in test_case_names:
                continue
                
            print(f"\n📊 Testing case: {case.name} ({case.description})")
            
            for method_func, method_name in test_methods:
                print(f"\n   🔧 Method: {method_name}")
                
                for semitones in test_semitones:
                    current_test += 1
                    progress = current_test / total_tests * 100
                    print(f"      [{progress:5.1f}%] Pitch: {semitones:+4.1f} semitones... ", end="")
                    
                    result = self.test_pitch_shift_method(
                        method_func, method_name, case.audio, case.sr, semitones, case.name
                    )
                    
                    self.test_results.append(result)
                    
                    if result.error:
                        print(f"❌ ERROR: {result.error}")
                    else:
                        snr = result.quality_metrics.get('snr', float('nan'))
                        time_ms = result.processing_time * 1000
                        memory_mb = result.memory_usage
                        print(f"✅ SNR: {snr:5.1f}dB, {time_ms:5.1f}ms, {memory_mb:+4.1f}MB")
                        
                        # Save processed audio
                        try:
                            import soundfile as sf
                            filename = f"output_{case.name}_{method_name}_{semitones:+.1f}st.wav"
                            sf.write(filename, result.output_audio, case.sr)
                        except:
                            pass
        
        print(f"\n✅ Tests completed! {len(self.test_results)} results collected.")
    
    def generate_advanced_report(self):
        """Generates advanced test report."""
        
        print("\n📊 ENHANCED COMPARATIVE PITCH SHIFTING REPORT")
        print("=" * 70)
        
        if not self.test_results:
            print("❌ No test results available!")
            return
        
        # Group results by method
        methods = defaultdict(list)
        for result in self.test_results:
            methods[result.algorithm_name].append(result)
        
        # 1. Executive Summary
        print("\n📋 EXECUTIVE SUMMARY")
        print("-" * 40)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.error is None])
        success_rate = successful_tests / total_tests * 100
        
        print(f"Total tests: {total_tests}")
        print(f"Successful tests: {successful_tests} ({success_rate:.1f}%)")
        print(f"Libraries tested: {len(methods)}")
        print(f"Test cases: {len(set(getattr(r, 'test_case', 'default') for r in self.test_results))}")
        
        # 2. Performance Ranking
        print("\n🏃 PERFORMANCE RANKING")
        print("-" * 40)
        
        performance_data = []
        for method_name, results in methods.items():
            valid_results = [r for r in results if r.error is None]
            if valid_results:
                avg_time = np.mean([r.processing_time for r in valid_results]) * 1000
                success_rate = len(valid_results) / len(results) * 100
                avg_memory = np.mean([r.memory_usage for r in valid_results])
                performance_data.append((method_name, avg_time, success_rate, avg_memory))
        
        # Sort by time
        performance_data.sort(key=lambda x: x[1])
        
        for i, (method, avg_time, success_rate, avg_memory) in enumerate(performance_data):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            print(f"{rank:3} {method:25} {avg_time:7.1f}ms {success_rate:5.1f}% {avg_memory:+6.1f}MB")
        
        # 3. Quality Ranking
        print("\n🎵 QUALITY RANKING (Average SNR)")
        print("-" * 40)
        
        quality_data = []
        for method_name, results in methods.items():
            valid_results = [r for r in results if r.error is None and not np.isnan(r.quality_metrics.get('snr', float('nan')))]
            if valid_results:
                avg_snr = np.mean([r.quality_metrics['snr'] for r in valid_results])
                avg_thd = np.mean([r.quality_metrics.get('thd', float('nan')) for r in valid_results if not np.isnan(r.quality_metrics.get('thd', float('nan')))])
                avg_phase = np.mean([r.quality_metrics.get('phase_coherence', float('nan')) for r in valid_results if not np.isnan(r.quality_metrics.get('phase_coherence', float('nan')))])
                quality_data.append((method_name, avg_snr, avg_thd, avg_phase))
        
        # Sort by SNR
        quality_data.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method, avg_snr, avg_thd, avg_phase) in enumerate(quality_data):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            thd_str = f"{avg_thd:5.1f}dB" if not np.isnan(avg_thd) else "  N/A  "
            phase_str = f"{avg_phase:4.2f}" if not np.isnan(avg_phase) else " N/A"
            print(f"{rank:3} {method:25} SNR:{avg_snr:6.1f}dB THD:{thd_str} Phase:{phase_str}")
        
        # 4. Robustness Analysis
        print("\n🛡️  ROBUSTNESS ANALYSIS")
        print("-" * 40)
        
        robustness_data = []
        for method_name, results in methods.items():
            # Group by test case
            case_success = defaultdict(list)
            for result in results:
                case_name = getattr(result, 'test_case', 'default')
                case_success[case_name].append(result.error is None)
            
            # Calculate success rate per case
            case_rates = []
            for case_name, successes in case_success.items():
                rate = sum(successes) / len(successes) * 100
                case_rates.append(rate)
            
            if case_rates:
                avg_robustness = np.mean(case_rates)
                min_robustness = np.min(case_rates)
                robustness_data.append((method_name, avg_robustness, min_robustness))
        
        robustness_data.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method, avg_rob, min_rob) in enumerate(robustness_data):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            status = "🟢" if avg_rob >= 90 else "🟡" if avg_rob >= 70 else "🔴"
            print(f"{rank:3} {method:25} Avg:{avg_rob:5.1f}% Min:{min_rob:5.1f}% {status}")
        
        # 5. Resource Efficiency Analysis
        print("\n💾 RESOURCE EFFICIENCY")
        print("-" * 40)
        
        resource_data = []
        for method_name, results in methods.items():
            valid_results = [r for r in results if r.error is None]
            if valid_results:
                avg_memory = np.mean([r.memory_usage for r in valid_results])
                avg_cpu = np.mean([r.cpu_usage for r in valid_results if not np.isnan(r.cpu_usage)])
                
                # Avoid division by zero and NaN
                if np.isnan(avg_cpu):
                    avg_cpu = 0.0
                if np.isnan(avg_memory):
                    avg_memory = 0.0
                    
                efficiency_score = 100 / (1 + abs(avg_memory) + max(0, avg_cpu)/10)  # Arbitrary score
                resource_data.append((method_name, avg_memory, avg_cpu, efficiency_score))
        
        resource_data.sort(key=lambda x: x[3], reverse=True)
        
        for i, (method, memory, cpu, score) in enumerate(resource_data):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            print(f"{rank:3} {method:25} Mem:{memory:+6.1f}MB CPU:{cpu:5.1f}% Score:{score:5.1f}")
        
        # 6. Advanced Recommendations
        print("\n💡 ADVANCED RECOMMENDATIONS")
        print("-" * 40)
        
        if performance_data and quality_data and robustness_data:
            # Best for real-time
            fastest = performance_data[0][0]
            print(f"⚡ Real-time/low latency: {fastest}")
            
            # Best quality
            best_quality = quality_data[0][0]
            print(f"🎵 Maximum audio quality: {best_quality}")
            
            # Most robust
            most_robust = robustness_data[0][0]
            print(f"🛡️  Highest robustness: {most_robust}")
            
            # Most efficient
            if resource_data:
                most_efficient = resource_data[0][0]
                print(f"💾 Highest efficiency: {most_efficient}")
            
            # Combined score
            combined_scores = defaultdict(float)
            
            # Performance score (weight 2)
            for i, (method, _, _, _) in enumerate(performance_data):
                combined_scores[method] += (len(performance_data) - i) * 2
            
            # Quality score (weight 3)
            for i, (method, _, _, _) in enumerate(quality_data):
                combined_scores[method] += (len(quality_data) - i) * 3
            
            # Robustness score (weight 2)
            for i, (method, _, _) in enumerate(robustness_data):
                combined_scores[method] += (len(robustness_data) - i) * 2
            
            # Efficiency score (weight 1)
            if resource_data:
                for i, (method, _, _, _) in enumerate(resource_data):
                    combined_scores[method] += (len(resource_data) - i) * 1
            
            best_overall = max(combined_scores.items(), key=lambda x: x[1])
            print(f"⭐ Best overall choice: {best_overall[0]} (score: {best_overall[1]})")
            
            print(f"\n🎯 SELECTION GUIDE:")
            print(f"   • Real-time applications: {fastest}")
            print(f"   • Professional music production: {best_quality}")
            print(f"   • Robust systems/production: {most_robust}")
            if resource_data:
                print(f"   • Resource-limited devices: {most_efficient}")
            print(f"   • General use: {fastest}")
        
        # Return structured data
        return {
            'performance': performance_data,
            'quality': quality_data,
            'robustness': robustness_data,
            'resources': resource_data,
            'recommendations': {
                'fastest': fastest if performance_data else None,
                'best_quality': best_quality if quality_data else None,
                'most_robust': most_robust if robustness_data else None,
                'most_efficient': most_efficient if resource_data else None
            }
        }
    
    def create_advanced_plots(self):
        """Creates advanced plots of test results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            
            print("\n📈 CREATING ADVANCED PLOTS")
            print("=" * 50)
            
            # Configure style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Organize data
            methods = {}
            for result in self.test_results:
                if result.error is None:
                    if result.algorithm_name not in methods:
                        methods[result.algorithm_name] = []
                    methods[result.algorithm_name].append(result)
            
            if not methods:
                print("❌ No valid results to plot!")
                return
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle('Comparative Analysis of Pitch Shifting Libraries', fontsize=16, fontweight='bold')
            
            # 1. Performance Plot (Time vs SNR)
            ax1 = plt.subplot(3, 3, 1)
            for method_name, results in methods.items():
                times = [r.processing_time * 1000 for r in results]  # ms
                snrs = [r.quality_metrics.get('snr', 0) for r in results]
                ax1.scatter(times, snrs, label=method_name, alpha=0.7, s=50)
            
            ax1.set_xlabel('Processing Time (ms)')
            ax1.set_ylabel('SNR (dB)')
            ax1.set_title('Performance: Time vs Quality')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. SNR Boxplot by method
            ax2 = plt.subplot(3, 3, 2)
            snr_data = []
            method_labels = []
            for method_name, results in methods.items():
                snrs = [r.quality_metrics.get('snr', 0) for r in results if not np.isnan(r.quality_metrics.get('snr', np.nan))]
                if snrs:
                    snr_data.extend(snrs)
                    method_labels.extend([method_name] * len(snrs))
            
            if snr_data:
                import pandas as pd
                df_snr = pd.DataFrame({'SNR': snr_data, 'Method': method_labels})
                sns.boxplot(data=df_snr, x='Method', y='SNR', ax=ax2)
                ax2.set_title('SNR Distribution by Method')
                ax2.tick_params(axis='x', rotation=45)
            
            # 3. Bar chart - Average time
            ax3 = plt.subplot(3, 3, 3)
            avg_times = []
            method_names = []
            for method_name, results in methods.items():
                times = [r.processing_time * 1000 for r in results]
                if times:
                    avg_times.append(np.mean(times))
                    method_names.append(method_name)
            
            if avg_times:
                bars = ax3.bar(range(len(avg_times)), avg_times, color=sns.color_palette("husl", len(avg_times)))
                ax3.set_xlabel('Method')
                ax3.set_ylabel('Average Time (ms)')
                ax3.set_title('Average Processing Time')
                ax3.set_xticks(range(len(method_names)))
                ax3.set_xticklabels(method_names, rotation=45, ha='right')
                
                # Add values on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
            
            # 4. Quality heatmap by semitone
            ax4 = plt.subplot(3, 3, 4)
            semitones_unique = sorted(list(set([r.semitones for r in self.test_results])))
            method_names_sorted = sorted(methods.keys())
            
            heatmap_data = np.full((len(method_names_sorted), len(semitones_unique)), np.nan)
            
            for i, method in enumerate(method_names_sorted):
                for j, semitone in enumerate(semitones_unique):
                    matching_results = [r for r in methods[method] if r.semitones == semitone]
                    if matching_results:
                        snrs = [r.quality_metrics.get('snr', np.nan) for r in matching_results]
                        valid_snrs = [s for s in snrs if not np.isnan(s)]
                        if valid_snrs:
                            heatmap_data[i, j] = np.mean(valid_snrs)
            
            im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', interpolation='nearest')
            ax4.set_xticks(range(len(semitones_unique)))
            ax4.set_xticklabels([f'{s:+.0f}' for s in semitones_unique])
            ax4.set_yticks(range(len(method_names_sorted)))
            ax4.set_yticklabels(method_names_sorted)
            ax4.set_xlabel('Semitones')
            ax4.set_ylabel('Method')
            ax4.set_title('SNR by Method and Semitone')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('SNR (dB)')
            
            # 5. Resource plot (CPU vs Memory)
            ax5 = plt.subplot(3, 3, 5)
            for method_name, results in methods.items():
                cpu_usage = [r.cpu_usage for r in results if r.cpu_usage > 0]
                memory_usage = [r.memory_usage for r in results if r.memory_usage != 0]
                
                if cpu_usage and memory_usage:
                    # Get average values if data exists
                    avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
                    avg_memory = np.mean(memory_usage) if memory_usage else 0
                    ax5.scatter(avg_cpu, avg_memory, label=method_name, s=100, alpha=0.7)
            
            ax5.set_xlabel('Average CPU (%)')
            ax5.set_ylabel('Average Memory (MB)')
            ax5.set_title('Resource Usage')
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax5.grid(True, alpha=0.3)
            
            # 6. Line plot - SNR vs Semitone
            ax6 = plt.subplot(3, 3, 6)
            for method_name, results in methods.items():
                semitone_snr = {}
                for result in results:
                    semitone = result.semitones
                    snr = result.quality_metrics.get('snr', np.nan)
                    if not np.isnan(snr):
                        if semitone not in semitone_snr:
                            semitone_snr[semitone] = []
                        semitone_snr[semitone].append(snr)
                
                if semitone_snr:
                    semitones_sorted = sorted(semitone_snr.keys())
                    avg_snrs = [np.mean(semitone_snr[s]) for s in semitones_sorted]
                    ax6.plot(semitones_sorted, avg_snrs, marker='o', label=method_name, linewidth=2, markersize=4)
            
            ax6.set_xlabel('Semitones')
            ax6.set_ylabel('Average SNR (dB)')
            ax6.set_title('Quality vs Pitch Shift')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # 7. Time distribution histogram
            ax7 = plt.subplot(3, 3, 7)
            all_times = []
            for method_name, results in methods.items():
                times = [r.processing_time * 1000 for r in results]
                all_times.extend(times)
            
            if all_times:
                ax7.hist(all_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax7.set_xlabel('Processing Time (ms)')
                ax7.set_ylabel('Frequency')
                ax7.set_title('Time Distribution')
                ax7.axvline(np.mean(all_times), color='red', linestyle='--', label=f'Mean: {np.mean(all_times):.1f}ms')
                ax7.legend()
            
            # 8. Radar plot - Multiple metrics
            ax8 = plt.subplot(3, 3, 8, projection='polar')
            
            # Select metrics for radar
            metrics = ['snr', 'spectral_stability', 'phase_coherence']
            available_metrics = []
            
            # Check which metrics are available
            for metric in metrics:
                has_data = any(
                    any(not np.isnan(r.quality_metrics.get(metric, np.nan)) for r in results)
                    for results in methods.values()
                )
                if has_data:
                    available_metrics.append(metric)
            
            if available_metrics and len(available_metrics) >= 3:
                angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
                angles += angles[:1]  # Close the circle
                
                for method_name, results in list(methods.items())[:3]:  # Maximum 3 methods for clarity
                    values = []
                    for metric in available_metrics:
                        metric_values = [r.quality_metrics.get(metric, np.nan) for r in results]
                        valid_values = [v for v in metric_values if not np.isnan(v)]
                        if valid_values:
                            # Normalize to 0-1
                            avg_value = np.mean(valid_values)
                            if metric == 'snr':
                                normalized = max(0, min(1, (avg_value + 10) / 20))  # SNR -10 to +10 -> 0 to 1
                            else:
                                normalized = max(0, min(1, avg_value))
                            values.append(normalized)
                        else:
                            values.append(0)
                    
                    values += values[:1]  # Close the circle
                    ax8.plot(angles, values, 'o-', linewidth=2, label=method_name)
                    ax8.fill(angles, values, alpha=0.25)
                
                ax8.set_xticks(angles[:-1])
                ax8.set_xticklabels(available_metrics)
                ax8.set_ylim(0, 1)
                ax8.set_title('Quality Profile (Normalized)')
                ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            else:
                ax8.text(0.5, 0.5, 'Insufficient data\nfor radar plot', 
                        transform=ax8.transAxes, ha='center', va='center')
                ax8.set_title('Quality Profile')
            
            # 9. Efficiency plot (Composite score)
            ax9 = plt.subplot(3, 3, 9)
            efficiency_scores = []
            method_names_eff = []
            
            for method_name, results in methods.items():
                valid_results = [r for r in results if r.error is None]
                if valid_results:
                    # Calculate efficiency score
                    avg_time = np.mean([r.processing_time for r in valid_results])
                    avg_snr = np.mean([r.quality_metrics.get('snr', 0) for r in valid_results])
                    avg_memory = np.mean([abs(r.memory_usage) for r in valid_results])
                    avg_cpu = np.mean([r.cpu_usage for r in valid_results])
                    
                    # Composite score (higher is better)
                    time_score = max(0, 100 - avg_time * 1000)  # Penalize high time
                    quality_score = max(0, min(100, (avg_snr + 10) * 5))  # SNR -10 to +10 -> 0 to 100
                    resource_score = max(0, 100 - avg_memory * 10 - avg_cpu)  # Penalize resource usage
                    
                    composite_score = (time_score + quality_score + resource_score) / 3
                    
                    efficiency_scores.append(composite_score)
                    method_names_eff.append(method_name)
            
            if efficiency_scores:
                bars = ax9.bar(range(len(efficiency_scores)), efficiency_scores, 
                              color=sns.color_palette("viridis", len(efficiency_scores)))
                ax9.set_xlabel('Method')
                ax9.set_ylabel('Efficiency Score')
                ax9.set_title('Composite Efficiency Score')
                ax9.set_xticks(range(len(method_names_eff)))
                ax9.set_xticklabels(method_names_eff, rotation=45, ha='right')
                ax9.set_ylim(0, 100)
                
                # Adiciona valores nas barras
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax9.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.0f}', ha='center', va='bottom', fontsize=8)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pitch_shifting_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
            print(f"✅ Plots saved in: {filename}")
            print(f"📊 {len(methods)} methods analyzed")
            print(f"📈 {len(self.test_results)} results processed")
            
            # Show plot if possible
            try:
                plt.show()
            except:
                print("💡 Plot saved to file (display not available)")
            
        except ImportError as e:
            print(f"❌ Error: Visualization libraries not available: {e}")
            print("💡 Install with: pip install matplotlib seaborn pandas")
        except Exception as e:
            print(f"❌ Error creating plots: {e}")
    
    def save_detailed_results(self, filename: str = None):
        """Saves detailed results in JSON."""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pitch_shift_results_enhanced_{timestamp}.json"
        
        results_data = {
            "metadata": {
                "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": len(self.test_results),
                "libraries_tested": len([lib for lib in self.libraries.values() if lib.available]),
                "test_cases": [{"name": case.name, "description": case.description} for case in self.test_cases],
                "version": "2.0.0"
            },
            "libraries": {
                lib_id: {
                    "name": lib_info.name,
                    "version": lib_info.version,
                    "available": lib_info.available,
                    "import_error": lib_info.import_error,
                    "algorithms": lib_info.algorithms,
                    "installation_notes": lib_info.installation_notes,
                    "requires_external": lib_info.requires_external
                }
                for lib_id, lib_info in self.libraries.items()
            },
            "results": []
        }
        
        for result in self.test_results:
            result_dict = {
                "library_name": result.library_name,
                "algorithm_name": result.algorithm_name,
                "semitones": result.semitones,
                "processing_time": result.processing_time,
                "error": result.error,
                "quality_metrics": result.quality_metrics,
                "memory_usage": result.memory_usage,
                "latency": result.latency,
                "cpu_usage": result.cpu_usage
            }
            results_data["results"].append(result_dict)
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"💾 Detailed results saved in: {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def install_missing_libraries_enhanced():
    """Automatically installs missing libraries (enhanced version)."""
    
    print("🔧 AUTOMATIC LIBRARY INSTALLATION (ENHANCED VERSION)")
    print("=" * 60)
    
    libraries_to_install = [
        ("librosa", "librosa resampy", False),  # resampy required for LibROSA
        ("pydub", "pydub", False),
        ("parselmouth", "praat-parselmouth", False),
        ("pedalboard", "pedalboard", False),
        ("pyrubberband", "pyrubberband", True),  # Requires external software
        # TimeSide removed due to complex dependencies:
        # ("timeside", "TimeSide", False),
        # Libraries removed due to compatibility issues:
        # ("aubio", "aubio", False),
        # ("essentia", "essentia", False),
        ("matplotlib", "matplotlib seaborn", False),
        ("scipy", "scipy", False),
        ("numpy", "numpy", False),
        ("psutil", "psutil", False)
    ]
    
    missing_libraries = []
    external_requirements = []
    
    # Check which ones are missing
    for lib_name, pip_name, requires_external in libraries_to_install:
        try:
            __import__(lib_name)
            print(f"✅ {lib_name} already installed")
        except ImportError:
            print(f"❌ {lib_name} not found")
            missing_libraries.append((lib_name, pip_name, requires_external))
            if requires_external:
                external_requirements.append(lib_name)
    
    if not missing_libraries:
        print("🎉 All libraries are installed!")
        return True
    
    print(f"\n📦 {len(missing_libraries)} libraries need to be installed:")
    for lib_name, pip_name, requires_external in missing_libraries:
        external_note = " (requires external software)" if requires_external else ""
        print(f"   • {lib_name} (pip install {pip_name}){external_note}")
    
    if external_requirements:
        print(f"\n⚠️  External requirements:")
        for lib in external_requirements:
            if lib == "pyrubberband":
                print(f"   • {lib}: requires 'rubberband-cli'")
                print(f"     Ubuntu/Debian: sudo apt-get install rubberband-cli")
                print(f"     macOS: brew install rubberband")
                print(f"     Windows: download from https://breakfastquay.com/rubberband/")
    
    try:
        response = input(f"\n❓ Do you want to install automatically? (y/N): ").lower()
        if response in ['s', 'sim', 'y', 'yes']:
            import subprocess
            import sys
            
            print("\n🚀 Installing libraries...")
            
            for lib_name, pip_name, requires_external in missing_libraries:
                print(f"   📦 Installing {lib_name}...")
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", pip_name
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print(f"   ✅ {lib_name} installed successfully")
                    else:
                        print(f"   ❌ Error installing {lib_name}: {result.stderr}")
                        if requires_external:
                            print(f"   💡 Check if external requirements are installed")
                        
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Timeout installing {lib_name}")
                except Exception as e:
                    print(f"   ❌ Unexpected error installing {lib_name}: {e}")
            
            print("\n✅ Installation completed!")
            if external_requirements:
                print("⚠️  Remember to install the external requirements mentioned above.")
            print("🔄 Restart the script to use the new libraries.")
            return True
        else:
            print("⚠️  Installation cancelled. Some libraries may not be available.")
            return False
            
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️  Installation cancelled by user.")
        return False


def main():
    """Main function of the enhanced script."""
    
    print("🎵" + "=" * 68 + "🎵")
    print("🎵 ENHANCED COMPARISON OF PITCH SHIFTING AND AUTOTUNE LIBRARIES 🎵")
    print("🎵" + "=" * 68 + "🎵")
    print()
    print("This enhanced script tests and compares different Python libraries")
    print("for pitch shifting and autotune, including:")
    print("  • Traditional libraries (LibROSA, PyDub, Parselmouth, etc.)")
    print("  • New libraries (pyrubberband)")
    print("  • Custom autotune implementations")
    print("  • Advanced quality and performance metrics")
    print("  • Robustness analysis with multiple test cases")
    print("  • Resource monitoring (CPU, memory)")
    print()
    
    # Check basic dependencies
    try:
        import numpy
        import scipy
    except ImportError:
        print("❌ NumPy and SciPy are required!")
        print("💡 Run: pip install numpy scipy")
        return
    
    # Option to install missing libraries
    try:
        choice = input("❓ Check and install missing libraries? (Y/n): ").lower()
        if choice not in ['n', 'no', 'não']:
            install_missing_libraries_enhanced()
            print()
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️  Proceeding without dependency check...")
    
    # Initialize tester
    tester = EnhancedPitchShiftingTester()
    
    # Show menu options
    while True:
        try:
            print("\n📋 ENHANCED TEST OPTIONS:")
            print("1. 🧪 Quick test (±1, ±3, ±6 semitones, basic cases)")
            print("2. 🔬 Complete test (±12 semitones, all cases)")
            print("3. 🎯 Custom test (choose semitones and cases)")
            print("4. 🎼 Robustness test (multiple audio cases)")
            print("5. 📊 View previous results")
            print("6. 💾 Save detailed results")
            print("7. 📈 Create advanced graphs")
            print("8. 🔍 Detailed comparative analysis")
            print("9. ❌ Exit")
            
            choice = input("\n❓ Choose an option (1-9): ").strip()
            
            if choice == "1":
                print("\n🧪 Running quick test...")
                tester.run_comprehensive_test(
                    test_semitones=[-6, -3, -1, 0, 1, 3, 6],
                    test_case_names=["pure_tone", "harmonic_tone"]
                )
                tester.generate_advanced_report()
                
            elif choice == "2":
                print("\n🔬 Running complete test...")
                tester.run_comprehensive_test(
                    test_semitones=[-12, -9, -6, -3, -1, 0, 1, 3, 6, 9, 12]
                )
                tester.generate_advanced_report()
                
            elif choice == "3":
                try:
                    print("\n🎯 Custom test")
                    semitones_input = input("Enter semitones separated by comma (ex: -12,-6,0,6,12): ")
                    custom_semitones = [float(s.strip()) for s in semitones_input.split(',')]
                    
                    print("Available test cases:")
                    for i, case in enumerate(tester.test_cases):
                        print(f"   {i+1}. {case.name}: {case.description}")
                    
                    cases_input = input("Enter case numbers separated by comma (ex: 1,2,3) or 'all': ")
                    if cases_input.lower() == 'all':
                        custom_cases = [case.name for case in tester.test_cases]
                    else:
                        case_indices = [int(i.strip()) - 1 for i in cases_input.split(',')]
                        custom_cases = [tester.test_cases[i].name for i in case_indices if 0 <= i < len(tester.test_cases)]
                    
                    print(f"Testing with semitones: {custom_semitones}")
                    print(f"Test cases: {custom_cases}")
                    
                    tester.run_comprehensive_test(
                        test_semitones=custom_semitones,
                        test_case_names=custom_cases
                    )
                    tester.generate_advanced_report()
                    
                except ValueError:
                    print("❌ Invalid input! Use numbers separated by comma.")
                    
            elif choice == "4":
                print("\n🎼 Running robustness test...")
                tester.run_comprehensive_test(
                    test_semitones=[-6, -3, 0, 3, 6],
                    test_case_names=[case.name for case in tester.test_cases]
                )
                tester.generate_advanced_report()
                
            elif choice == "5":
                if tester.test_results:
                    tester.generate_advanced_report()
                else:
                    print("❌ No test has been run yet!")
                    
            elif choice == "6":
                if tester.test_results:
                    filename = input("File name (default: pitch_shift_results_enhanced.json): ").strip()
                    if not filename:
                        filename = "pitch_shift_results_enhanced.json"
                    tester.save_detailed_results(filename)
                else:
                    print("❌ No results to save!")
                    
            elif choice == "7":
                if tester.test_results:
                    tester.create_advanced_plots()
                else:
                    print("❌ No results to visualize!")
                    
            elif choice == "8":
                if tester.test_results:
                    print("🔍 Detailed comparative analysis will be implemented...")
                    print("💡 Use the current advanced report for detailed analysis.")
                else:
                    print("❌ No results to analyze!")
                    
            elif choice == "9":
                print("👋 Thank you for using the enhanced pitch shifting comparator!")
                break
                
            else:
                print("❌ Invalid option! Choose from 1 to 9.")
                
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

