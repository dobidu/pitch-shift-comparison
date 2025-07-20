#!/usr/bin/env python3
"""
Advanced Audio Quality Metrics Module for Audio Analysis
=======================================================

This module provides specialized metrics for evaluating the quality
of pitch shifting and autotune algorithms.

Included metrics:
- Pitch accuracy analysis
- Processing artifact measurements
- Timbre preservation analysis
- Vocal naturalness metrics
- Advanced spectral analysis
- Perceptual measures

Author: Advanced Audio Quality Analysis
Version: 1.0.0
"""

import numpy as np
import scipy.signal
import scipy.stats
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class PitchAnalyzer:
    """Specialized pitch and tuning analyzer."""
    
    @staticmethod
    def detect_pitch_yin(audio: np.ndarray, sr: int, frame_length: int = 2048) -> List[float]:
        """
        Detects pitch using YIN algorithm.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            frame_length: Analysis window size
            
        Returns:
            List of pitch values in Hz
        """
        def yin_pitch_detection(frame):
            # Difference function
            diff_function = np.zeros(frame_length // 2)
            for tau in range(1, frame_length // 2):
                diff_function[tau] = np.sum((frame[:-tau] - frame[tau:]) ** 2)
            
            # Normalized cumulative difference function
            cmnd = np.zeros_like(diff_function)
            cmnd[0] = 1
            for tau in range(1, len(diff_function)):
                cmnd[tau] = diff_function[tau] / (np.sum(diff_function[1:tau+1]) / tau)
            
            # Find first minimum below threshold
            threshold = 0.1
            for tau in range(1, len(cmnd)):
                if cmnd[tau] < threshold:
                    # Parabolic interpolation for higher precision
                    if tau < len(cmnd) - 1:
                        x0, x1, x2 = tau - 1, tau, tau + 1
                        y0, y1, y2 = cmnd[x0], cmnd[x1], cmnd[x2]
                        
                        # Parabolic interpolation
                        a = (y0 - 2*y1 + y2) / 2
                        b = (y2 - y0) / 2
                        
                        if a != 0:
                            tau_precise = tau - b / (2 * a)
                        else:
                            tau_precise = tau
                        
                        return sr / tau_precise
            
            return 0.0  # Pitch not detected
        
        pitches = []
        hop_length = frame_length // 4
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            pitch = yin_pitch_detection(frame)
            pitches.append(pitch)
        
        return pitches
    
    @staticmethod
    def calculate_pitch_stability(pitches: List[float], threshold_cents: float = 50) -> Dict[str, float]:
        """
        Calculates pitch stability metrics.
        
        Args:
            pitches: List of pitch values
            threshold_cents: Threshold in cents to consider pitch stable
            
        Returns:
            Dictionary with stability metrics
        """
        valid_pitches = [p for p in pitches if p > 0]
        
        if len(valid_pitches) < 2:
            return {
                'stability_ratio': 0.0,
                'pitch_variance_cents': float('inf'),
                'median_pitch': 0.0,
                'pitch_range_cents': 0.0
            }
        
        median_pitch = np.median(valid_pitches)
        
        # Convert to cents
        cents_deviations = [1200 * np.log2(p / median_pitch) for p in valid_pitches]
        
        # Calculate stability
        stable_frames = sum(1 for dev in cents_deviations if abs(dev) <= threshold_cents)
        stability_ratio = stable_frames / len(cents_deviations)
        
        return {
            'stability_ratio': stability_ratio,
            'pitch_variance_cents': float(np.var(cents_deviations)),
            'median_pitch': float(median_pitch),
            'pitch_range_cents': float(max(cents_deviations) - min(cents_deviations))
        }
    
    @staticmethod
    def calculate_pitch_accuracy(target_pitch: float, detected_pitches: List[float]) -> Dict[str, float]:
        """
        Calculates pitch accuracy relative to a target.
        
        Args:
            target_pitch: Target pitch in Hz
            detected_pitches: List of detected pitches
            
        Returns:
            Accuracy metrics
        """
        valid_pitches = [p for p in detected_pitches if p > 0]
        
        if not valid_pitches:
            return {
                'mean_error_cents': float('inf'),
                'std_error_cents': float('inf'),
                'max_error_cents': float('inf'),
                'accuracy_percentage': 0.0
            }
        
        # Calculate errors in cents
        errors_cents = [1200 * np.log2(p / target_pitch) for p in valid_pitches]
        
        # Calculate accuracy (frames within ±20 cents)
        accurate_frames = sum(1 for error in errors_cents if abs(error) <= 20)
        accuracy_percentage = accurate_frames / len(errors_cents) * 100
        
        return {
            'mean_error_cents': float(np.mean(errors_cents)),
            'std_error_cents': float(np.std(errors_cents)),
            'max_error_cents': float(max(abs(e) for e in errors_cents)),
            'accuracy_percentage': accuracy_percentage
        }

class SpectralAnalyzer:
    """Advanced spectral analyzer."""
    
    @staticmethod
    def calculate_spectral_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculates spectral characteristics of audio.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            
        Returns:
            Dictionary with spectral characteristics
        """
        # FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Consider only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Normalize magnitude
        positive_magnitude = positive_magnitude / np.sum(positive_magnitude)
        
        # Spectral centroid
        spectral_centroid = np.sum(positive_freqs * positive_magnitude)
        
        # Spectral width
        spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * positive_magnitude))
        
        # Spectral skewness
        spectral_skewness = np.sum(((positive_freqs - spectral_centroid) ** 3) * positive_magnitude) / (spectral_spread ** 3)
        
        # Spectral kurtosis
        spectral_kurtosis = np.sum(((positive_freqs - spectral_centroid) ** 4) * positive_magnitude) / (spectral_spread ** 4)
        
        # Spectral rolloff (95%)
        cumsum_magnitude = np.cumsum(positive_magnitude)
        rolloff_idx = np.where(cumsum_magnitude >= 0.95 * cumsum_magnitude[-1])[0]
        spectral_rolloff = positive_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else positive_freqs[-1]
        
        # Spectral flux (difference between consecutive frames)
        spectral_flux = 0.0
        if len(audio) > 2048:
            frame1 = np.abs(np.fft.fft(audio[:2048]))[:1024]
            frame2 = np.abs(np.fft.fft(audio[1024:3072]))[:1024]
            spectral_flux = float(np.sum((frame2 - frame1) ** 2))
        
        return {
            'spectral_centroid': float(spectral_centroid),
            'spectral_spread': float(spectral_spread),
            'spectral_skewness': float(spectral_skewness),
            'spectral_kurtosis': float(spectral_kurtosis),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_flux': spectral_flux
        }
    
    @staticmethod
    def calculate_harmonic_content(audio: np.ndarray, sr: int, fundamental_freq: float) -> Dict[str, float]:
        """
        Analyzes harmonic content of the signal.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            fundamental_freq: Expected fundamental frequency
            
        Returns:
            Harmonic content metrics
        """
        # FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Find harmonic peaks
        harmonic_powers = []
        harmonic_freqs = []
        
        for h in range(1, 11):  # Up to 10th harmonic
            target_freq = fundamental_freq * h
            
            # Find closest index
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            
            # Consider a window around the harmonic
            window_size = max(1, int(sr / fundamental_freq / 20))  # ±5% of fundamental frequency
            start_idx = max(0, freq_idx - window_size)
            end_idx = min(len(magnitude), freq_idx + window_size)
            
            # Find peak within the window
            window_magnitude = magnitude[start_idx:end_idx]
            if len(window_magnitude) > 0:
                peak_idx = np.argmax(window_magnitude)
                peak_power = window_magnitude[peak_idx] ** 2
                peak_freq = freqs[start_idx + peak_idx]
                
                harmonic_powers.append(peak_power)
                harmonic_freqs.append(peak_freq)
            else:
                harmonic_powers.append(0.0)
                harmonic_freqs.append(target_freq)
        
        # Calculate metrics
        total_power = sum(harmonic_powers)
        fundamental_power = harmonic_powers[0] if harmonic_powers else 0
        
        if total_power > 0:
            harmonic_ratio = fundamental_power / total_power
            
            # THD (Total Harmonic Distortion)
            harmonic_distortion_power = sum(harmonic_powers[1:])
            thd = harmonic_distortion_power / fundamental_power if fundamental_power > 0 else float('inf')
            
            # Inharmonicity (deviation of harmonic frequencies from ideal)
            inharmonicity = 0.0
            for i, (detected_freq, expected_freq) in enumerate(zip(harmonic_freqs, [fundamental_freq * (i+1) for i in range(len(harmonic_freqs))])):
                if expected_freq > 0:
                    inharmonicity += abs(detected_freq - expected_freq) / expected_freq
            inharmonicity /= len(harmonic_freqs)
            
        else:
            harmonic_ratio = 0.0
            thd = float('inf')
            inharmonicity = float('inf')
        
        return {
            'harmonic_ratio': float(harmonic_ratio),
            'thd': float(thd),
            'inharmonicity': float(inharmonicity),
            'harmonic_powers': [float(p) for p in harmonic_powers],
            'total_harmonic_power': float(total_power)
        }

class ArtifactDetector:
    """Artifact detector."""
    
    @staticmethod
    def detect_aliasing(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Detects aliasing artifacts.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            
        Returns:
            Aliasing metrics
        """
        # FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Consider only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Analyze energy in high frequencies (above sr/4)
        nyquist = sr / 2
        high_freq_threshold = nyquist * 0.75
        
        high_freq_mask = positive_freqs >= high_freq_threshold
        low_freq_mask = positive_freqs < high_freq_threshold
        
        high_freq_energy = np.sum(positive_magnitude[high_freq_mask] ** 2)
        total_energy = np.sum(positive_magnitude ** 2)
        
        aliasing_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Detect spurious peaks in high frequencies
        if np.any(high_freq_mask):
            high_freq_peaks = scipy.signal.find_peaks(positive_magnitude[high_freq_mask], height=np.max(positive_magnitude) * 0.1)[0]
            spurious_peaks = len(high_freq_peaks)
        else:
            spurious_peaks = 0
        
        return {
            'aliasing_ratio': float(aliasing_ratio),
            'spurious_peaks': int(spurious_peaks),
            'high_freq_energy': float(high_freq_energy)
        }
    
    @staticmethod
    def detect_clicks_pops(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Detects clicks and pops in audio.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            
        Returns:
            Clicks and pops metrics
        """
        # Calculate differences between consecutive samples
        diff = np.diff(audio)
        
        # Detect abrupt changes
        threshold = np.std(diff) * 5  # 5 standard deviations
        clicks = np.abs(diff) > threshold
        
        # Count clicks
        click_count = np.sum(clicks)
        click_rate = click_count / (len(audio) / sr)  # clicks per second
        
        # Calculate energy of clicks
        click_energy = np.sum(diff[clicks] ** 2) if click_count > 0 else 0
        total_energy = np.sum(diff ** 2)
        click_energy_ratio = click_energy / total_energy if total_energy > 0 else 0
        
        return {
            'click_count': int(click_count),
            'click_rate': float(click_rate),
            'click_energy_ratio': float(click_energy_ratio)
        }
    
    @staticmethod
    def detect_modulation_artifacts(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Detects modulation artifacts (undesired tremolo, vibrato).
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            
        Returns:
            Modulation artifact metrics
        """
        # Calculate amplitude envelope
        analytic_signal = scipy.signal.hilbert(audio)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Remove DC component
        amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
        
        # FFT of envelope
        envelope_fft = np.fft.fft(amplitude_envelope)
        envelope_magnitude = np.abs(envelope_fft)
        envelope_freqs = np.fft.fftfreq(len(amplitude_envelope), 1/sr)
        
        # Consider only low positive frequencies (0-20 Hz for modulation)
        low_freq_mask = (envelope_freqs >= 0) & (envelope_freqs <= 20)
        low_freq_magnitude = envelope_magnitude[low_freq_mask]
        low_freq_freqs = envelope_freqs[low_freq_mask]
        
        # Detect modulation peaks
        if len(low_freq_magnitude) > 1:
            peaks, _ = scipy.signal.find_peaks(low_freq_magnitude, height=np.max(low_freq_magnitude) * 0.1)
            
            modulation_strength = 0.0
            dominant_mod_freq = 0.0
            
            if len(peaks) > 0:
                # Find dominant peak (excluding DC)
                non_dc_peaks = peaks[low_freq_freqs[peaks] > 0.5]  # Exclude very low frequencies
                if len(non_dc_peaks) > 0:
                    dominant_peak_idx = non_dc_peaks[np.argmax(low_freq_magnitude[non_dc_peaks])]
                    dominant_mod_freq = low_freq_freqs[dominant_peak_idx]
                    modulation_strength = low_freq_magnitude[dominant_peak_idx] / np.sum(low_freq_magnitude)
        else:
            modulation_strength = 0.0
            dominant_mod_freq = 0.0
        
        return {
            'modulation_strength': float(modulation_strength),
            'dominant_modulation_freq': float(dominant_mod_freq),
            'amplitude_variance': float(np.var(amplitude_envelope))
        }

class PerceptualAnalyzer:
    """Perceptual quality analyzer."""
    
    @staticmethod
    def calculate_roughness(audio: np.ndarray, sr: int) -> float:
        """
        Calculates perceptual roughness of audio.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            
        Returns:
            Roughness value
        """
        # Simplified implementation based on amplitude fluctuations
        # In a full implementation, we would use psychoacoustic models
        
        # Calculate amplitude envelope in frequency bands
        # Divide spectrum into critical bands (approximation)
        n_bands = 24
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        fft = np.fft.fft(audio)
        
        # Logarithmic bands
        min_freq = 50
        max_freq = sr / 2
        band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bands + 1)
        
        roughness_total = 0.0
        
        for i in range(n_bands):
            # Filter band
            band_mask = (np.abs(freqs) >= band_edges[i]) & (np.abs(freqs) < band_edges[i + 1])
            band_fft = fft.copy()
            band_fft[~band_mask] = 0
            
            # Reconstruct band signal
            band_signal = np.real(np.fft.ifft(band_fft))
            
            # Calculate envelope
            analytic_signal = scipy.signal.hilbert(band_signal)
            envelope = np.abs(analytic_signal)
            
            # Calculate envelope fluctuations (3-300 Hz)
            if len(envelope) > sr // 10:  # At least 0.1s of audio
                envelope_fft = np.fft.fft(envelope)
                envelope_freqs = np.fft.fftfreq(len(envelope), 1/sr)
                
                # Consider fluctuations between 3-300 Hz
                modulation_mask = (np.abs(envelope_freqs) >= 3) & (np.abs(envelope_freqs) <= 300)
                modulation_energy = np.sum(np.abs(envelope_fft[modulation_mask]) ** 2)
                
                # Band weight (approximation of auditory sensitivity)
                band_center = np.sqrt(band_edges[i] * band_edges[i + 1])
                weight = 1.0 / (1.0 + (band_center / 1000) ** 2)  # Larger weight for medium frequencies
                
                roughness_total += modulation_energy * weight
        
        return float(roughness_total)
    
    @staticmethod
    def calculate_sharpness(audio: np.ndarray, sr: int) -> float:
        """
        Calculates perceptual sharpness of audio.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            
        Returns:
            Sharpness value
        """
        # FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        
        # Consider only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Weight based on frequency (sharpness increases with frequency)
        # Simple approximation: logarithmic weight
        weights = np.log10(positive_freqs + 1)  # +1 to avoid log(0)
        
        # Calculate weighted energy
        total_energy = np.sum(positive_magnitude ** 2)
        weighted_energy = np.sum((positive_magnitude ** 2) * weights)
        
        sharpness = weighted_energy / total_energy if total_energy > 0 else 0
        
        return float(sharpness)

class AutotuneQualityAnalyzer:
    """Specialized autotune quality analyzer."""
    
    def __init__(self):
        self.pitch_analyzer = PitchAnalyzer()
        self.spectral_analyzer = SpectralAnalyzer()
        self.artifact_detector = ArtifactDetector()
        self.perceptual_analyzer = PerceptualAnalyzer()
    
    def analyze_autotune_quality(self, original: np.ndarray, processed: np.ndarray, 
                                sr: int, target_pitch: Optional[float] = None) -> Dict[str, Union[float, Dict]]:
        """
        Full autotune quality analysis.
        
        Args:
            original: Original audio
            processed: Processed audio
            sr: Sampling rate
            target_pitch: Target pitch (optional)
            
        Returns:
            Dictionary with all quality metrics
        """
        results = {}
        
        # 1. Pitch analysis
        original_pitches = self.pitch_analyzer.detect_pitch_yin(original, sr)
        processed_pitches = self.pitch_analyzer.detect_pitch_yin(processed, sr)
        
        results['pitch_analysis'] = {
            'original_stability': self.pitch_analyzer.calculate_pitch_stability(original_pitches),
            'processed_stability': self.pitch_analyzer.calculate_pitch_stability(processed_pitches)
        }
        
        if target_pitch:
            results['pitch_analysis']['accuracy'] = self.pitch_analyzer.calculate_pitch_accuracy(
                target_pitch, processed_pitches
            )
        
        # 2. Spectral analysis
        results['spectral_analysis'] = {
            'original_features': self.spectral_analyzer.calculate_spectral_features(original, sr),
            'processed_features': self.spectral_analyzer.calculate_spectral_features(processed, sr)
        }
        
        # Calculate spectral preservation
        orig_centroid = results['spectral_analysis']['original_features']['spectral_centroid']
        proc_centroid = results['spectral_analysis']['processed_features']['spectral_centroid']
        
        results['spectral_analysis']['centroid_preservation'] = 1.0 - abs(proc_centroid - orig_centroid) / orig_centroid if orig_centroid > 0 else 0
        
        # 3. Harmonic analysis
        if target_pitch:
            results['harmonic_analysis'] = {
                'original': self.spectral_analyzer.calculate_harmonic_content(original, sr, target_pitch),
                'processed': self.spectral_analyzer.calculate_harmonic_content(processed, sr, target_pitch)
            }
        
        # 4. Artifact detection
        results['artifact_detection'] = {
            'aliasing': self.artifact_detector.detect_aliasing(processed, sr),
            'clicks_pops': self.artifact_detector.detect_clicks_pops(processed, sr),
            'modulation_artifacts': self.artifact_detector.detect_modulation_artifacts(processed, sr)
        }
        
        # 5. Perceptual analysis
        results['perceptual_analysis'] = {
            'original_roughness': self.perceptual_analyzer.calculate_roughness(original, sr),
            'processed_roughness': self.perceptual_analyzer.calculate_roughness(processed, sr),
            'original_sharpness': self.perceptual_analyzer.calculate_sharpness(original, sr),
            'processed_sharpness': self.perceptual_analyzer.calculate_sharpness(processed, sr)
        }
        
        # 6. Global metrics
        results['global_metrics'] = self._calculate_global_metrics(original, processed, sr)
        
        # 7. Overall quality score
        results['quality_score'] = self._calculate_quality_score(results)
        
        return results
    
    def _calculate_global_metrics(self, original: np.ndarray, processed: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculates global quality metrics."""
        # SNR
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((processed - original) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Correlation
        correlation = np.corrcoef(original, processed)[0, 1] if len(original) == len(processed) else 0
        
        # Energy preservation
        original_energy = np.sum(original ** 2)
        processed_energy = np.sum(processed ** 2)
        energy_preservation = min(original_energy, processed_energy) / max(original_energy, processed_energy) if max(original_energy, processed_energy) > 0 else 0
        
        # Dynamic range
        original_dr = 20 * np.log10(np.max(np.abs(original)) / (np.mean(np.abs(original)) + 1e-10))
        processed_dr = 20 * np.log10(np.max(np.abs(processed)) / (np.mean(np.abs(processed)) + 1e-10))
        dr_preservation = 1.0 - abs(processed_dr - original_dr) / original_dr if original_dr > 0 else 0
        
        return {
            'snr': float(snr),
            'correlation': float(correlation),
            'energy_preservation': float(energy_preservation),
            'dynamic_range_preservation': float(dr_preservation)
        }
    
    def _calculate_quality_score(self, analysis_results: Dict) -> Dict[str, float]:
        """Calculates overall quality score based on all metrics."""
        scores = {}
        
        # Pitch score (0-100)
        if 'accuracy' in analysis_results.get('pitch_analysis', {}):
            pitch_accuracy = analysis_results['pitch_analysis']['accuracy']['accuracy_percentage']
            scores['pitch_score'] = pitch_accuracy
        else:
            # Use stability as a proxy
            stability = analysis_results.get('pitch_analysis', {}).get('processed_stability', {}).get('stability_ratio', 0)
            scores['pitch_score'] = stability * 100
        
        # Spectral score (0-100)
        centroid_preservation = analysis_results.get('spectral_analysis', {}).get('centroid_preservation', 0)
        scores['spectral_score'] = centroid_preservation * 100
        
        # Artifact score (0-100, where 100 = no artifacts)
        aliasing_ratio = analysis_results.get('artifact_detection', {}).get('aliasing', {}).get('aliasing_ratio', 0)
        click_rate = analysis_results.get('artifact_detection', {}).get('clicks_pops', {}).get('click_rate', 0)
        modulation_strength = analysis_results.get('artifact_detection', {}).get('modulation_artifacts', {}).get('modulation_strength', 0)
        
        artifact_score = 100 * (1 - min(1, aliasing_ratio + click_rate/10 + modulation_strength))
        scores['artifact_score'] = artifact_score
        
        # Global score (0-100)
        snr = analysis_results.get('global_metrics', {}).get('snr', 0)
        correlation = analysis_results.get('global_metrics', {}).get('correlation', 0)
        energy_preservation = analysis_results.get('global_metrics', {}).get('energy_preservation', 0)
        
        # Normalize SNR to 0-100 (assume max useful SNR of 60dB)
        snr_normalized = min(100, max(0, snr * 100 / 60)) if snr != float('inf') else 100
        
        global_score = (snr_normalized + correlation * 100 + energy_preservation * 100) / 3
        scores['global_score'] = global_score
        
        # Final score (weighted average)
        weights = {'pitch': 0.3, 'spectral': 0.2, 'artifact': 0.3, 'global': 0.2}
        final_score = (
            scores['pitch_score'] * weights['pitch'] +
            scores['spectral_score'] * weights['spectral'] +
            scores['artifact_score'] * weights['artifact'] +
            scores['global_score'] * weights['global']
        )
        scores['final_score'] = final_score
        
        return scores

# Convenience function for quick analysis
def quick_autotune_analysis(original: np.ndarray, processed: np.ndarray, 
                           sr: int, target_pitch: Optional[float] = None) -> Dict:
    """
    Quick autotune quality analysis.
    
    Args:
        original: Original audio
        processed: Processed audio
        sr: Sampling rate
        target_pitch: Target pitch (optional)
        
    Returns:
        Dictionary with main metrics
    """
    analyzer = AutotuneQualityAnalyzer()
    full_analysis = analyzer.analyze_autotune_quality(original, processed, sr, target_pitch)
    
    # Extract main metrics
    quick_results = {
        'quality_score': full_analysis.get('quality_score', {}),
        'snr': full_analysis.get('global_metrics', {}).get('snr', 0),
        'pitch_accuracy': full_analysis.get('pitch_analysis', {}).get('accuracy', {}).get('accuracy_percentage', 0),
        'artifact_level': 100 - full_analysis.get('quality_score', {}).get('artifact_score', 100),
        'spectral_preservation': full_analysis.get('spectral_analysis', {}).get('centroid_preservation', 0) * 100
    }
    
    return quick_results

if __name__ == "__main__":
    # Example usage
    print("Advanced Audio Quality Metrics Module for Audio Analysis")
    print("=" * 60)
    print("This module provides detailed quality analysis for pitch shifting and autotune algorithms.")
    print("\nMain functions:")
    print("- PitchAnalyzer: Pitch and tuning analysis")
    print("- SpectralAnalyzer: Advanced spectral analysis")
    print("- ArtifactDetector: Artifact detection")
    print("- PerceptualAnalyzer: Perceptual analysis")
    print("- AutotuneQualityAnalyzer: Full autotune analysis")
    print("- quick_autotune_analysis(): Quick analysis")

