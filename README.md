# Enhanced Pitch Shifting and Autotune Library Comparison 🎵

A comprehensive tool for testing, comparing, and evaluating different Python pitch shifting and autotune libraries, with advanced quality metrics, performance analysis, and detailed visualizations.

## 🎯 Key Features

- **Support for multiple libraries**: LibROSA, PyDub, Parselmouth, pedalboard, pyrubberband, SciPy
- **Advanced quality metrics**: SNR, THD, phase coherence, spectral stability
- **Performance analysis**: Processing time, CPU and memory usage
- **Diverse test cases**: Pure tone, harmonic signals, chords, noisy signals
- **Interactive visualizations**: Comparative charts, heatmaps, spectral analysis
- **Artifact detection**: Aliasing, clicks, pops, modulation artifacts
- **Detailed reports**: Rankings, recommendations, robustness analysis

## 📊 Supported Libraries

| Library | Algorithms | Status | Installation |
|---------|------------|--------|--------------|
| **LibROSA** | pitch_shift, phase_vocoder | ✅ Stable | `pip install librosa` |
| **PyDub** | speed_change | ✅ Stable | `pip install pydub` |
| **Parselmouth** | PSOLA, change_gender | ✅ Stable | `pip install praat-parselmouth` |
| **pedalboard** | pitch_shift | ✅ Stable | `pip install pedalboard` |
| **pyrubberband** | pitch_shift, time_stretch | ⚠️ Requires external | `pip install pyrubberband` + rubberband-cli |
| **SciPy** | manual_implementation | ✅ Stable | `pip install scipy` |

## 🚀 Quick Installation

### Prerequisites
- Python 3.8+
- pip

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/dobidu/pitch-shift-comparison.git
cd pitch-shift-comparison

# Install basic dependencies
pip install numpy scipy matplotlib seaborn pandas psutil soundfile

# Run the script with automatic installation
python pitch_shift_comparison_enhanced.py
```

### Complete Installation
```bash
# Install all supported libraries
pip install librosa pydub praat-parselmouth pedalboard pyrubberband
pip install matplotlib seaborn pandas psutil soundfile tqdm

# For pyrubberband (requires external software):
# Ubuntu/Debian:
sudo apt-get install rubberband-cli

# macOS:
brew install rubberband

# Windows:
# Download from https://breakfastquay.com/rubberband/
```

## 📋 Detailed Dependencies

### Required
```
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
soundfile>=0.12.1
psutil>=5.9.0
```

### Optional (pitch shifting libraries)
```
librosa>=0.10.0          # Audio analysis and processing
pydub>=0.25.0            # Basic audio manipulation
praat-parselmouth>=0.4.0 # Speech analysis algorithms
pedalboard>=0.7.0        # Spotify audio effects
pyrubberband>=0.3.0      # Rubberband wrapper
```

### Visualization and analysis
```
seaborn>=0.11.0          # Statistical visualizations
pandas>=1.5.0            # Data analysis
tqdm>=4.64.0             # Progress bars
```

## 🎛️ Usage

### Interactive Execution
```bash
python pitch_shift_comparison_enhanced.py
```

The script offers an interactive menu with the following options:

1. **🧪 Quick test**: Tests ±1, ±3, ±6 semitones with basic cases
2. **🔬 Complete test**: Tests ±12 semitones with all cases
3. **🎯 Custom test**: Allows choosing specific semitones and cases
4. **🎼 Robustness test**: Tests multiple audio cases
5. **📊 View results**: Displays reports from previous tests
6. **💾 Save results**: Exports data to JSON
7. **📈 Create charts**: Generates advanced visualizations
8. **🔍 Comparative analysis**: Detailed performance report

### Programmatic Usage
```python
from pitch_shift_comparison_enhanced import EnhancedPitchShiftingTester

# Initialize tester
tester = EnhancedPitchShiftingTester()

# Run complete test
tester.run_comprehensive_test(
    test_semitones=[-12, -6, -3, 0, 3, 6, 12],
    test_case_names=["pure_tone", "harmonic_tone", "chord"]
)

# Generate report
tester.generate_advanced_report()

# Create charts
tester.create_advanced_plots()

# Save results
tester.save_detailed_results("my_results.json")
```

## 📊 Analysis Metrics

### Audio Quality
- **SNR (Signal-to-Noise Ratio)**: Signal-to-noise ratio
- **THD (Total Harmonic Distortion)**: Total harmonic distortion
- **Phase Coherence**: Phase preservation between signals
- **Spectral Stability**: Spectral centroid variation
- **Pitch Accuracy**: Error in cents compared to target

### Performance
- **Processing Time**: Execution latency
- **CPU Usage**: Average and peak percentage
- **Memory Usage**: Memory delta during processing
- **Success Rate**: Percentage of successful tests

### Artifact Detection
- **Aliasing**: Energy in spurious high frequencies
- **Clicks/Pops**: Signal discontinuities
- **Modulation Artifacts**: Unwanted tremolo/vibrato

## 🎼 Test Cases

| Case | Description | Characteristics |
|------|-------------|----------------|
| **pure_tone** | Pure 440Hz tone | Simple signal, ideal for basic tests |
| **harmonic_tone** | Tone with harmonics + envelope | Simulates musical instruments |
| **chord** | C major chord | Multiple simultaneous frequencies |
| **noisy_tone** | Pure tone with noise | Tests robustness against noise |
| **freq_sweep** | 440-640Hz sweep | Tests response to frequency changes |

## 📈 Visualizations

The script generates comprehensive charts:

### Performance Charts
- **Time vs SNR**: Quality-speed relationship
- **SNR Distribution**: Boxplots by method
- **Speed Ranking**: Average time bars

### Spectral Analysis
- **Quality Heatmap**: SNR by method and semitone
- **Quality Line**: SNR vs pitch shift
- **Spectral Preservation**: Centroid changes

### Resource Analysis
- **CPU vs Memory**: Efficiency scatter plot
- **Efficiency Score**: Composite metric
- **Radar Chart**: Multi-dimensional profile

## 🏆 Results Interpretation

### Available Rankings

1. **Performance**: Ordered by processing speed
2. **Quality**: Ordered by average SNR
3. **Robustness**: Success rate in different scenarios
4. **Efficiency**: Quality vs resources balance

### Automatic Recommendations

The system provides specific recommendations:

- **⚡ Real-time**: Fastest library
- **🎵 Maximum Quality**: Best average SNR
- **🛡️ Greatest Robustness**: Most stable across scenarios
- **💾 Greatest Efficiency**: Best resource usage
- **⭐ Best Overall**: Weighted combined score

### Score Interpretation

| Score | Interpretation |
|-------|---------------|
| 90-100 | Excellent |
| 80-89 | Very Good |
| 70-79 | Good |
| 60-69 | Acceptable |
| <60 | Needs improvement |

## 🔧 Advanced Configuration

### Customizing Tests
```python
# Custom test cases
custom_semitones = [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
custom_cases = ["pure_tone", "chord", "noisy_tone"]

tester.run_comprehensive_test(
    test_semitones=custom_semitones,
    test_case_names=custom_cases
)
```

### Creating Custom Test Cases
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class AudioTestCase:
    name: str
    audio: np.ndarray
    sr: int
    description: str
    fundamental_freq: float = 440.0

# Create custom case
sr = 44100
duration = 2.0
t = np.linspace(0, duration, int(sr * duration))
custom_audio = 0.5 * np.sin(2 * np.pi * 330 * t)  # E4

custom_case = AudioTestCase(
    name="custom_e4",
    audio=custom_audio.astype(np.float32),
    sr=sr,
    description="Pure E4 tone (330Hz)",
    fundamental_freq=330.0
)

tester.test_cases.append(custom_case)
```

## 📁 Output File Structure

```
output_pure_tone_librosa_standard_+3.0st.wav    # Processed audio
output_harmonic_tone_pyrubberband_shift_-6.0st.wav
pitch_shifting_analysis_20241220_143022.png     # Charts
pitch_shift_results_enhanced_20241220_143022.json  # Detailed data
```

### JSON Results Format
```json
{
  "metadata": {
    "test_date": "2024-12-20 14:30:22",
    "total_tests": 315,
    "libraries_tested": 6,
    "version": "2.0.0"
  },
  "libraries": {
    "librosa": {
      "name": "LibROSA",
      "version": "0.10.1",
      "available": true,
      "algorithms": ["pitch_shift", "phase_vocoder"]
    }
  },
  "results": [
    {
      "library_name": "librosa",
      "algorithm_name": "librosa_standard",
      "semitones": 3.0,
      "processing_time": 0.245,
      "quality_metrics": {
        "snr": 18.5,
        "thd": -45.2,
        "spectral_stability": 0.15
      }
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

#### ImportError: No module named 'librosa'
```bash
pip install librosa soundfile
```

#### pyrubberband error: rubberband not found
```bash
# Ubuntu/Debian
sudo apt-get install rubberband-cli

# macOS
brew install rubberband

# Verify installation
rubberband --help
```

#### Memory error with large files
- Reduce test case duration
- Use fewer semitones per test
- Run smaller tests separately

#### Charts don't appear
```bash
pip install matplotlib seaborn
# For systems without display:
export MPLBACKEND=Agg
```

### Debug Logs

For detailed debugging, modify the logging level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### How to Contribute

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/new-library`)
3. Commit your changes (`git commit -am 'Add support for new library'`)
4. Push to the branch (`git push origin feature/new-library`)
5. Open a Pull Request

### Adding New Libraries

To add support for a new library:

1. **Detect the library** in `_detect_libraries()`:
```python
try:
    import new_library
    self.libraries['new_lib'] = LibraryInfo(
        name='New Library',
        version=new_library.__version__,
        available=True,
        import_error=None,
        algorithms=['pitch_shift_new'],
        installation_notes="pip install new_library"
    )
except ImportError as e:
    # Handle error...
```

2. **Implement pitch shifting method**:
```python
def pitch_shift_new_lib(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch shifting using New Library."""
    import new_library
    return new_library.pitch_shift(audio, sr, semitones)
```

3. **Add to test** in `run_comprehensive_test()`:
```python
if self.libraries['new_lib'].available:
    test_methods.append((self.pitch_shift_new_lib, "new_lib_shift"))
```

### Code Guidelines

- Use type hints whenever possible
- Document functions with docstrings
- Maintain compatibility with Python 3.8+
- Add tests for new functionalities
- Follow PEP 8 for code style

## 📝 Changelog

### v2.0.0
- ✨ Added pyrubberband support
- ✨ Advanced audio quality metrics
- ✨ Robustness analysis with multiple test cases
- ✨ Resource monitoring (CPU, memory)
- ✨ Advanced charts with multiple visualizations
- ✨ Automatic artifact detection
- ✨ Intelligent recommendation system
- 🐛 Fixed compatibility issues
- 📚 Complete documentation

### v1.0.0
- 🎉 Initial version
- ✨ Basic support for LibROSA, PyDub, Parselmouth
- ✨ Basic quality metrics
- ✨ Simple reports

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LibROSA**: For the excellent audio analysis library
- **Parselmouth**: For making Praat accessible via Python
- **Spotify pedalboard**: For high-quality audio effects
- **Rubber Band**: For the robust time-stretching algorithm
- **Python Community**: For all the libraries that make this project possible

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/pitch-shift-comparison/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/pitch-shift-comparison/discussions)
- 📧 **Email**: your-email@example.com

## 🔗 Useful Links

- [LibROSA Documentation](https://librosa.org/doc/latest/)
- [Parselmouth Documentation](https://parselmouth.readthedocs.io/)
- [pedalboard Documentation](https://spotify.github.io/pedalboard/)
- [Rubber Band Audio Processor](https://breakfastquay.com/rubberband/)
- [PyDub Documentation](https://pydub.com/)

---

**⭐ If this project was useful to you, consider giving it a star on GitHub!**