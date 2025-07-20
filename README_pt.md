# Comparação Aprimorada de Bibliotecas de Pitch Shifting e Autotune 🎵

Uma ferramenta abrangente para testar, comparar e avaliar diferentes bibliotecas Python de pitch shifting e autotune, com métricas avançadas de qualidade, análise de performance e visualizações detalhadas.

## 🎯 Características Principais

- **Suporte a múltiplas bibliotecas**: LibROSA, PyDub, Parselmouth, pedalboard, pyrubberband, SciPy
- **Métricas avançadas de qualidade**: SNR, THD, coerência de fase, estabilidade espectral
- **Análise de performance**: Tempo de processamento, uso de CPU e memória
- **Casos de teste diversificados**: Tom puro, sinais harmônicos, acordes, sinais com ruído
- **Visualizações interativas**: Gráficos comparativos, heatmaps, análise espectral
- **Detecção de artefatos**: Aliasing, clicks, pops, artefatos de modulação
- **Relatórios detalhados**: Rankings, recomendações, análise de robustez

## 📊 Bibliotecas Suportadas

| Biblioteca | Algoritmos | Status | Instalação |
|------------|------------|--------|------------|
| **LibROSA** | pitch_shift, phase_vocoder | ✅ Estável | `pip install librosa` |
| **PyDub** | speed_change | ✅ Estável | `pip install pydub` |
| **Parselmouth** | PSOLA, change_gender | ✅ Estável | `pip install praat-parselmouth` |
| **pedalboard** | pitch_shift | ✅ Estável | `pip install pedalboard` |
| **pyrubberband** | pitch_shift, time_stretch | ⚠️ Requer externa | `pip install pyrubberband` + rubberband-cli |
| **SciPy** | manual_implementation | ✅ Estável | `pip install scipy` |

## 🚀 Instalação Rápida

### Pré-requisitos
- Python 3.8+
- pip

### Instalação Básica
```bash
# Clone o repositório
git clone https://github.com/dobidu/pitch-shift-comparison.git
cd pitch-shift-comparison

# Instale dependências básicas
pip install numpy scipy matplotlib seaborn pandas psutil soundfile

# Execute o script com instalação automática
python pitch_shift_comparison_enhanced.py
```

### Instalação Completa
```bash
# Instala todas as bibliotecas suportadas
pip install librosa pydub praat-parselmouth pedalboard pyrubberband
pip install matplotlib seaborn pandas psutil soundfile tqdm

# Para pyrubberband (requer software externo):
# Ubuntu/Debian:
sudo apt-get install rubberband-cli

# macOS:
brew install rubberband

# Windows:
# Baixe de https://breakfastquay.com/rubberband/
```

## 📋 Dependências Detalhadas

### Obrigatórias
```
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
soundfile>=0.12.1
psutil>=5.9.0
```

### Opcionais (bibliotecas de pitch shifting)
```
librosa>=0.10.0          # Análise e processamento de áudio
pydub>=0.25.0            # Manipulação básica de áudio
praat-parselmouth>=0.4.0 # Algoritmos de análise de fala
pedalboard>=0.7.0        # Efeitos de áudio do Spotify
pyrubberband>=0.3.0      # Wrapper para Rubberband
```

### Visualização e análise
```
seaborn>=0.11.0          # Visualizações estatísticas
pandas>=1.5.0            # Análise de dados
tqdm>=4.64.0             # Barras de progresso
```

## 🎛️ Uso

### Execução Interativa
```bash
python pitch_shift_comparison_enhanced.py
```

O script oferece um menu interativo com as seguintes opções:

1. **🧪 Teste rápido**: Testa ±1, ±3, ±6 semitons com casos básicos
2. **🔬 Teste completo**: Testa ±12 semitons com todos os casos
3. **🎯 Teste personalizado**: Permite escolher semitons e casos específicos
4. **🎼 Teste de robustez**: Testa múltiplos casos de áudio
5. **📊 Ver resultados**: Exibe relatórios de testes anteriores
6. **💾 Salvar resultados**: Exporta dados em JSON
7. **📈 Criar gráficos**: Gera visualizações avançadas
8. **🔍 Análise comparativa**: Relatório detalhado de performance

### Uso Programático
```python
from pitch_shift_comparison_enhanced import EnhancedPitchShiftingTester

# Inicializa o testador
tester = EnhancedPitchShiftingTester()

# Executa teste completo
tester.run_comprehensive_test(
    test_semitones=[-12, -6, -3, 0, 3, 6, 12],
    test_case_names=["pure_tone", "harmonic_tone", "chord"]
)

# Gera relatório
tester.generate_advanced_report()

# Cria gráficos
tester.create_advanced_plots()

# Salva resultados
tester.save_detailed_results("my_results.json")
```

## 📊 Métricas de Análise

### Qualidade de Áudio
- **SNR (Signal-to-Noise Ratio)**: Relação sinal-ruído
- **THD (Total Harmonic Distortion)**: Distorção harmônica total
- **Coerência de Fase**: Preservação da fase entre sinais
- **Estabilidade Espectral**: Variação do centroide espectral
- **Precisão de Pitch**: Erro em cents comparado ao alvo

### Performance
- **Tempo de Processamento**: Latência de execução
- **Uso de CPU**: Percentual médio e pico
- **Uso de Memória**: Delta de memória durante processamento
- **Taxa de Sucesso**: Percentual de testes bem-sucedidos

### Detecção de Artefatos
- **Aliasing**: Energia em altas frequências espúrias
- **Clicks/Pops**: Descontinuidades no sinal
- **Artefatos de Modulação**: Tremolo/vibrato indesejado

## 🎼 Casos de Teste

| Caso | Descrição | Características |
|------|-----------|----------------|
| **pure_tone** | Tom puro 440Hz | Sinal simples, ideal para testes básicos |
| **harmonic_tone** | Tom com harmônicos + envelope | Simula instrumentos musicais |
| **chord** | Acorde C maior | Múltiplas frequências simultâneas |
| **noisy_tone** | Tom puro com ruído | Testa robustez contra ruído |
| **freq_sweep** | Sweep 440-640Hz | Testa resposta a mudanças de frequência |

## 📈 Visualizações

O script gera gráficos abrangentes:

### Gráficos de Performance
- **Tempo vs SNR**: Relação qualidade-velocidade
- **Distribuição de SNR**: Boxplots por método
- **Ranking de Velocidade**: Barras de tempo médio

### Análise Espectral
- **Heatmap de Qualidade**: SNR por método e semitom
- **Linha de Qualidade**: SNR vs pitch shift
- **Preservação Espectral**: Mudanças no centroide

### Análise de Recursos
- **CPU vs Memória**: Scatter plot de eficiência
- **Score de Eficiência**: Métrica composta
- **Gráfico Radar**: Perfil multi-dimensional

## 🏆 Interpretação de Resultados

### Rankings Disponíveis

1. **Performance**: Ordena por velocidade de processamento
2. **Qualidade**: Ordena por SNR médio
3. **Robustez**: Taxa de sucesso em diferentes cenários
4. **Eficiência**: Balanceamento de qualidade vs recursos

### Recomendações Automáticas

O sistema fornece recomendações específicas:

- **⚡ Tempo Real**: Biblioteca mais rápida
- **🎵 Máxima Qualidade**: Melhor SNR médio
- **🛡️ Maior Robustez**: Mais estável em diversos cenários
- **💾 Maior Eficiência**: Melhor uso de recursos
- **⭐ Melhor Geral**: Score combinado ponderado

### Interpretação de Scores

| Score | Interpretação |
|-------|---------------|
| 90-100 | Excelente |
| 80-89 | Muito Bom |
| 70-79 | Bom |
| 60-69 | Aceitável |
| <60 | Precisa melhorias |

## 🔧 Configuração Avançada

### Personalizando Testes
```python
# Casos de teste personalizados
custom_semitones = [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
custom_cases = ["pure_tone", "chord", "noisy_tone"]

tester.run_comprehensive_test(
    test_semitones=custom_semitones,
    test_case_names=custom_cases
)
```

### Criando Casos de Teste Próprios
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

# Cria caso personalizado
sr = 44100
duration = 2.0
t = np.linspace(0, duration, int(sr * duration))
custom_audio = 0.5 * np.sin(2 * np.pi * 330 * t)  # E4

custom_case = AudioTestCase(
    name="custom_e4",
    audio=custom_audio.astype(np.float32),
    sr=sr,
    description="Tom puro E4 (330Hz)",
    fundamental_freq=330.0
)

tester.test_cases.append(custom_case)
```

## 📁 Estrutura de Arquivos de Saída

```
output_pure_tone_librosa_standard_+3.0st.wav    # Áudios processados
output_harmonic_tone_pyrubberband_shift_-6.0st.wav
pitch_shifting_analysis_20241220_143022.png     # Gráficos
pitch_shift_results_enhanced_20241220_143022.json  # Dados detalhados
```

### Formato do JSON de Resultados
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

## 🐛 Solução de Problemas

### Problemas Comuns

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

# Verificar instalação
rubberband --help
```

#### Erro de memória em arquivos grandes
- Reduza a duração dos casos de teste
- Use menos semitons por teste
- Execute testes menores separadamente

#### Gráficos não aparecem
```bash
pip install matplotlib seaborn
# Para sistemas sem display:
export MPLBACKEND=Agg
```

### Logs de Debug

Para debug detalhado, modifique o nível de logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contribuindo

### Como Contribuir

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-biblioteca`)
3. Commit suas mudanças (`git commit -am 'Adiciona suporte para nova biblioteca'`)
4. Push para a branch (`git push origin feature/nova-biblioteca`)
5. Abra um Pull Request

### Adicionando Novas Bibliotecas

Para adicionar suporte a uma nova biblioteca:

1. **Detectar a biblioteca** em `_detect_libraries()`:
```python
try:
    import nova_biblioteca
    self.libraries['nova_lib'] = LibraryInfo(
        name='Nova Biblioteca',
        version=nova_biblioteca.__version__,
        available=True,
        import_error=None,
        algorithms=['pitch_shift_novo'],
        installation_notes="pip install nova_biblioteca"
    )
except ImportError as e:
    # Handle error...
```

2. **Implementar método de pitch shifting**:
```python
def pitch_shift_nova_lib(self, audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch shifting usando Nova Biblioteca."""
    import nova_biblioteca
    return nova_biblioteca.pitch_shift(audio, sr, semitones)
```

3. **Adicionar ao teste** em `run_comprehensive_test()`:
```python
if self.libraries['nova_lib'].available:
    test_methods.append((self.pitch_shift_nova_lib, "nova_lib_shift"))
```

### Guidelines de Código

- Use type hints sempre que possível
- Documente funções com docstrings
- Mantenha compatibilidade com Python 3.8+
- Adicione testes para novas funcionalidades
- Siga PEP 8 para estilo de código

## 📝 Changelog

### v2.0.0
- ✨ Adicionado suporte para pyrubberband
- ✨ Métricas avançadas de qualidade de áudio
- ✨ Análise de robustez com múltiplos casos de teste
- ✨ Monitoramento de recursos (CPU, memória)
- ✨ Gráficos avançados com múltiplas visualizações
- ✨ Detecção automática de artefatos
- ✨ Sistema de recomendações inteligente
- 🐛 Correção de problemas de compatibilidade
- 📚 Documentação completa

### v1.0.0
- 🎉 Versão inicial
- ✨ Suporte básico para LibROSA, PyDub, Parselmouth
- ✨ Métricas básicas de qualidade
- ✨ Relatórios simples

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **LibROSA**: Pela excelente biblioteca de análise de áudio
- **Parselmouth**: Por tornar o Praat acessível via Python
- **Spotify pedalboard**: Pelos efeitos de áudio de alta qualidade
- **Rubber Band**: Pelo algoritmo robusto de time-stretching
- **Comunidade Python**: Por todas as bibliotecas que tornam este projeto possível

## 📞 Suporte

- 🐛 **Issues**: [GitHub Issues](https://github.com/dobidu/pitch-shift-comparison/issues)
- 💬 **Discussões**: [GitHub Discussions](https://github.com/dobidu/pitch-shift-comparison/discussions)

## 🔗 Links Úteis

- [LibROSA Documentation](https://librosa.org/doc/latest/)
- [Parselmouth Documentation](https://parselmouth.readthedocs.io/)
- [pedalboard Documentation](https://spotify.github.io/pedalboard/)
- [Rubber Band Audio Processor](https://breakfastquay.com/rubberband/)
- [PyDub Documentation](https://pydub.com/)

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no GitHub!**
