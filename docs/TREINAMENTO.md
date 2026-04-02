# Guia de Treinamento — Agente RL Mindustry

Este guia explica como configurar o ambiente, iniciar o treinamento do agente A2C e monitorar o progresso.

---

## Pré-requisitos

| Requisito | Versão mínima | Observação |
|-----------|--------------|------------|
| Python | 3.10+ | Testado com 3.14 |
| Mindustry | v7 (build 146+) | Desktop ou Server |
| Mod Mimi Gateway | v1.0.0 | Incluído neste repo |
| VRAM / RAM | 4 GB | Para treino com PyTorch CPU |

O mod Mimi Gateway **precisa estar instalado e ativo** no Mindustry antes de qualquer treino. Consulte o [GUIA_INSTALACAO_PT.md](../GUIA_INSTALACAO_PT.md) se ainda não fez isso.

---

## 1. Configurar o Ambiente Python

O projeto usa um virtualenv isolado em `rl/venv/`. Crie-o uma única vez:

```bash
# Na raiz do repositório
python -m venv rl/venv

# Ativar (Linux / macOS)
source rl/venv/bin/activate

# Ativar (Windows)
.\rl\venv\Scripts\activate

# Instalar dependências
pip install -r rl/requirements.txt
```

> **Arch Linux / externally-managed-environment:** o `pip install` direto falha. Use sempre o venv acima.

Dependências instaladas:

| Pacote | Função |
|--------|--------|
| `stable-baselines3[extra]` | Algoritmo A2C + utilitários |
| `gymnasium` | Interface padrão do ambiente |
| `torch` | Backend de rede neural |
| `tensorboard` | Visualização de métricas |
| `numpy` | Operações vetoriais |

---

## 2. Preparar o Mindustry

O agente se conecta ao jogo via TCP na porta **9000**. O jogo precisa estar rodando **antes** de iniciar o script de treino.

### Modo Desktop (recomendado para testes iniciais)

1. Abra o Mindustry normalmente.
2. Carregue um mapa — o mod só aceita conexões durante uma partida ativa.
3. Verifique no console F1 que aparece:
   ```
   [Mimi Gateway] Servidor iniciado na porta 9000
   ```

### Modo Servidor Headless (recomendado para treinos longos)

```bash
# Baixe o server.jar em https://github.com/Anuken/Mindustry/releases
java -jar server.jar
```

No prompt do servidor:
```
host          # inicia o servidor com um mapa padrão
config debug true  # opcional — exibe logs do mod
```

Para acelerar a simulação (mais passos por segundo = treino mais rápido):
```
fps 60        # limita a 60 FPS (padrão)
# Sem limite de FPS — use com cuidado
```

> O mod envia estado a cada **10 ticks** (~6 Hz com FPS padrão). Aumentar o FPS do servidor acelera proporcionalmente a coleta de dados.

---

## 3. Iniciar o Treinamento

Com o venv ativo e o Mindustry rodando:

```bash
# Treino padrão (1 000 000 timesteps)
python -m rl.train

# Especificando todos os parâmetros
python -m rl.train \
  --timesteps 2000000 \
  --host localhost \
  --port 9000 \
  --lr 7e-4 \
  --n-steps 128 \
  --max-steps 5000 \
  --models-dir rl/models \
  --logs-dir rl/logs
```

### Referência de argumentos

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--timesteps` | `1_000_000` | Total de passos de treinamento |
| `--host` | `localhost` | IP do servidor Mindustry |
| `--port` | `9000` | Porta TCP do mod |
| `--lr` | `7e-4` | Taxa de aprendizado do A2C |
| `--n-steps` | `128` | Passos por rollout antes de atualizar a política |
| `--max-steps` | `5000` | Passos máximos por episódio (truncamento) |
| `--models-dir` | `rl/models` | Diretório para salvar checkpoints e modelo final |
| `--logs-dir` | `rl/logs` | Diretório para logs do TensorBoard e Monitor |

### O que acontece durante o treino

1. O script conecta ao Mindustry via TCP e envia `MSG;RESET` para sinalizar início.
2. A cada passo, o agente recebe o estado do jogo (grid 31×31 + 43 features globais), escolhe uma ação e recebe a recompensa calculada pela função multi-objetivo.
3. A cada **10 000 passos**, um checkpoint é salvo automaticamente em `rl/models/mindustry_a2c_<N>_steps.zip`.
4. Ao final, o modelo completo é salvo em `rl/models/final_model.zip`.

### Saída esperada no terminal

```
Starting A2C training for 1,000,000 timesteps...
Using cuda device
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 312      |
|    ep_rew_mean     | -0.142   |
| time/              |          |
|    fps             | 47       |
|    iterations      | 1        |
|    time_elapsed    | 2        |
|    total_timesteps | 128      |
---------------------------------
...
Training complete. Model saved to rl/models/final_model.zip
```

---

## 4. Monitorar com TensorBoard

Em um terminal separado (com o venv ativo):

```bash
tensorboard --logdir rl/logs/
```

Acesse **http://localhost:6006** no navegador.

### Métricas disponíveis

| Métrica | Caminho no TensorBoard | Descrição |
|---------|----------------------|-----------|
| Recompensa média por episódio | `rollout/mean_episode_reward` | Calculada pelo `RewardLoggerCallback` |
| Tamanho médio do episódio | `rollout/ep_len_mean` | Passos até término ou truncamento |
| Loss do valor | `train/value_loss` | Erro do crítico |
| Loss da política | `train/policy_gradient_loss` | Gradiente do ator |
| Entropia | `train/entropy_loss` | Exploração — deve diminuir gradualmente |
| Taxa de aprendizado | `train/learning_rate` | Monitorar estabilidade |

> **Sinal de progresso saudável:** `rollout/mean_episode_reward` subindo de valores negativos (−0.1 a −0.3 nas primeiras iterações) para próximos de zero ou positivos. Episódios que terminam por truncamento (max_steps atingido) sem o core sendo destruído indicam que o agente aprendeu a sobreviver.

---

## 5. Checkpoints e Retomada

Checkpoints são salvos automaticamente a cada 10 000 passos:

```
rl/models/
├── mindustry_a2c_10000_steps.zip
├── mindustry_a2c_20000_steps.zip
├── ...
└── final_model.zip
```

Para retomar treino a partir de um checkpoint (**não implementado no `train.py` atual** — requer adaptação manual):

```python
from stable_baselines3 import A2C
from rl.env.mindustry_env import MindustryEnv

env = MindustryEnv()
model = A2C.load("rl/models/mindustry_a2c_50000_steps", env=env)
model.learn(total_timesteps=500_000)
model.save("rl/models/final_model")
```

---

## 6. Avaliar um Modelo Salvo

```bash
# Avalia o modelo final por 3 episódios
python -m rl.evaluate --model rl/models/final_model

# Mais episódios, servidor remoto
python -m rl.evaluate \
  --model rl/models/final_model \
  --episodes 10 \
  --host 192.168.1.100 \
  --port 9000
```

### Referência de argumentos

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--model` | (obrigatório) | Caminho para o `.zip` sem extensão |
| `--episodes` | `3` | Número de episódios de avaliação |
| `--host` | `localhost` | IP do servidor |
| `--port` | `9000` | Porta TCP |
| `--max-steps` | `5000` | Passos máximos por episódio |

### Saída esperada

```
Episode 1: reward=0.847
Episode 2: reward=1.203
Episode 3: reward=0.991

Mean reward over 3 episodes: 1.014
```

---

## 7. Estrutura do Espaço de Ações

O agente pode executar 8 tipos de ação em cada passo, com coordenadas `(x, y)` no grid 31×31:

| `action_type` | Ação | Comando enviado |
|--------------|------|-----------------|
| `0` | Aguardar | `MSG;WAIT` |
| `1` | Construir torrente (`duo`) | `BUILD;duo;x;y;0` |
| `2` | Construir muro (`wall`) | `BUILD;wall;x;y;0` |
| `3` | Construir painel solar (`solar-panel`) | `BUILD;solar-panel;x;y;0` |
| `4` | Reparar edifício em (x, y) | `REPAIR;x;y` |
| `5` | Mover unidade amiga para (x, y) | `UNIT_MOVE;<id>;x;y` |
| `6` | Atacar com unidade amiga em (x, y) | `ATTACK;<id>;x;y` |
| `7` | Criar unidade (factory em x, y) | `FACTORY;x;y;poly` |

---

## 8. Função de Recompensa

```
reward = 0.50 × Δcore_hp
       + 0.20 × wave_survived_bonus   (1.0 se sobreviveu a uma wave)
       + 0.15 × (Δresources / 500)    (500 ≈ capacidade do core)
       + 0.15 × friendly_ratio        (unidades amigas / total)
       − 0.001                        (penalidade de tempo)

# Se done=True e core_hp ≤ 0:
reward −= 1.0                         (penalidade terminal)
```

---

## 9. Resolução de Problemas

### `ConnectionRefusedError: [Errno 111] Connection refused`
O Mindustry não está rodando ou o mod não foi carregado. Verifique:
- O jogo está aberto e uma partida está ativa.
- O console F1 mostra `[Mimi Gateway] Servidor iniciado na porta 9000`.
- Nenhum firewall está bloqueando a porta 9000.

### `RuntimeError: Failed to receive initial state from Mindustry server`
O mod conectou mas não enviou estado. Possíveis causas:
- O mapa ainda está carregando — aguarde alguns segundos e tente novamente.
- O mod está em modo de erro — verifique o console F1 por exceções Java.

### `RuntimeError: Lost connection to Mindustry server during step`
A conexão foi interrompida durante o treino. O treino para imediatamente. Reinicie o Mindustry e retome pelo último checkpoint.

### Treino muito lento (< 10 FPS)
- Reduza `--n-steps` para `64` — menos dados por update, mais atualizações por segundo.
- Use servidor headless em vez do cliente desktop.
- Aumente o FPS do servidor Mindustry.
- Se disponível, PyTorch com CUDA acelera significativamente: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Logs de JSON malformado no terminal
O mod enviou uma linha inválida (comum durante carregamento de mapa). O cliente descarta silenciosamente e o `WARNING` aparece no log. Não interrompe o treino.

---

## 10. Referência Rápida

```bash
# 1. Ativar ambiente
source rl/venv/bin/activate

# 2. Verificar testes
python -m pytest rl/tests/ -v

# 3. Iniciar treino (Mindustry já aberto)
python -m rl.train --timesteps 1000000

# 4. Monitorar (terminal separado)
tensorboard --logdir rl/logs/

# 5. Avaliar modelo final
python -m rl.evaluate --model rl/models/final_model --episodes 5
```
