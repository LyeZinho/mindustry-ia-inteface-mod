# Guia de Treinamento — Agente RL Mindustry

Este guia explica como configurar o ambiente, iniciar o treinamento do agente MaskablePPO e monitorar o progresso.

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

# Ativar (Linux / macOS — bash/zsh)
source rl/venv/bin/activate

# Ativar (fish shell)
source rl/venv/bin/activate.fish

# Ativar (Windows)
.\rl\venv\Scripts\activate

# Instalar dependências
pip install -r rl/requirements.txt
```

> **Arch Linux / externally-managed-environment:** o `pip install` direto falha. Use sempre o venv acima.

Dependências instaladas:

| Pacote | Função |
|--------|--------|
| `stable-baselines3[extra]` | Utilitários de RL (VecEnv, Monitor, callbacks) |
| `sb3-contrib` | Algoritmo MaskablePPO com mascaramento de ações |
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

> O mod envia estado a cada **10 ticks** (~6 Hz com FPS padrão). O loop de comunicação roda a **~20 Hz** (sleep 50 ms). Aumentar o FPS do servidor acelera proporcionalmente a coleta de dados.

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
  --lr 3e-4 \
  --n-steps 256 \
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
| `--lr` | `3e-4` | Taxa de aprendizado do MaskablePPO |
| `--n-steps` | `256` | Passos por rollout antes de atualizar a política |
| `--max-steps` | `5000` | Passos máximos por episódio (truncamento) |
| `--models-dir` | `rl/models` | Diretório para salvar checkpoints e modelo final |
| `--logs-dir` | `rl/logs` | Diretório para logs do TensorBoard e Monitor |

### O que acontece durante o treino

1. O script conecta ao Mindustry via TCP e envia `RESET;<mapa>` para iniciar um episódio.
2. A cada passo, o agente recebe o estado do jogo (grid 31×31 + 47 features globais), calcula quais ações são válidas (mascaramento), escolhe uma ação e recebe a recompensa calculada pela função multi-objetivo.
3. O mascaramento de ações impede que o agente tente construir sem recursos ou reparar sem edifícios — evitando gradiente desperdiçado em ações impossíveis.
4. A cada **10 000 passos**, um checkpoint é salvo automaticamente em `rl/models/mindustry_ppo_<N>_steps.zip`.
5. Ao final, o modelo completo é salvo em `rl/models/final_model.zip`.

### Saída esperada no terminal

```
Starting MaskablePPO training for 1,000,000 timesteps (4 envs)...
Using cpu device
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
| KL aproximado | `train/approx_kl` | Divergência da política (PPO) |
| Fração de clip | `train/clip_fraction` | % de passos que ativaram o clipping do PPO |

> **Sinal de progresso saudável:** `rollout/mean_episode_reward` subindo de valores negativos (−0.1 a −0.3 nas primeiras iterações) para próximos de zero ou positivos. Episódios que terminam por truncamento (max_steps atingido) sem o core sendo destruído indicam que o agente aprendeu a sobreviver.

---

## 5. Checkpoints e Retomada

Checkpoints são salvos automaticamente a cada 10 000 passos:

```
rl/models/
├── mindustry_ppo_10000_steps.zip
├── mindustry_ppo_20000_steps.zip
├── ...
└── final_model.zip
```

Para retomar treino a partir de um checkpoint (**não implementado no `train.py` atual** — requer adaptação manual):

```python
from sb3_contrib import MaskablePPO
from rl.env.mindustry_env import MindustryEnv

env = MindustryEnv()
model = MaskablePPO.load("rl/models/mindustry_ppo_50000_steps", env=env)
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

## 7. Espaço de Ações

O agente usa um espaço `MultiDiscrete([7, 9])` — duas dimensões por passo:

- **action_type** (0–6): qual ação executar
- **arg** (0–8): direção (para MOVE) ou slot 3×3 relativo ao jogador (para BUILD/REPAIR)

| `action_type` | Ação | Comando enviado |
|--------------|------|-----------------|
| `0` | Aguardar | *(nenhum)* |
| `1` | Mover jogador | `PLAYER_MOVE;<dir>` |
| `2` | Construir torrente (`duo`) | `PLAYER_BUILD;duo;<slot>` |
| `3` | Construir muro (`copper-wall`) | `PLAYER_BUILD;copper-wall;<slot>` |
| `4` | Construir painel solar (`solar-panel`) | `PLAYER_BUILD;solar-panel;<slot>` |
| `5` | Construir drill (`mechanical-drill`) | `PLAYER_BUILD;mechanical-drill;<slot>` |
| `6` | Reparar edifício em slot | `REPAIR_SLOT;<slot>` |

### Mascaramento de ações (MaskablePPO)

O agente não pode escolher ações impossíveis. A máscara (shape `(16,)`) é calculada a cada passo:

| Ação mascarada | Condição |
|----------------|----------|
| MOVE, BUILD_*, REPAIR | Jogador morto |
| BUILD_TURRET | Cobre < 6 |
| BUILD_WALL | Cobre < 6 |
| BUILD_POWER | Chumbo < 14 ou cobre < 10 |
| BUILD_DRILL | Cobre < 12 |
| REPAIR | Nenhum edifício existente |

---

## 8. Função de Recompensa

```
reward = 0.30 × Δcore_hp
       + 0.20 × wave_survived_bonus   (1.0 se sobreviveu a uma wave)
       + 0.20 × player_alive_bonus    (1.0 se jogador vivo)
       + 0.10 × (Δresources / 500)    (500 ≈ capacidade do core)
       + 0.08 × drill_bonus           (contínuo: Δcopper/10, clamped [0,1])
       + 0.07 × power_balance_bonus   (produção ≥ consumo)
       + 0.05 × build_efficiency_bonus
       − 0.002                        (penalidade de tempo)

# Se done=True e core_hp ≤ 0:
reward −= 0.4                         (penalidade terminal — core destruído)

# Se done=True e player morto (core vivo):
reward −= 0.5                         (penalidade terminal — jogador morto)
```

> O sinal de sobrevivência (`player_alive_bonus = 0.20`) foi aumentado para que o agente aprenda a não morrer antes de aprender a construir defesas.

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
A conexão foi interrompida durante o treino. O ambiente reinicia automaticamente com backoff exponencial (até 5 tentativas). Se persistir, verifique o servidor Mindustry.

### Treino muito lento (< 10 FPS)
- Use servidor headless em vez do cliente desktop.
- Aumente o FPS do servidor Mindustry.
- Se disponível, PyTorch com CUDA acelera significativamente: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Logs de JSON malformado no terminal
O mod enviou uma linha inválida (comum durante carregamento de mapa). O cliente descarta silenciosamente e o `WARNING` aparece no log. Não interrompe o treino.

---

## 10. Referência Rápida

```bash
# 1. Ativar ambiente (bash/zsh)
source rl/venv/bin/activate
# ou, se usar fish shell:
source rl/venv/bin/activate.fish

# 2. Verificar testes
python -m pytest rl/tests/ -v

# 3. Iniciar treino (Mindustry já aberto)
python -m rl.train --timesteps 1000000

# 4. Monitorar (terminal separado)
tensorboard --logdir rl/logs/

# 5. Avaliar modelo final
python -m rl.evaluate --model rl/models/final_model --episodes 5
```
