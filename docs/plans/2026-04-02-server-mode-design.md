# Design: Migração para Server Mode

**Data:** 2026-04-02  
**Status:** Aprovado

---

## Contexto

O pipeline de treino atual roda o Mindustry como cliente (com player local). Isso limita o controle do ambiente: resets são manuais, não há como observar o treinamento de fora, e o script de treino não gerencia o ciclo de vida do jogo.

A migração para **dedicated server** (`server-release.jar`) resolve isso:
- O `train.py` spawna e mata o server automaticamente
- O usuário pode conectar ao server para observar o treino em tempo real
- Resets de episódio são controlados programaticamente via protocolo TCP do mod
- Múltiplos mapas de treino rodam em sequência sem intervenção manual

---

## Decisões de Design

| Questão | Decisão |
|---|---|
| Reset de episódio | Comando `RESET;[mapname]` via TCP do mod |
| Spawn do server | `train.py` gerencia o processo automaticamente |
| Mapas de treino | Lista fixa de ~6 mapas vanilla cobrindo diferentes terrenos |
| Canal de comunicação | Único: TCP port 9000 (sem stdin/stdout do server) |

---

## Arquitetura

```
python -m rl.train --server-jar server-release.jar
  │
  ├── MindustryServer (rl/server/manager.py)  [NOVO]
  │     └── subprocess: java -jar server-release.jar
  │           └── stdout monitorado → detecta "Server loaded"
  │
  └── MindustryEnv (rl/env/mindustry_env.py)  [ATUALIZADO]
        └── MimiClient TCP :9000
              └── ←→ Mimi Gateway mod (dentro do server JVM)
                    ├── percepção: envia JSON state a cada N ticks
                    ├── ação: BUILD, UNIT_MOVE, ATTACK, REPAIR, ...
                    └── controle: RESET;[mapname] → carrega novo mapa
```

---

## Componentes

### 1. `rl/server/manager.py` — novo

```python
class MindustryServer:
    def __init__(self, jar_path: str, port: int = 9000, data_dir: str = "rl/server_data")
    def start(self) -> None          # spawna subprocess, aguarda "Server loaded"
    def stop(self) -> None           # mata o processo limpo
    def is_running(self) -> bool
```

- Spawna: `java -jar server-release.jar` com `cwd` em `data_dir/`
- Monitora stdout em thread daemon — detecta quando o server está pronto
- Timeout de 30s para startup
- `data_dir/` contém `config/` e `saves/` do Mindustry server (criado automaticamente)

### 2. `scripts/main.js` — novo comando RESET

Adiciona ao `validateCommand` / `processCommand` / handler:

```
RESET;[mapname]
```

Handler (`handleResetCommand`):
1. Busca o mapa por nome em `Maps.all()`
2. Se não encontrar, usa o primeiro disponível
3. Executa no game thread via `Core.app.post(...)`:
   ```javascript
   Vars.logic.reset();
   Vars.world.loadMap(map, map.applyRules(Gamemode.survival));
   Vars.state.set(State.playing);
   Vars.logic.play();
   ```
4. O mod retoma o loop normal — o próximo `sendGameState()` entrega o estado do mapa novo

### 3. `rl/env/mindustry_env.py` — atualizado

- Construtor aceita `maps: list[str]` (lista de nomes de mapa)
- `reset()`:
  1. Sorteia/cicla o próximo mapa da lista
  2. Envia `RESET;[mapname]` via `MimiClient`
  3. Aguarda estado válido (retry com timeout de 10s)
  4. Retorna observação inicial
- Remove o `self._client.message("RESET")` atual (era placeholder)

### 4. `rl/train.py` — atualizado

Novos argumentos:
```
--server-jar   path para server-release.jar (default: server-release.jar)
--maps         mapas separados por vírgula (default: lista vanilla hardcoded)
--no-server    não spawna o server (conecta a um server já rodando)
```

Fluxo:
```python
server = MindustryServer(args.server_jar)
server.start()
try:
    env = MindustryEnv(..., maps=args.maps)
    model.learn(...)
finally:
    server.stop()
```

### 5. Lista de mapas padrão

6 mapas vanilla do Mindustry cobrindo diferentes terrenos:
- `Ancient Caldera` — terreno aberto, planície
- `Windswept Islands` — separado por água, pontes
- `Tarpit Depths` — ácido, obstáculos
- `Craters` — terreno irregular
- `Fungal Pass` — corredor estreito
- `Nuclear Complex` — ambiente industrial denso

---

## Protocolo — extensão

```
Novo comando (Python → Mod):
  RESET;[mapname]\n          — recarrega mapa, inicia novo episódio

Sem mudança nos outros comandos.
Sem novo campo no JSON de estado.
```

---

## Fluxo de Episódio (server mode)

```
train.py                    MindustryServer          Mod (TCP)
   │                              │                      │
   ├─ server.start() ────────────►│                      │
   │                    [spawna jar, aguarda ready]       │
   │◄─ ready ────────────────────┤                      │
   │                              │                      │
   ├─ env.reset() ────────────────────────────────────►  │
   │              RESET;Ancient Caldera\n                 │
   │                              │          [carrega mapa]
   │◄──────────────────────────────────── JSON state ─── │
   │                              │                      │
   │  [loop de treino A2C]        │                      │
   ├─ env.step(action) ──────────────────────────────►   │
   │              BUILD;copper-wall;15;15;0\n             │
   │◄──────────────────────────────────── JSON state ─── │
   │  ...                         │                      │
   │                              │                      │
   ├─ [episódio termina]          │                      │
   ├─ env.reset() ────────────────────────────────────►  │
   │              RESET;Windswept Islands\n               │
   │  ...                         │                      │
```

---

## Considerações

- **`Vars.player` é null no server** — o mod já trata isso em todos os handlers (fallback para `Team.sharded`)
- **Thread safety** — o comando RESET deve executar no game thread (`Core.app.post`), não no thread TCP do mod
- **Startup race** — o Python deve aguardar alguns segundos após "Server loaded" antes de tentar conectar o TCP (o mod precisa inicializar)
- **Mapa não encontrado** — fallback para o primeiro mapa disponível, log de warning
- **`--no-server`** — útil para desenvolvimento: user inicia o server manualmente, Python só conecta
