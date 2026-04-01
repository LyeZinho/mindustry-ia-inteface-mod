# 🛰️ Mimi v2: Mindustry Gateway Mod

Este mod atua como o **Sistema Nervoso Periférico** para o agente **Mimi v2**. Ele expõe a API interna do Mindustry (Java/Arc) via um servidor Socket assíncrono, permitindo que modelos de **Aprendizado por Reforço (RL)** e o **ActionBrain** do Mimi percebam o estado do mundo e executem ações em tempo real sem o uso de OCR.

## 🧠 Arquitetura de Integração

O mod funciona como um **Proxy de Percepção e Atuação**:
1. **Percepção:** Coleta dados de recursos, grade de energia, posição de unidades e ameaças.
2. **Comunicação:** Serializa esses dados em JSON e os envia via TCP para o pipeline Python do Mimi.
3. **Atuação:** Recebe comandos brutos do `ActionBrain` (ex: `BUILD`, `MOVE`, `ATTACK`) e os traduz em chamadas nativas `Call.constructBlock` ou `UnitControl`.

---

## 🛠️ Funcionalidades Implementadas

- [ ] **Socket Server (Porta 9000):** Servidor TCP daemonizado para comunicação não-bloqueante.
- [ ] **State Streaming:** Envio de snapshots do estado do jogo a cada 10 ticks (6Hz/10Hz).
- [ ] **Action Interpreter:** Parser de comandos para construção e movimentação de unidades.
- [ ] **Headless Friendly:** Compatível com `server.jar` para treinamento acelerado de IA.

---

## 📂 Estrutura do Diretório

```text
mimi-gateway/
├── mod.hjson           # Metadados do Mod
├── icon.png            # Identidade visual no gerenciador de mods
└── scripts/
    ├── main.js         # Entry point e gerenciamento de Threads
    ├── perception.js   # Lógica de extração de dados do mapa (Inputs)
    └── actuator.js     # Lógica de execução de ordens da IA (Outputs)
```

---

## 📡 Protocolo de Comunicação

### 1. Mensagem de Estado (Mod → Mimi)
O mod envia um objeto JSON contendo:
```json
{
  "resources": {"copper": 450, "lead": 120, "graphite": 0},
  "power": {"produced": 1200, "consumed": 800, "stored": 5000},
  "core_hp": 0.95,
  "enemies_nearby": 3,
  "grid_snapshot": [...] 
}
```

### 2. Comandos de Ação (Mimi → Mod)
O Mimi v2 deve enviar strings formatadas no socket:
* `BUILD;block_name;x;y;rotation`
* `UNIT_MOVE;unit_id;target_x;target_y`
* `MSG;text` (Exibe uma mensagem no chat do jogo vinda do agente)

---

## 🚀 Como Instalar

1. Certifique-se de que o **Mimi v2** (Python) está configurado e com o `ActionBrain` pronto para ouvir na porta 9000.
2. Clone este repositório na sua pasta de mods do Mindustry:
   * **Windows:** `%appdata%/Mindustry/mods/`
   * **Linux:** `~/.local/share/Mindustry/mods/`
3. Inicie o jogo e verifique o log (`F1`) para confirmar a mensagem: `Mimi Gateway: À espera de conexão na porta 9000`.

---

## 🧪 Treinamento de RL

Para usar este mod como ambiente de treino:
1. Rode o Mindustry com a flag `-no-graphics`.
2. Conecte seu script de treinamento (ex: Stable-Baselines3 + Gymnasium) ao socket.
3. Utilize o comando `fps` do console do Mindustry para acelerar o tempo de treino.

---

## 📜 Licença
Este mod é parte integrante do projeto **Mimi v2** e segue a mesma licença do agente.

---

> **Nota:** Este mod é estritamente funcional e **hidden: true**, o que significa que ele não aparecerá na lista de conteúdos de campanha para não interferir na experiência padrão, a menos que seja evocado pelo agente.
