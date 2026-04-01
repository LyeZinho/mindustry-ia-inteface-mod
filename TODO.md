### 📂 Estrutura do Projeto do Mod  (TODO)
```text
mimi-gateway-mod/
├── mod.hjson            # Metadados do mod
├── icon.png             # Ícone para o gerenciador de mods
└── scripts/
    └── main.js          # O "Cérebro" que comunica com o Python
```

---

### 📝 TODO: Mindustry Gateway Mod

#### Fase 1: Fundação e Metadados
- [ ] **Criar `mod.hjson`**:
    - Definir `name: "mimi-gateway"`.
    - Definir `hidden: true` (para evitar conflitos em multiplayer).
    - Definir `minGameVersion: 146` (ou a versão que estiveres a usar).
- [ ] **Instalação Base**: Criar a pasta no diretório de mods do Mindustry e verificar se o jogo reconhece o mod (mesmo vazio).

#### Fase 2: O Servidor de Comunicação (Networking)
- [ ] **Importar Classes Java no `main.js`**:
    - `import { ServerSocket, PrintWriter, BufferedReader, InputStreamReader } from 'java.net'`.
- [ ] **Implementar Socket Server Assíncrono**:
    - Criar uma `Thread` separada no JavaScript (usando `Threads.daemon`) para não travar a interface do jogo.
    - Abrir uma porta (ex: `9000`) para o Mimi v2 se conectar.
- [ ] **Loop de Escuta**: Criar a lógica para aceitar conexões e manter o socket aberto.

#### Fase 3: Extração de Percepção (Inputs para a IA)
- [ ] **Snapshot do Estado Global**:
    - Criar uma função que lê `Vars.player.team().core().items` (Recursos).
    - Ler `Vars.player.team().data().getPowerGraph()` (Energia).
- [ ] **Mapeamento de Proximidade**:
    - Usar `Units.nearbyEnemies` para detetar ameaças num raio X.
    - Criar uma matriz simplificada (Grid) dos blocos ao redor do jogador/núcleo.
- [ ] **Serialização**: Converter esses dados Java para uma String JSON compacta para enviar via Socket.

#### Fase 4: Sistema de Execução (Outputs/Comandos)
- [ ] **Parser de Comandos**:
    - Criar um interpretador no `main.js` que recebe Strings do Python (ex: `"BUILD;duo;10;20"` ou `"MOVE;unitId;x;y"`).
- [ ] **Execução de Construção**:
    - Implementar `Vars.control.input.requestItem(...)` e `Call.constructBlock(...)` para que a IA possa construir fisicamente.
- [ ] **Controlo de Unidades**:
    - Usar `unit.plans.add(new BuildPlan(x, y, block, rotation))` para delegar tarefas às unidades Poly/Mega.

#### Fase 5: Sincronização com Mimi v2 (O "Handshake")
- [ ] **Event Trigger**:
    - Configurar o Mod para enviar o estado apenas a cada 10 frames (`Events.on(Trigger.update, ...)` + contador) para não sobrecarregar a rede.
- [ ] **Logs de Debug**: 
    - Usar `Log.info()` para ver no console do Mindustry se o Mimi v2 está a enviar comandos válidos.

---

### Exemplo de Código Base para o `scripts/main.js`:

```javascript
// Mimi v2 Gateway - Mindustry Mod
const port = 9000;

Threads.daemon(() => {
    try {
        let serverSocket = new java.net.ServerSocket(port);
        Log.info("Mimi Gateway: À espera de conexão na porta " + port);
        
        while(true) {
            let clientSocket = serverSocket.accept();
            let out = new java.io.PrintWriter(clientSocket.getOutputStream(), true);
            let input = new java.io.BufferedReader(new java.io.InputStreamReader(clientSocket.getInputStream()));

            // Loop de comunicação com o ActionBrain do Mimi
            while(!clientSocket.isClosed()){
                // 1. Enviar Estado
                let state = {
                    copper: Vars.player.team().core().items.get(Items.copper),
                    x: Vars.player.x,
                    y: Vars.player.y
                };
                out.println(JSON.stringify(state));

                // 2. Receber Comando
                let command = input.readLine();
                if(command) {
                    // Lógica para executar comando no jogo (ex: Call.constructBlock...)
                    Log.info("Mimi enviou comando: " + command);
                }
                
                java.lang.Thread.sleep(100); // 10Hz de atualização
            }
        }
    } catch(e) {
        Log.err("Erro no Gateway do Mimi: " + e);
    }
});
```
