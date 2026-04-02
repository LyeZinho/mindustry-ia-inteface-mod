# Mimi Gateway - Guia de Instalação em Português

Mod para Mindustry que expõe a API do jogo via socket TCP, permitindo que agentes RL e sistemas de IA percebam o estado do jogo e executem ações em tempo real.

---

## Instalação Rápida

### 1. Localize a pasta de mods do Mindustry

**Windows:**
```
%APPDATA%\Mindustry\mods\
```
(Ou: Abra o Explorador de Arquivos e digite `%APPDATA%` na barra de endereço)

**Linux:**
```
~/.local/share/Mindustry/mods/
```

**macOS:**
```
~/Library/Application Support/Mindustry/mods/
```

### 2. Copie os arquivos do mod

Crie uma pasta chamada `mimi-gateway` dentro da pasta de mods e copie:

```
~/.local/share/Mindustry/mods/mimi-gateway/
├── mod.hjson
└── scripts/
    └── main.js
```

**Opção A - Usando Git (recomendado):**
```bash
cd ~/.local/share/Mindustry/mods/
git clone https://github.com/seu-usuario/mindustry-ia-interface-mod.git mimi-gateway
```

**Opção B - Copiar manualmente:**
1. Baixe os arquivos do repositório como ZIP
2. Extraia em uma pasta chamada `mimi-gateway`
3. Copie para a pasta de mods do Mindustry

### 3. Inicie o Mindustry

1. Abra Mindustry
2. Inicie uma partida (Campanha ou Sandbox)
3. Pressione **F1** para abrir o console
4. Procure por esta mensagem:
   ```
   [Mimi Gateway] Servidor iniciado na porta 9000
   ```

Se ver a mensagem, a instalação foi bem-sucedida! ✅

---

## Solução de Problemas

### O mod não aparece na lista de mods

**Solução:**
- Verifique a estrutura de pastas:
  ```
  mods/
  └── mimi-gateway/
      ├── mod.hjson
      └── scripts/main.js
  ```
- Reinicie o Mindustry

### Vejo erro "Mod inválido"

**Solução:**
- Abra `mod.hjson` em um editor de texto
- Procure por erros de sintaxe JSON
- Valide em: https://jsonlint.com/

### Nenhuma mensagem no console (F1)

**Possíveis causas:**
1. Mod não está na pasta correta
2. Arquivo `scripts/main.js` está faltando
3. Erro de sintaxe no JavaScript

**Solução:**
- Verifique o caminho exato da pasta de mods
- Confirme que ambos os arquivos existem
- Verifique o `mod.hjson` para erros de sintaxe

### Erro: "Porta já está em uso"

**Se a porta 9000 já está ocupada:**

1. Abra `scripts/main.js` em um editor
2. Procure por: `port = 9000`
3. Mude para: `port = 9001` (ou outro número)
4. Salve e reinicie o Mindustry

---

## Conectando seu cliente Python

Depois que o mod estiver carregado, conecte seu cliente:

```python
import socket
import json

class ClienteMimi:
    def __init__(self, host='localhost', porta=9000):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, porta))
        print("Conectado ao Mimi Gateway!")
    
    def receber_estado(self):
        linha = self.socket.recv(4096).decode()
        if linha:
            return json.loads(linha)
        return None
    
    def enviar_comando(self, comando):
        self.socket.send(f"{comando}\n".encode())
    
    def construir(self, bloco, x, y, rotacao=0):
        self.enviar_comando(f"BUILD;{bloco};{x};{y};{rotacao}")

cliente = ClienteMimi()

while True:
    estado = cliente.receber_estado()
    if estado:
        print(f"Onda: {estado['wave']}, Saúde do Núcleo: {estado['core']['hp']}")
```

---

## Comandos Disponíveis

### BUILD - Construir um bloco
```
BUILD;nome_bloco;x;y;rotacao
```
Exemplo: `BUILD;duo;15;20;0`

### UNIT_MOVE - Mover uma unidade
```
UNIT_MOVE;id_unidade;x_destino;y_destino
```
Exemplo: `UNIT_MOVE;2;25;30`

### FACTORY - Spawnear unidade de fábrica
```
FACTORY;x_fabrica;y_fabrica;tipo_unidade
```
Exemplo: `FACTORY;10;12;poly`

### ATTACK - Comandar unidade para atacar
```
ATTACK;id_unidade;x_alvo;y_alvo
```
Exemplo: `ATTACK;2;30;35`

### STOP - Parar unidade
```
STOP;id_unidade
```
Ou parar todos: `STOP`

### REPAIR - Reparar construção
```
REPAIR;x;y
```

### DELETE - Desconstruir
```
DELETE;x;y
```

### UPGRADE - Melhorar bloco
```
UPGRADE;x;y
```

### MSG - Mensagem no chat
```
MSG;texto
```
Exemplo: `MSG;Construindo defesa no setor 7`

---

## Configuração Avançada

Edite `scripts/main.js` para ajustar:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `port` | 9000 | Porta do servidor TCP |
| `updateInterval` | 10 | Ticks entre atualizações de estado (~6 Hz) |
| `gridRadius` | 15 | Raio da grade em tiles (31x31) |
| `debug` | true | Logs verbosos |

---

## Suporte

Para problemas, consulte:
- **API_DOCUMENTATION.md** - Especificação completa do protocolo
- **DEPLOYMENT_GUIDE.md** - Guia detalhado (em inglês)
- **PYTHON_CLIENT_QUICKSTART.md** - Exemplos práticos

---

## Licença

Parte do projeto Mimi v2 AI Agent.
