// ============================================================================
// Mimi v2 Gateway - Mindustry Mod
// ============================================================================
// This mod acts as the Peripheral Nervous System for the Mimi v2 RL agent.
// It exposes the Mindustry internal API via an async TCP socket server.
// ============================================================================

const config = {
    port: 9000,
    updateInterval: 10,
    gridRadius: 15,
    debug: true
};

// ============================================================================
// GLOBAL STATE
// ============================================================================
let serverSocket = null;
let clientSocket = null;
let outputWriter = null;
let inputReader = null;
let isConnected = false;
let tickCounter = 0;

// ============================================================================
// PHASE 2: Socket Server Implementation
// ============================================================================
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 5000;

function startSocketServer() {
    Threads.daemon(() => {
        reconnectAttempts = 0;
        
        while (reconnectAttempts < maxReconnectAttempts) {
            try {
                if (serverSocket != null && !serverSocket.isClosed()) {
                    serverSocket.close();
                }
                
                serverSocket = new java.net.ServerSocket(config.port);
                Log.info("[Mimi Gateway] Servidor iniciado na porta " + config.port);
                reconnectAttempts = 0;
                
                while (true) {
                    clientSocket = null;
                    isConnected = false;
                    
                    try {
                        Log.info("[Mimi Gateway] Aguardando conexão...");
                        clientSocket = serverSocket.accept();
                        clientSocket.setSoTimeout(0);
                        
                        let outputStream = clientSocket.getOutputStream();
                        outputWriter = new java.io.PrintWriter(outputStream, true);
                        
                        let inputStream = clientSocket.getInputStream();
                        inputReader = new java.io.BufferedReader(new java.io.InputStreamReader(inputStream));
                        
                        isConnected = true;
                        reconnectAttempts = 0;
                        Log.info("[Mimi Gateway] Mimi v2 conectado!");
                        
                        communicationLoop();
                        
                    } catch (e) {
                        isConnected = false;
                        if (config.debug) Log.info("[Mimi Gateway] Erro na comunicação: " + e);
                        try { java.lang.Thread.sleep(2000); } catch(sleep) {}
                    } finally {
                        try { if (inputReader != null) inputReader.close(); } catch(e) {}
                        try { if (outputWriter != null) outputWriter.close(); } catch(e) {}
                        try { if (clientSocket != null && !clientSocket.isClosed()) clientSocket.close(); } catch(e) {}
                    }
                }
                
            } catch (e) {
                reconnectAttempts++;
                Log.err("[Mimi Gateway] Erro no servidor (tentativa " + reconnectAttempts + "/" + maxReconnectAttempts + "): " + e);
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    try { java.lang.Thread.sleep(reconnectDelay); } catch(sleep) {}
                } else {
                    Log.err("[Mimi Gateway] Máximo de tentativas de reconexão atingido");
                    break;
                }
            }
        }
    });
}

function communicationLoop() {
    while (isConnected && clientSocket != null && !clientSocket.isClosed()) {
        try {
            // PHASE 5: Send state periodically
            if (tickCounter >= config.updateInterval) {
                sendGameState();
                tickCounter = 0;
            }
            tickCounter++;
            
            // PHASE 4: Process commands from Python
            if (inputReader != null) {
                // Non-blocking read check
                let ready = inputReader.ready();
                if (ready) {
                    let command = inputReader.readLine();
                    if (command != null && command.length() > 0) {
                        processCommand(command);
                    }
                }
            }
            
            java.lang.Thread.sleep(100); // ~10Hz loop
            
        } catch (e) {
            if (config.debug) Log.info("[Mimi Gateway] Loop erro: " + e);
            break;
        }
    }
}

function cleanup() {
    isConnected = false;
    try { if (outputWriter != null) outputWriter.close(); } catch(e) {}
    try { if (inputReader != null) inputReader.close(); } catch(e) {}
    try { if (clientSocket != null) clientSocket.close(); } catch(e) {}
    try { if (serverSocket != null) serverSocket.close(); } catch(e) {}
    Log.info("[Mimi Gateway] Conexão encerrada");
}

// ============================================================================
// PHASE 3: Perception Extraction
// ============================================================================
function captureGameState() {
    let state = {
        tick: java.lang.System.currentTimeMillis(),
        time: Vars.state.tick,
        wave: Vars.state.wave,
        waveTime: Vars.state.wavetime,
        resources: {},
        power: { produced: 0, consumed: 0, stored: 0, capacity: 0 },
        core: { hp: 0, x: 0, y: 0, size: 0 },
        player: { x: 0, y: 0 },
        enemies: [],
        friendlyUnits: [],
        buildings: [],
        grid: []
    };
    
    try {
        let team = Vars.player != null ? Vars.player.team() : null;
        if (team == null) {
            team = Vars.state.rules.teams.get(Team.sharded);
        }
        
        let core = team != null ? team.core() : null;
        if (core != null && core.items != null) {
            let itemsArray = Vars.content.items();
            for (let i = 0; i < itemsArray.size(); i++) {
                let item = itemsArray.get(i);
                let amount = core.items.get(item);
                state.resources[item.name] = amount;
            }
            state.core.hp = core.health / core.maxHealth;
            state.core.x = Math.floor(core.x / 8);
            state.core.y = Math.floor(core.y / 8);
            state.core.size = core.block.size;
        }
        
        if (Vars.player != null) {
            state.player.x = Math.floor(Vars.player.x / 8);
            state.player.y = Math.floor(Vars.player.y / 8);
        }
        
        if (team != null) {
            let powerGraph = team.data().power;
            if (powerGraph != null) {
                state.power.produced = Math.floor(powerGraph.getPowerProduced() * 100) / 100;
                state.power.consumed = Math.floor(powerGraph.getPowerUsed() * 100) / 100;
                state.power.stored = Math.floor(powerGraph.getPowerStored() * 100) / 100;
                state.power.capacity = Math.floor(powerGraph.getPowerCapacity() * 100) / 100;
            }
        }
        
        let centerX = state.core.x;
        let centerY = state.core.y;
        let radius = config.gridRadius;
        let tileSize = 8;
        
        Units.nearbyEnemies(Team.sharded, centerX * tileSize, centerY * tileSize, radius * tileSize, (unit) => {
            state.enemies.push({
                id: unit.id,
                type: unit.type.name,
                hp: Math.floor((unit.health / unit.maxHealth) * 100) / 100,
                x: Math.floor(unit.x / 8),
                y: Math.floor(unit.y / 8),
                command: unit.command != null ? unit.command.name : "idle"
            });
        });
        
        if (team != null) {
            Units.nearby(team, centerX * tileSize, centerY * tileSize, radius * tileSize, (unit) => {
                state.friendlyUnits.push({
                    id: unit.id,
                    type: unit.type.name,
                    hp: Math.floor((unit.health / unit.maxHealth) * 100) / 100,
                    x: Math.floor(unit.x / 8),
                    y: Math.floor(unit.y / 8),
                    command: unit.command != null ? unit.command.name : "idle"
                });
            });
        }
        
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dy = -radius; dy <= radius; dy++) {
                let worldX = (centerX + dx) * tileSize;
                let worldY = (centerY + dy) * tileSize;
                let tile = Vars.world.tile(worldX, worldY);
                
                if (tile != null) {
                    let block = tile.block();
                    let build = tile.build;
                    state.grid.push({
                        x: dx,
                        y: dy,
                        block: block != null ? block.name : "air",
                        floor: tile.floor().name,
                        team: build != null && build.team != null ? build.team.name : "neutral",
                        hp: build != null ? Math.floor((build.health / build.maxHealth) * 100) / 100 : 0,
                        rotation: build != null ? build.rotation : 0
                    });
                }
            }
        }
        
        let allTeams = [Team.sharded, Team.crux, Team.derelict];
        allTeams.forEach(t => {
            let buildings = t.buildings();
            if (buildings != null) {
                buildings.forEach(b => {
                    if (b.x >= (centerX - radius) * tileSize && b.x <= (centerX + radius) * tileSize &&
                        b.y >= (centerY - radius) * tileSize && b.y <= (centerY + radius) * tileSize) {
                        state.buildings.push({
                            block: b.block.name,
                            team: b.team.name,
                            x: Math.floor(b.x / 8),
                            y: Math.floor(b.y / 8),
                            hp: Math.floor((b.health / b.maxHealth) * 100) / 100,
                            rotation: b.rotation
                        });
                    }
                });
            }
        });
        
    } catch (e) {
        if (config.debug) Log.info("[Mimi Gateway] Erro ao capturar estado: " + e);
    }
    
    return state;
}

function sendGameState() {
    if (!isConnected || outputWriter == null) return;
    
    try {
        let state = captureGameState();
        let jsonState = JSON.stringify(state);
        outputWriter.println(jsonState);
        
        if (config.debug) {
            // Log only first 100 chars to avoid spam
            if (jsonState.length > 100) {
                Log.info("[Mimi Gateway] Estado enviado: " + jsonState.substring(0, 100) + "...");
            } else {
                Log.info("[Mimi Gateway] Estado enviado: " + jsonState);
            }
        }
    } catch (e) {
        if (config.debug) Log.info("[Mimi Gateway] Erro ao enviar estado: " + e);
    }
}

// ============================================================================
// PHASE 4: Command Parser and Execution
// ============================================================================
function validateCommand(commandStr) {
    if (commandStr == null || commandStr.length === 0) {
        return { valid: false, error: "Comando vazio" };
    }
    
    if (commandStr.length > 1000) {
        return { valid: false, error: "Comando muito longo (máx 1000 caracteres)" };
    }
    
    let parts = commandStr.split(";");
    if (parts.length === 0) {
        return { valid: false, error: "Formato de comando inválido" };
    }
    
    let cmd = parts[0].toUpperCase();
    let validCommands = ["BUILD", "UNIT_MOVE", "MSG", "ATTACK", "STOP", "FACTORY", "REPAIR", "DELETE", "UPGRADE"];
    
    if (validCommands.indexOf(cmd) === -1) {
        return { valid: false, error: "Comando desconhecido: " + cmd };
    }
    
    return { valid: true, command: cmd, parts: parts };
}

function processCommand(commandStr) {
    let validation = validateCommand(commandStr);
    
    if (!validation.valid) {
        if (config.debug) Log.info("[Mimi Gateway] Comando inválido: " + validation.error);
        return;
    }
    
    if (config.debug) Log.info("[Mimi Gateway] Comando recebido: " + commandStr);
    
    try {
        let commandType = validation.command;
        let parts = validation.parts;
        
        switch (commandType) {
            case "BUILD":
                handleBuildCommand(parts);
                break;
            case "UNIT_MOVE":
                handleUnitMoveCommand(parts);
                break;
            case "MSG":
                handleMessageCommand(parts);
                break;
            case "ATTACK":
                handleAttackCommand(parts);
                break;
            case "STOP":
                handleStopCommand(parts);
                break;
            case "FACTORY":
                handleFactoryCommand(parts);
                break;
            case "REPAIR":
                handleRepairCommand(parts);
                break;
            case "DELETE":
                handleDeleteCommand(parts);
                break;
            case "UPGRADE":
                handleUpgradeCommand(parts);
                break;
        }
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao processar comando: " + e);
        if (config.debug) Log.err(e.stack);
    }
}

function handleBuildCommand(parts) {
    if (parts.length < 4) {
        Log.info("[Mimi Gateway] BUILD: parâmetros insuficientes");
        return;
    }
    
    let blockName = parts[1];
    let targetX = parseInt(parts[2]);
    let targetY = parseInt(parts[3]);
    let rotation = parts[4] ? parseInt(parts[4]) : 0;
    
    let blockType = Vars.content.blocks().find(b => b.name === blockName);
    if (blockType == null) {
        Log.info("[Mimi Gateway] Bloco não encontrado: " + blockName);
        return;
    }
    
    let worldTile = Vars.world.tile(targetX, targetY);
    if (worldTile == null) {
        Log.info("[Mimi Gateway] Tile inválido: " + targetX + "," + targetY);
        return;
    }
    
    try {
        let team = Vars.player != null ? Vars.player.team() : Team.sharded;
        let canPlace = Vars.control.input.canPlace(worldTile, blockType, rotation, team);
        
        if (canPlace) {
            Call.constructBlock(team, worldTile, blockType, rotation);
            Log.info("[Mimi Gateway] Construir: " + blockName + " em " + targetX + "," + targetY);
        } else {
            Log.info("[Mimi Gateway] Não pode construir: " + blockName + " em " + targetX + "," + targetY);
        }
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao construir: " + e);
    }
}

function handleUnitMoveCommand(parts) {
    // UNIT_MOVE;unit_id;target_x;target_y
    if (parts.length < 4) {
        Log.info("[Mimi Gateway] UNIT_MOVE: parâmetros insuficientes");
        return;
    }
    
    let unitId = parseInt(parts[1]);
    let targetX = parseFloat(parts[2]);
    let targetY = parseFloat(parts[3]);
    
    // Find the unit - iterate through all nearby units to locate
    let unit = null;
    let found = false;
    
    try {
        // Search through all allied units
        let allUnits = Vars.state.teams.active.map(t => t.units).flat();
        for (let i = 0; i < allUnits.size; i++) {
            let u = allUnits.get(i);
            if (u != null && u.id === unitId) {
                unit = u;
                found = true;
                break;
            }
        }
        
        if (!found || unit == null) {
            Log.info("[Mimi Gateway] Unidade não encontrada: " + unitId);
            return;
        }
        
        // Move the unit to the target position (world coordinates)
        let worldX = targetX * 8;
        let worldY = targetY * 8;
        
        unit.moveTo(worldX, worldY);
        Log.info("[Mimi Gateway] Mover unidade " + unitId + " para " + targetX + "," + targetY);
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao mover unidade " + unitId + ": " + e);
        if (config.debug) Log.err(e.stack);
    }
}

function handleMessageCommand(parts) {
    // MSG;text
    if (parts.length < 2) return;
    
    let text = parts.slice(1).join(";");
    Call.sendMessage("[cyan][Mimi v2]:[] " + text);
    Log.info("[Mimi Gateway] Mensagem enviada: " + text);
}

function handleAttackCommand(parts) {
    // ATTACK;unit_id;target_x;target_y
    if (parts.length < 4) {
        Log.info("[Mimi Gateway] ATTACK: parâmetros insuficientes");
        return;
    }
    
    let unitId = parseInt(parts[1]);
    let targetX = parseFloat(parts[2]);
    let targetY = parseFloat(parts[3]);
    
    try {
        let unit = null;
        let allUnits = Vars.state.teams.active.map(t => t.units).flat();
        for (let i = 0; i < allUnits.size; i++) {
            let u = allUnits.get(i);
            if (u != null && u.id === unitId) {
                unit = u;
                break;
            }
        }
        
        if (unit == null) {
            Log.info("[Mimi Gateway] ATTACK: unidade não encontrada: " + unitId);
            return;
        }
        
        let targetTile = Vars.world.tile(targetX, targetY);
        if (targetTile != null && targetTile.build != null) {
            unit.target(targetTile.build);
            Log.info("[Mimi Gateway] Ataque ordenado: unidade " + unitId + " -> bloco em " + targetX + "," + targetY);
        } else {
            Log.info("[Mimi Gateway] ATTACK: alvo inválido em " + targetX + "," + targetY);
        }
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao atacar: " + e);
        if (config.debug) Log.err(e.stack);
    }
}

function handleStopCommand(parts) {
    try {
        if (parts.length < 2 || parts[1] === "" || parts[1] == null) {
            let allTeams = Vars.state.teams.active;
            allTeams.forEach(team => {
                if (team != null && team.units != null) {
                    team.units.forEach(u => {
                        if (u != null) {
                            u.clearCommand();
                        }
                    });
                }
            });
            Log.info("[Mimi Gateway] Todas as unidades paradas");
        } else {
            let unitId = parseInt(parts[1]);
            let found = false;
            
            let allTeams = Vars.state.teams.active;
            allTeams.forEach(team => {
                if (team != null && team.units != null) {
                    team.units.forEach(u => {
                        if (u != null && u.id === unitId) {
                            u.clearCommand();
                            found = true;
                        }
                    });
                }
            });
            
            if (found) {
                Log.info("[Mimi Gateway] Unidade " + unitId + " parada");
            } else {
                Log.info("[Mimi Gateway] Unidade não encontrada: " + unitId);
            }
        }
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao parar unidades: " + e);
        if (config.debug) Log.err(e.stack);
    }
}

function handleFactoryCommand(parts) {
    if (parts.length < 3) {
        Log.info("[Mimi Gateway] FACTORY: parâmetros insuficientes");
        return;
    }
    
    let factoryX = parseInt(parts[1]);
    let factoryY = parseInt(parts[2]);
    let unitType = parts[3] ? parts[3] : "poly";
    
    try {
        let factoryTile = Vars.world.tile(factoryX, factoryY);
        if (factoryTile == null || factoryTile.build == null) {
            Log.info("[Mimi Gateway] FACTORY: tile inválido");
            return;
        }
        
        let factory = factoryTile.build;
        let factoryBlock = factory.block;
        
        if (factoryBlock == null || !factoryBlock.name.includes("factory") && !factoryBlock.name.includes("reconstructor")) {
            Log.info("[Mimi Gateway] FACTORY: não é uma fábrica: " + (factoryBlock != null ? factoryBlock.name : "null"));
            return;
        }
        
        let unitTypeObj = Vars.content.units().find(u => u.name === unitType);
        if (unitTypeObj == null) {
            Log.info("[Mimi Gateway] FACTORY: tipo de unidade não encontrado: " + unitType);
            return;
        }
        
        let team = factory.team != null ? factory.team : Vars.player != null ? Vars.player.team() : Team.sharded;
        
        Call.effect("spawn", factory.x, factory.y);
        Call.createUnit(team, unitTypeObj, factory.x, factory.y);
        
        Log.info("[Mimi Gateway] FÁBRICA: " + unitType + " em " + factoryX + "," + factoryY);
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao produzir unidade: " + e);
        if (config.debug) Log.err(e.stack);
    }
}

function handleRepairCommand(parts) {
    if (parts.length < 3) {
        Log.info("[Mimi Gateway] REPAIR: parâmetros insuficientes");
        return;
    }
    
    let targetX = parseInt(parts[1]);
    let targetY = parseInt(parts[2]);
    let tile = Vars.world.tile(targetX, targetY);
    
    if (tile == null || tile.build == null) {
        Log.info("[Mimi Gateway] REPAIR: bloco não encontrado");
        return;
    }
    
    let team = Vars.player != null ? Vars.player.team() : Team.sharded;
    let build = tile.build;
    
    if (build.health < build.maxHealth) {
        Call.setHealth(build, build.health + build.maxHealth * 0.5);
        Log.info("[Mimi Gateway] REPAIR: " + targetX + "," + targetY);
    }
}

function handleDeleteCommand(parts) {
    if (parts.length < 3) {
        Log.info("[Mimi Gateway] DELETE: parâmetros insuficientes");
        return;
    }
    
    let targetX = parseInt(parts[1]);
    let targetY = parseInt(parts[2]);
    let tile = Vars.world.tile(targetX, targetY);
    
    if (tile == null || tile.build == null) {
        Log.info("[Mimi Gateway] DELETE: bloco não encontrado");
        return;
    }
    
    let team = Vars.player != null ? Vars.player.team() : Team.sharded;
    
    try {
        Call.deconstructFinish(team, tile.build);
        tile.removeBlock();
        Log.info("[Mimi Gateway] DELETE: " + targetX + "," + targetY);
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao remover bloco: " + e);
    }
}

function handleUpgradeCommand(parts) {
    if (parts.length < 3) {
        Log.info("[Mimi Gateway] UPGRADE: parâmetros insuficientes");
        return;
    }
    
    let targetX = parseInt(parts[1]);
    let targetY = parseInt(parts[2]);
    
    try {
        let tile = Vars.world.tile(targetX, targetY);
        if (tile == null || tile.build == null) {
            Log.info("[Mimi Gateway] UPGRADE: bloco não encontrado");
            return;
        }
        
        let build = tile.build;
        let block = build.block;
        
        if (block == null || block.upgrades == null || block.upgrades.length === 0) {
            Log.info("[Mimi Gateway] UPGRADE: bloco não tem upgrades");
            return;
        }
        
        let upgradePath = block.upgrades;
        if (upgradePath == null || upgradePath.length < 2) {
            Log.info("[Mimi Gateway] UPGRADE: caminho de upgrade inválido");
            return;
        }
        
        let nextBlock = upgradePath[1];
        if (nextBlock == null) {
            Log.info("[Mimi Gateway] UPGRADE: próximo bloco não encontrado");
            return;
        }
        
        Call.deconstructFinish(build.team, build);
        Call.constructBlock(build.team, tile, nextBlock, build.rotation);
        
        Log.info("[Mimi Gateway] UPGRADE: " + targetX + "," + targetY + " -> " + nextBlock.name);
    } catch (e) {
        Log.err("[Mimi Gateway] Erro ao fazer upgrade: " + e);
        if (config.debug) Log.err(e.stack);
    }
}

// ============================================================================
// PHASE 5: Event Triggers
// ============================================================================
Events.on(Trigger.update, () => {
    // Only increment when player is in game
    if (Vars.state.isGame()) {
        tickCounter++;
    }
});

// ============================================================================
// MOD INITIALIZATION
// ============================================================================
function init() {
    Log.info("==============================================");
    Log.info("[Mimi Gateway] v1.0.0 - Inicializando...");
    Log.info("[Mimi Gateway] Porta: " + config.port);
    Log.info("[Mimi Gateway] Intervalo de atualização: " + config.updateInterval + " ticks");
    Log.info("==============================================");
    
    startSocketServer();
}

// Start the mod
init();