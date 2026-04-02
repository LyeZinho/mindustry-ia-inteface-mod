# 📦 Releases - Mimi Gateway Mod

## Version 1.0.0

**Release Date**: April 2, 2026

### Download

- **[mimi-gateway-v1.0.0.zip](./mimi-gateway-v1.0.0.zip)** - Complete mod package (5.9 KB)

### What's Included

```
mimi-gateway/
├── mod.hjson          # Mod metadata
├── scripts/main.js    # Main implementation (25 KB)
└── INSTALL.txt        # Quick installation guide
```

## Version 1.0.0

**Release Date**: April 2, 2026

### Download

- **[mimi-gateway-v1.0.0.zip](./mimi-gateway-v1.0.0.zip)** - Complete mod package (5.9 KB)

### What's Included

```
mod.hjson          # Mod metadata
scripts/main.js    # Main implementation (25 KB)
INSTALL.txt        # Quick installation guide
```

### Installation Steps

**Option 1: From ZIP (Recommended)**

1. Download **[mimi-gateway-v1.0.0.zip](./mimi-gateway-v1.0.0.zip)**
2. Extract the ZIP file
3. Rename the extracted folder to `mimi-gateway`
4. Copy to your Mindustry mods folder:
   - Windows: `%APPDATA%\Mindustry\mods\`
   - Linux: `~/.local/share/Mindustry/mods/`
   - macOS: `~/Library/Application Support/Mindustry/mods/`
5. Launch Mindustry and start a game
6. Check F1 console for: `[Mimi Gateway] Servidor iniciado na porta 9000`

**Option 2: From Git**

```bash
cd ~/.local/share/Mindustry/mods/
git clone https://github.com/seu-usuario/mindustry-ia-interface-mod.git mimi-gateway
```

### Requirements

- Mindustry v146 or later
- TCP port 9000 available (or configure in `scripts/main.js`)
- Python 3.6+ for client (optional, for testing)

### Features

✅ TCP socket server on port 9000  
✅ Real-time game state streaming (JSON)  
✅ 9 commands for game control (BUILD, UNIT_MOVE, ATTACK, etc.)  
✅ Connection auto-reconnect (5 attempts)  
✅ Full command validation  
✅ Mindustry v146+ compatible  

### Documentation

- **[GUIA_INSTALACAO_PT.md](./GUIA_INSTALACAO_PT.md)** - Installation guide in Portuguese
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** - Detailed setup guide (English)
- **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** - Complete protocol reference
- **[PYTHON_CLIENT_QUICKSTART.md](./PYTHON_CLIENT_QUICKSTART.md)** - Python examples

### Changelog

#### v1.0.0
- ✅ Initial stable release
- ✅ 9 commands implemented (BUILD, UNIT_MOVE, FACTORY, ATTACK, STOP, REPAIR, DELETE, UPGRADE, MSG)
- ✅ Full state perception (game, resources, units, buildings, grid)
- ✅ Connection resilience with auto-reconnect
- ✅ Comprehensive documentation and Python test client
- ✅ Command validation framework
- ✅ Error handling (15+ defensive patterns)

### Installation Verification Checklist

After installation, verify with this checklist:

- [ ] Mod folder exists: `mods/mimi-gateway/`
- [ ] Files present: `mod.hjson` and `scripts/main.js`
- [ ] Mindustry loaded the mod (F1 console message)
- [ ] Python client can connect: `python3 test_mimi_client.py`
- [ ] All 9 commands work (BUILD, UNIT_MOVE, etc.)

### Troubleshooting

**Mod not loading?**
- Check folder structure: `mods/mimi-gateway/mod.hjson`
- Validate `mod.hjson` syntax with https://jsonlint.com/

**Connection refused on port 9000?**
- Verify mod loaded (F1 console)
- Check if port 9000 is in use
- Try changing port in `scripts/main.js`: `port = 9001`

**Commands not executing?**
- Ensure game is running (not paused)
- Verify player has resources for builds
- Check command syntax in API_DOCUMENTATION.md

### Support

For issues, questions, or feature requests:
1. Check **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** for protocol reference
2. Review **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** for setup help
3. Run **test_mimi_client.py** to verify installation

### License

MIT License - See LICENSE file for details
