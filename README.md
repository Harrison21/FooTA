# FooTA: A Synthetic-to-Real Dataset for Team-Level Football Performance Assessment

This is the official PyTorch implementation of the paper **"FooTA: A Synthetic-to-Real Dataset for Team-Level Football Performance Assessment"**. 

### Data structure (FooTA)

Use one directory per **league** under a FooTA root (for example `../FooTA`). In the layout below, pre-extracted player-centric video features (`.npy`), team performance labels, and player bounding-box traces sit **together** under each league. In `configs/train_football.py`, set `football_dirs` and `football_meta_dirs` to the league paths that contain your `.npy` files and stats JSONs (often the same list), and set `bbox_root` to the FooTA root so graph mode can find `*_rough_player_positions.json`.

**Synthetic leagues** (`Premier League`, `LaLiga`, `Bundesliga`, `Ligue_1`): matches are grouped by broadcast round. Each round directory holds one row of files per match (same basename for `.npy`, stats JSON, and bbox JSON).

**Real-world leagues** (`england_epl`, `france_league`): each match is one subdirectory named by match id; features, a single stats JSON, and bbox JSONs for that match live inside that folder. Add these league folder names to `football_leagues` / `football_dirs` in the config when you use them.

```text
$FOOTBALL_ROOT
├── Premier League
│   ├── 1st
│   │   ├── 1st_Arsenal_vs_Sunderland.npy
│   │   ├── 1st_Arsenal_vs_Sunderland_team_stats.json
│   │   └── 1st_Arsenal_vs_Sunderland_rough_player_positions.json
│   ├── 2nd
│   │   └── ...
│   └── 7th
│       └── ...
├── LaLiga
├── Bundesliga
├── Ligue_1
├── france_league
├── england_epl
    └── 2023-08-12_newcastle-united-aston-villa-premier-league
        ├── 2023-08-12_newcastle-united-aston-villa-premier-league.npy
        ├── 2023-08-12_newcastle-united-aston-villa-premier-league.json
        ├── newcastle-united-aston-villa-premier-league_full_rough_player_positions.json
        ├── newcastle-united-aston-villa-premier-league_1_rough_player_positions.json
        └── newcastle-united-aston-villa-premier-league_2_rough_player_positions.json
```

## 🛠️ Installation

### Step 1: Create and activate conda environment ###

```bash
conda create -n FooTA python=3.8 -y
conda activate FooTA
```
### Step 2: Install dependencies ###

```bash
pip install -r requirements.txt
```
### Step 3: Training ###
To train the model, run:
```bash
python3 main.py --config configs/train_football.py
