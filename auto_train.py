#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import glob
import re
import yaml
from datetime import datetime, date

# ---------------------------------------------------------------------
# Función para leer configuración YAML
# ---------------------------------------------------------------------
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo de configuración: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ---------------------------------------------------------------------
# Busca el último episodio disponible en results/{asset}_training_metrics_ep*.csv
# ---------------------------------------------------------------------
def get_latest_episode_for_assets(assets):
    latest = 0
    for sym in assets:
        files = glob.glob(f"results/{sym}_training_metrics_ep*.csv")
        for fpath in files:
            m = re.search(r'ep(\d+)\.csv$', fpath)
            if m:
                try:
                    val = int(m.group(1))
                    if val > latest:
                        latest = val
                except:
                    pass
    return latest

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # -------- Cargar configuración desde config.yaml --------
    cfg = load_config("config.yaml")

    asset_group = cfg.get("asset_group", 'Crypto')
    assets = cfg.get("assets", ['BTCUSDT'])
    interval = cfg.get("interval", '1h')
    start_date = cfg.get("start_date", '2023-01-01')
    end_date = cfg.get("end_date", "auto")
    episodes = cfg.get("episodes", 600)
    save_freq = cfg.get("save_freq", 1)
    max_restarts = cfg.get("max_restarts", 20)
    mixed_precision = cfg.get("mixed_precision", False)

    if end_date == "auto":
        end_date = date.today().strftime('%Y-%m-%d')

    # Ruta al script que invoca train_agent
    train_bot_script = os.path.join(os.path.dirname(__file__), 'train_bot.py')

    restart_count = 0
    current_episode = get_latest_episode_for_assets(assets)

    # Log file
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/auto_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("="*60)
    print("AUTO-RESUME (MULTI-ACTIVO)")
    print(f"Assets: {assets}")
    print(f"Starting from episode {current_episode}/{episodes}")
    print("="*60)

    while current_episode < episodes and restart_count < max_restarts:
        with open(log_file, "a") as f:
            f.write(f"{datetime.now()}: Starting training from episode {current_episode}\n")

        # Monta el comando dinámicamente desde la config
        cmd = [
            sys.executable, train_bot_script,
            "--asset_group", asset_group,
            "--assets", *assets,
            "--interval", interval,
            "--start-date", start_date,
            "--end-date", end_date,
            "--episodes", str(episodes),
            "--save-freq", str(save_freq),
            "--resume-from", str(current_episode)
        ]

        if mixed_precision:
            cmd.append("--mixed-precision")

        print(f"🚀 Launching training process (resume {current_episode})...")
        try:
            p = subprocess.Popen(cmd)
            p.wait()

            if p.returncode == 0:
                print("✅ Training process exited normally (code 0).")
                break
            else:
                restart_count += 1
                print(f"⚠️ Training crashed with return code {p.returncode} — restart {restart_count}/{max_restarts}")
                with open(log_file, "a") as f:
                    f.write(f"{datetime.now()}: Crash returncode={p.returncode}. Restart {restart_count}\n")

                # Allow filesystem to settle then query latest episode
                time.sleep(5)
                current_episode = get_latest_episode_for_assets(assets)
                print(f"Will resume from episode {current_episode}")

        except KeyboardInterrupt:
            print("🛑 User interrupted auto-train. Exiting.")
            with open(log_file, "a") as f:
                f.write(f"{datetime.now()}: Manual interrupt.\n")
            break
        except Exception as e:
            restart_count += 1
            print(f"❌ Exception while launching training: {e}. Restart {restart_count}/{max_restarts}")
            with open(log_file, "a") as f:
                f.write(f"{datetime.now()}: Exception: {e}\n")
            time.sleep(5)
            current_episode = get_latest_episode_for_assets(assets)

    # Report final
    if current_episode >= episodes:
        print("🎉 All episodes finished. Training complete.")
    elif restart_count >= max_restarts:
        print("🚫 Max restarts reached; stopping auto-resume.")

if __name__ == "__main__":
    main()
