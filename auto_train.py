#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import glob
import re
from datetime import datetime, date

# Busca el último episodio disponible en results/{asset}_training_metrics_ep*.csv
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

def main():
    # -------- Configuración (ajusta según tu caso) --------
    assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']   # <-- lista de activos a entrenar
    interval = '1h'
    start_date = '2023-01-01'
    end_date = date.today().strftime('%Y-%m-%d')
    episodes = 3000
    save_freq = 1
    max_restarts = 20

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

        # Monta el comando: pasamos --assets ... y --resume-from current_episode
        cmd = [
            sys.executable, train_bot_script,
            "--assets", *assets,
            "--interval", interval,
            "--start-date", start_date,
            "--end-date", end_date,
            "--episodes", str(episodes),
            "--save-freq", str(save_freq),
            "--resume-from", str(current_episode),
            "--mixed-precision"
        ]

        print(f"Launching training process (resume {current_episode})...")
        try:
            p = subprocess.Popen(cmd)
            p.wait()

            if p.returncode == 0:
                print("Training process exited normally (code 0).")
                break
            else:
                restart_count += 1
                print(f"Training crashed with return code {p.returncode} — restart {restart_count}/{max_restarts}")
                with open(log_file, "a") as f:
                    f.write(f"{datetime.now()}: Crash returncode={p.returncode}. Restart {restart_count}\n")

                # Allow filesystem to settle then query latest episode
                time.sleep(5)
                current_episode = get_latest_episode_for_assets(assets)
                print(f"Will resume from episode {current_episode}")

        except KeyboardInterrupt:
            print("User interrupted auto-train. Exiting.")
            with open(log_file, "a") as f:
                f.write(f"{datetime.now()}: Manual interrupt.\n")
            break
        except Exception as e:
            restart_count += 1
            print(f"Exception while launching training: {e}. Restart {restart_count}/{max_restarts}")
            with open(log_file, "a") as f:
                f.write(f"{datetime.now()}: Exception: {e}\n")
            time.sleep(5)
            current_episode = get_latest_episode_for_assets(assets)

    # Report
    if current_episode >= episodes:
        print("All episodes finished. Training complete.")
    elif restart_count >= max_restarts:
        print("Max restarts reached; stopping auto-resume.")

if __name__ == "__main__":
    main()
