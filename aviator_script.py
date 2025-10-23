import os
import time
import csv
import pickle
import random
import threading
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------- SELENIUM ----------------------
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ---------------------- DATABASE ----------------------
from database import log_bet_result, set_running, is_running, init_db

# ---------------------- ML & ANALYTICS ----------------------
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from joblib import dump, load


init_db()
# ---------------------- DASHBOARD LOGGING ----------------------
log_messages = []
MAX_LOGS = 100  # keep more logs for dashboard

def add_log(message):
    """Add message to in-memory logs and print to console."""
    global log_messages
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    
    # Always print to console
    print(entry)
    
    # Define important keywords
    important = ["login", "iframe", "bet", "win", "loss", "stopped", "error", "appended", "payout", "started", "balance"]
    
    # Always include important logs or cycle summaries
    if any(k in message.lower() for k in important) or message.startswith("--- Cycle"):
        log_messages.append(entry)
    
    # Trim if exceeds max length
    if len(log_messages) > MAX_LOGS:
        log_messages = log_messages[-MAX_LOGS:]  
# ---------------------- CONFIG ----------------------
URL = "https://www.betika.com"
AVIATOR_URL = "https://www.betika.com/en-ke/aviator"
CSV_FILE = "payout_history2.csv"
BET_RESULTS_FILE = "bet_results2.csv"
PICKLE_FILE = "payouts2.pkl"
MODEL_FILE = "lstm_payout_model2.keras"

bot_running = False
bot_thread = None

# ===================== LSTM SETUP =====================
SEQ_LEN = 10
TRAIN_INTERVAL = 90  # 2.5 minutes
MAX_PICKLE_LEN = 10000
_cached_model = None
_scaler = None
_scaler = None
_cached_model = None
MODEL_FILE_XGB = "xgb_model.joblib"
MODEL_FILE_RF = "rf_model.joblib"
HISTORY_FILE = "training_history.csv"
# ---------------------- UTILITIES ----------------------




# ---------------------- DATA HANDLING ----------------------
def save_payouts_to_pickle(new_payouts):
    try:
        if not new_payouts:
            return

        data = []
        if os.path.exists(PICKLE_FILE):
            with open(PICKLE_FILE, "rb") as f:
                data = pickle.load(f)

        data = [float(x) for x in data]
        new_payouts = [float(x) for x in new_payouts]
        recent_check_set = set(data[-100:])
        unique_new = [p for p in new_payouts if p not in recent_check_set]

        if unique_new:
            data.extend(unique_new)
            if len(data) > MAX_PICKLE_LEN:
                data = data[-MAX_PICKLE_LEN:]

            with open(PICKLE_FILE, "wb") as f:
                pickle.dump(data, f)

            add_log(f"✅ Added {len(unique_new)} new payouts | Total: {len(data)} samples.")
        else:
            add_log("ℹ️ No new unique payouts this round.")
    except Exception as e:
        add_log(f"⚠️ save_payouts_to_pickle error: {e}")


def load_payouts_from_pickle():
    if os.path.exists(PICKLE_FILE):
        try:
            with open(PICKLE_FILE, "rb") as f:
                return pickle.load(f)
        except:
            return []
    return []


HISTORY_FILE = "training_history.csv"

# Must exist in your environment:
#   add_log(msg)                -> your custom logger function
#   load_payouts_from_pickle()  -> returns list/array of payout floats


# ---------------------- CLASS CONVERSION ----------------------
def payout_to_class(value):
    """Converts a payout value into one of three class indices."""
    if value < 2.0:
        return 0  # Blue
    elif 2.0 <= value < 10.0:
        return 1  # Purple
    else:
        return 2  # Pink

# ---------------------- DATA PREPARATION ----------------------

# ---------------------- ADAPTIVE MODEL CHOICE ----------------------
def _choose_model(data):
    """
    Chooses between XGBoost and RandomForest dynamically
    based on volatility (XGB for high volatility).
    """
    std_recent = np.std(data[-50:]) if len(data) >= 50 else np.std(data)
    mean_recent = np.mean(data[-50:]) if len(data) >= 50 else np.mean(data)
    volatility = std_recent / (mean_recent + 1e-6)

    if volatility > 0.5:
        model_type = "xgb"
    else:
        model_type = "rf"
    add_log(f"📈 Volatility={volatility:.3f} → Using {model_type.upper()} model")
    return model_type

# ---------------------- TRAIN MODEL ----------------------
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# ---------------------- IMPROVED DATA PREPARATION ----------------------
def _prepare_classification_data(data):
    """
    Converts payout data to sequence features for classification.
    Applies rolling mean smoothing to reduce noise.
    """
    global _scaler
    data = np.array(data, dtype=float)
    n = len(data)
    if n < SEQ_LEN + 1:
        raise ValueError("Not enough data for training")

    # Smooth data to reduce randomness
    smooth_data = pd.Series(data).rolling(window=3, min_periods=1).mean().values

    # Normalize with persistent scaler
    if _scaler is None:
        _scaler = StandardScaler()
        norm_data = _scaler.fit_transform(smooth_data.reshape(-1, 1)).flatten()
        add_log("⚙️ Scaler fitted and cached.")
    else:
        norm_data = _scaler.transform(smooth_data.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(n - SEQ_LEN):
        seq = norm_data[i:i + SEQ_LEN]
        target_value = smooth_data[i + SEQ_LEN]
        target_class = payout_to_class(target_value)
        X.append(seq)
        y.append(target_class)

    return np.array(X), np.array(y)


# ---------------------- IMPROVED TRAINING FUNCTION ----------------------
def _train_model(data):
    """Fine-tunes and trains the model with adaptive tuning."""
    model_type = _choose_model(data)
    X, y = _prepare_classification_data(data)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    y_classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=y_classes, y=y)
    class_weights = dict(enumerate(weights))

    if model_type == "xgb":
        base_model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )

        # Fine-tuning grid
        param_grid = {
            "n_estimators": [150, 250, 350],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9]
        }

        grid = GridSearchCV(
            base_model, param_grid, cv=tscv, scoring="f1_weighted", n_jobs=-1, verbose=0
        )
        grid.fit(X, y, sample_weight=[class_weights[i] for i in y])
        model = grid.best_estimator_
        add_log(f"🎯 Best XGB Params: {grid.best_params_}")
        dump(model, MODEL_FILE_XGB)
        add_log("✅ Fine-tuned XGBoost model trained and saved.")
    else:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        dump(model, MODEL_FILE_RF)
        add_log("✅ Fine-tuned Random Forest model trained and saved.")

    # Calibrate probabilities for realistic confidence
    calibrated = CalibratedClassifierCV(model, cv=tscv)
    calibrated.fit(X, y)
    _cached_model = calibrated

    # Evaluate
    preds = calibrated.predict(X)
    report = classification_report(y, preds, target_names=["Blue", "Purple", "Pink"], digits=3)
    add_log("📊 Training Report:\n" + report)

    df = pd.DataFrame({
        "timestamp": [pd.Timestamp.now()],
        "model_type": [model_type],
        "samples": [len(X)],
        "accuracy": [np.mean(preds == y)]
    })
    if os.path.exists(HISTORY_FILE):
        df_existing = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

    return calibrated


# ---------------------- MODEL LOADING ----------------------
def _load_or_get_model():
    """Loads the cached model if available."""
    global _cached_model
    if _cached_model is None:
        if os.path.exists(MODEL_FILE_XGB):
            _cached_model = load(MODEL_FILE_XGB)
            add_log("🧠 XGBoost model loaded from disk.")
        elif os.path.exists(MODEL_FILE_RF):
            _cached_model = load(MODEL_FILE_RF)
            add_log("🧠 Random Forest model loaded from disk.")
        else:
            add_log("⚠️ No trained model found.")
            return None
    return _cached_model

# ---------------------- PREDICTION ----------------------
def predict_next_payout():
    """Predicts the next payout class (Blue, Purple, Pink)."""
    model = _load_or_get_model()
    if model is None or _scaler is None:
        return None, 0.0, None

    data = np.array(load_payouts_from_pickle(), dtype=float)
    if len(data) < SEQ_LEN:
        return None, 0.0, None

    norm_data = _scaler.transform(data.reshape(-1, 1)).flatten()
    arr = norm_data[-SEQ_LEN:].reshape(1, -1)

    preds = model.predict_proba(arr)[0]
    pred_class = int(np.argmax(preds))
    conf = float(np.max(preds))

    color_map = {0: "blue", 1: "purple", 2: "pink"}
    color = color_map.get(pred_class, "gray")

    add_log(f"🎯 Predicted: {color} | Confidence={conf:.2f}")
    return pred_class, conf, color

# ---------------------- PERIODIC TRAINING ----------------------
def train_models_periodically():
    """Continuously trains models every few minutes."""
    while True:
        try:
            data = np.array(load_payouts_from_pickle(), dtype=float)
            if len(data) >= SEQ_LEN + 5:
                _train_model(data)
            else:
                add_log("ℹ️ Not enough data to train yet.")
        except Exception as e:
            add_log(f"⚠️ Training error: {e}")
        time.sleep(TRAIN_INTERVAL)

# ---------------------- START TRAINING THREAD ----------------------
threading.Thread(target=train_models_periodically, daemon=True).start()
# ===================== ORIGINAL BOT CODE =====================
def get_current_multiplier(driver):
    try:
        el = driver.find_element(By.XPATH, "/html/body/app-root/app-game/div/div[1]/div[1]/app-header/div/div[2]/div[1]/span[1]")
        value_text = el.text.replace(",", "")
        value = float(value_text)
        print(f"Current balance : {value}")
        return value
    except NoSuchElementException:
        print("Element not found for current balance.")
        return None
    except ValueError:
        print(f"Could not convert '{value_text}' to float.")
        return None

def start_bot(bet_amount, phone, password, check_interval, check_duration):
    global bot_running, bot_thread
    if bot_running:
        return "Bot already running."
    bot_running = True
    bot_thread = threading.Thread(
        target=run_bot,
        args=(bet_amount, phone, password, check_interval, check_duration),
        daemon=True
    )
    bot_thread.start()
    return "Bot started successfully."

def stop_bot():
    global bot_running
    bot_running = False
    return "Bot stopped."

def is_running():
    return bot_running

def is_bet_active(driver):
    try:
        return bool(driver.find_elements(By.XPATH, "//button/span/label[contains(text(),'Cancel')]"))
    except:
        return False

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import shutil

chrome_path = shutil.which("chromium")
driver_path = shutil.which("chromedriver")




service = Service(driver_path)

# ---------------------- CORE BOT LOGIC ----------------------
def run_bot(bet_amount, phone, password, check_interval, check_duration):
    global bot_running
    try:
        options = webdriver.ChromeOptions()
        chrome_path = shutil.which("chromium")
        driver_path = shutil.which("chromedriver")

        options.binary_location = chrome_path
      

        service = Service(driver_path) 
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--start-maximized")
        options.add_argument("--headless") 
        options.add_argument("--disable-infobars")
        options.add_argument("--enable-unsafe-swiftshader")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-blink-features=AutomationControlled")
        prefs = {
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False
        }
        options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(service=service, options=options)
        driver.get(URL)
        wait_and_click(driver, "/html/body/div[3]/div[1]/header/div[1]/div[2]/div[1]/a[1]", "Login button")
        time.sleep(2)

        try:
            phone_input = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
                (By.XPATH, "/html/body/div[3]/div[1]/div[1]/div[2]/div/div/div[2]/div[1]/input")))
            password_input = driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div[1]/div[2]/div/div/div[2]/div[2]/div/input")
            phone_input.send_keys(phone)
            password_input.send_keys(password)
            wait_and_click(driver, "/html/body/div[3]/div[1]/div[1]/div[2]/div/div/div[2]/div[4]/button", "Submit login")
            add_log("✅ Login successful!")
        except Exception as e:
            add_log(f"❌ Login error:{e} ")

        time.sleep(5)
        driver.get(AVIATOR_URL)
        time.sleep(4)
        switch_to_game_iframe(driver)

        bet = False
        last_payout = None

        while bot_running:
            add_log(f"--- Cycle started at {datetime.now().strftime('%H:%M:%S')} ---")
            start_time = time.time()

            while time.time() - start_time < check_duration * 60 and bot_running:
                try:
                    payouts = get_recent_payouts(driver)
                    if not payouts:
                        time.sleep(check_interval)
                        continue

                    log_payouts(payouts)

                    if bet or is_bet_active(driver):
                        time.sleep(check_interval)
                        continue

                    if should_bet(payouts):  # LSTM-enhanced decision
                        add_log("📈 Conditions met — placing bet...")
                        place_bet(driver, bet_amount)

                        current_multiplier = get_current_multiplier(driver)
                        if current_multiplier and current_multiplier > 0:
                            add_log(f"✅ Current Balance :{current_multiplier} — placing bet!")
                            bet = True
                            last_payout = payouts[0]
                            add_log(f"✅ Bet placed at payout {last_payout}x")

                            result_payout = wait_for_result_by_index(driver, last_payout)
                            if result_payout is not None:
                                if isinstance(result_payout, str):
                                    result_payout = float(result_payout.replace(',', '').strip())

                                if result_payout > bet_amount:
                                    profit_loss = round(result_payout - bet_amount, 2)
                                    log_bet_result(result_payout, bet_amount, "WIN", profit_loss)
                                    add_log(f"🏆 WIN — payout {result_payout}x | P/L: {bet_amount*10}")
                                else:
                                    profit_loss = round(-bet_amount, 2)
                                    log_bet_result(result_payout, bet_amount, "LOSS", profit_loss)
                                    add_log(f"❌ LOSS — payout {result_payout}x | P/L: {-10}")
                            else:
                                add_log("⚠️ No result found after waiting.")
                            bet = False
                        else:
                            add_log(f"❌ Current Balance: {current_multiplier} is insufficient, bet not placed!")
                except Exception as e:
                    add_log(f"⚠️ Error in bot cycle: {e}")
                time.sleep(check_interval)
    except Exception as e:
        add_log(f"Bot error: {e}")
        bot_running = False

# ---------------------- INDEX-BASED RESULT ----------------------
def wait_for_result_by_index(driver, last_payout):
    start_wait = time.time()
    max_wait_time = 180
    while time.time() - start_wait < max_wait_time:
        payouts = get_recent_payouts(driver)
        if payouts:
            try:
                bet_index = payouts.index(last_payout)
                result_index = bet_index - 2
                if result_index >= 0:
                    result_payout = payouts[result_index]
                    print(f"Detected result payout {result_payout}x for bet at {last_payout}x")
                    return result_payout
            except ValueError:
                pass
        time.sleep(5)
    return None

# ---------------------- UTILITIES ----------------------
def place_bet(driver, bet_amount):
    wait_and_click(driver, "/html/body/app-root/app-game/div/div[1]/div[2]/div/div[2]/div[4]/app-bet-controls/div/app-bet-control[1]/div/div[1]/div[1]/app-navigation-switcher/div/button[2]", "3rd button")
    wait_and_click(driver, "/html/body/app-root/app-game/div/div[1]/div[2]/div/div[2]/div[4]/app-bet-controls/div/app-bet-control[1]/div/div[3]/div/div[2]/div[1]/app-ui-switcher/div", "4th button")
    safe_enter_text(driver, "/html/body/app-root/app-game/div/div[1]/div[2]/div/div[2]/div[4]/app-bet-controls/div/app-bet-control[1]/div/div[3]/div/div[2]/div[2]/div/app-spinner/div/div[2]/input", str(bet_amount))
    wait_and_click(driver, "/html/body/app-root/app-game/div/div[1]/div[2]/div/div[2]/div[4]/app-bet-controls/div/app-bet-control[1]/div/div[1]/div[2]/div[2]", "Bet button")

# ---------------------- HELPER FUNCTIONS ----------------------
def wait_and_click(driver, xpath, name, timeout=40):
    try:
        element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
        time.sleep(0.3)
        element.click()
        print(f"{name} clicked successfully.")
        return True
    except TimeoutException:
        print(f"Timeout — {name} not found.")
        return False
    except Exception as e:
        print(f"Error clicking {name}: ")
        return False

def safe_enter_text(driver, xpath, value):
    try:
        textbox = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, xpath)))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", textbox)
        textbox.click()
        textbox.send_keys(Keys.CONTROL, "a")
        textbox.send_keys(Keys.BACKSPACE)
        textbox.send_keys(value)
        print(f"Entered amount: {value}")
    except Exception as e:
        print(f"Error entering text: ")


def switch_to_game_iframe(driver):
    try:
        driver.switch_to.default_content()
        iframe = WebDriverWait(driver, 15).until(lambda d: d.find_elements(By.TAG_NAME, "iframe"))
        if iframe:
            driver.switch_to.frame(iframe[0])
            print("Switched to game iframe.")
    except Exception:
        add_log(f"Iframe switch error:")

def get_recent_payouts(driver):
    try:
        wait_and_click(driver, "/html/body/app-root/app-game/div/div/div[2]/div/div[2]/div[2]/app-stats-widget/div/div[2]/div/div", "Dropdown toggle")
        time.sleep(2)
        container = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "/html/body/app-root/app-game/div/div/div[2]/div/div[2]/div[2]/app-stats-widget/div/app-stats-dropdown/div/div[2]")))
        text = driver.execute_script("return arguments[0].innerText;", container)
        parts = text.replace("\n", " ").split()
        values = [float(p.replace("x", "").replace(",", "")) for p in parts if "x" in p]
        return values
    except Exception:
        add_log(f"Payout read failed:")
        return []
    


# ---------------------- LSTM-ENHANCED should_bet ----------------------
def should_bet(payouts):
    if len(payouts) < 3:
        return False

    save_payouts_to_pickle(payouts)
    last3 = payouts[:3]  # use last 3 recent payouts, not first 3
    mean_val = sum(last3) / 3.0

    pred_class, conf, color = predict_next_payout()

    if pred_class is not None and conf >= 0.75:
        # Decision logic based on predicted class
        if pred_class == 0:   # blue (1.00–1.99)
            cond = False
        elif pred_class == 1: # purple (2.00–9.99)
            cond = True
        else:                 # pink (≥10)
            cond = True  # rare, high payout — strong bet signal

        reason = f"LSTM Class={pred_class} ({color}), Conf={conf:.2f}"
    else:
        cond = False
        reason = f"model — skipping bet (Pred={color}, Conf={conf:.2f})"

    add_log(f"Last 3: {last3} | Mean={mean_val:.2f} | {reason} → {'BET' if cond else 'SKIP'}")
    return cond



def is_bet_active(driver):
    try:
        return bool(driver.find_elements(By.XPATH, "//button/span/label[contains(text(),'Cancel')]"))
    except:
        return False

def log_payouts(new_payouts):
    """Append only *new* payouts to payouts.csv (avoid duplicates)."""
    if not new_payouts:
        return

    # Load existing payouts to compare
    existing = set()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    existing.add(row[1])  # store only payout value strings

    # Filter new unique values
    unique_payouts = [p for p in new_payouts if str(p) not in existing]

    # Append only new ones
    if unique_payouts:
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for p in unique_payouts:
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), p])

        add_log(f"✅ Appended {len(unique_payouts)} new payouts.")
    else:
        add_log("ℹ️ No new payouts to append.")



