"""
nsl_kdd_loader.py
Loads & preprocesses the NSL-KDD dataset.
Exposes a clean API consumed by phase1, phase2, phase3 and app.py.
"""

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# NSL-KDD column names (41 features + label + difficulty)
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

ATTACK_MAP = {
    "normal":           "Normal",
    # DoS
    "back":"DoS","land":"DoS","neptune":"DoS","pod":"DoS","smurf":"DoS",
    "teardrop":"DoS","mailbomb":"DoS","apache2":"DoS","processtable":"DoS","udpstorm":"DoS",
    # Probe
    "ipsweep":"Probe","nmap":"Probe","portsweep":"Probe","satan":"Probe",
    "mscan":"Probe","saint":"Probe",
    # R2L
    "ftp_write":"R2L","guess_passwd":"R2L","imap":"R2L","multihop":"R2L",
    "phf":"R2L","spy":"R2L","warezclient":"R2L","warezmaster":"R2L",
    "sendmail":"R2L","named":"R2L","snmpattack":"R2L","snmpguess":"R2L",
    "worm":"R2L","xlock":"R2L","xsnoop":"R2L","httptunnel":"R2L",
    # U2R
    "buffer_overflow":"U2R","loadmodule":"U2R","perl":"U2R","rootkit":"U2R",
    "ps":"U2R","sqlattack":"U2R","xterm":"U2R",
}

CATEGORICAL = ["protocol_type", "service", "flag"]


class NSLKDDLoader:
    def __init__(self, csv_path="cleaned_nsl_kdd.csv", txt_path="KDDTrain+.txt"):
        self.csv_path = csv_path
        self.txt_path = txt_path

        # populated after load()
        self.df_raw      = None
        self.df          = None
        self.X           = None
        self.y_binary    = None   # 0=normal, 1=anomaly
        self.y_multiclass= None   # 0=Normal,1=DoS,2=Probe,3=R2L,4=U2R
        self.scaler      = StandardScaler()
        self.encoders    = {}
        self.feature_names = []

        # stats
        self.total_records   = 0
        self.normal_records  = 0
        self.anomaly_records = 0
        self.num_features    = 0
        self.missing_values  = 0

    # ── public ────────────────────────────────────────────────────────────────

    def load(self):
        """Load data from CSV (preferred) or fallback to KDDTrain+.txt."""
        try:
            self.df_raw = pd.read_csv(self.csv_path)
            print(f"[Loader] Loaded from {self.csv_path}: {len(self.df_raw)} rows")
        except FileNotFoundError:
            self.df_raw = pd.read_csv(
                self.txt_path, names=COLUMNS
            ).drop(columns=["difficulty"], errors="ignore")
            print(f"[Loader] Loaded from {self.txt_path}: {len(self.df_raw)} rows")

        self._preprocess()
        return self

    def get_train_test_binary(self, test_size=0.3, random_state=42):
        return train_test_split(
            self.X, self.y_binary,
            test_size=test_size, random_state=random_state, stratify=self.y_binary
        )

    def get_train_test_multiclass(self, test_size=0.3, random_state=42):
        # only anomaly rows for phase 2
        mask = self.y_binary == 1
        X_a  = self.X[mask]
        y_a  = self.y_multiclass[mask]
        return train_test_split(
            X_a, y_a,
            test_size=test_size, random_state=random_state, stratify=y_a
        )

    def sample_events(self, n=20):
        """Return n random rows as event dicts for live simulation."""
        sample = self.df_raw.sample(n=n).copy()
        events = []
        for _, row in sample.iterrows():
            proto   = row.get("protocol_type", "TCP")
            service = row.get("service", "http")
            label   = str(row.get("label", "normal")).lower()
            cat     = ATTACK_MAP.get(label, "Unknown")
            events.append({
                "src_ip":      self._fake_ip(),
                "protocol":    str(proto).upper(),
                "service":     str(service),
                "bytes":       int(row.get("src_bytes", 0)),
                "label":       label,
                "attack_cat":  cat if cat != "Normal" else "—",
                "features":    self._row_to_features(row),
            })
        return events

    # ── private ───────────────────────────────────────────────────────────────

    def _preprocess(self):
        df = self.df_raw.copy()

        # drop difficulty if present
        df.drop(columns=["difficulty"], errors="ignore", inplace=True)

        # ensure label column exists
        label_col = "label" if "label" in df.columns else df.columns[-1]
        df.rename(columns={label_col: "label"}, inplace=True)

        self.missing_values = int(df.isnull().sum().sum())
        df.fillna(0, inplace=True)

        # binary label
        df["binary"] = df["label"].apply(
            lambda x: 0 if str(x).strip().lower() == "normal" else 1
        )

        # multiclass label
        cat_map = {"Normal":0,"DoS":1,"Probe":2,"R2L":3,"U2R":4}
        df["multiclass"] = df["label"].apply(
            lambda x: cat_map.get(ATTACK_MAP.get(str(x).strip().lower(), "Normal"), 0)
        )

        # encode categoricals
        for col in CATEGORICAL:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le

        feature_cols = [c for c in df.columns if c not in ("label","binary","multiclass")]
        self.feature_names = feature_cols
        self.num_features  = len(feature_cols)

        X = df[feature_cols].values.astype(np.float32)
        X = self.scaler.fit_transform(X)

        self.X            = X
        self.y_binary     = df["binary"].values
        self.y_multiclass = df["multiclass"].values
        self.df           = df

        self.total_records   = len(df)
        self.normal_records  = int((self.y_binary == 0).sum())
        self.anomaly_records = int((self.y_binary == 1).sum())

        print(f"[Loader] Features: {self.num_features} | "
              f"Normal: {self.normal_records} | Anomaly: {self.anomaly_records}")

    def _row_to_features(self, row):
        """Convert a raw dataframe row to a scaled feature vector."""
        feat_vals = []
        for col in self.feature_names:
            val = row.get(col, 0)
            if col in self.encoders:
                try:
                    val = self.encoders[col].transform([str(val)])[0]
                except ValueError:
                    val = 0
            feat_vals.append(float(val))
        arr = np.array(feat_vals, dtype=np.float32).reshape(1, -1)
        return self.scaler.transform(arr)[0].tolist()

    @staticmethod
    def _fake_ip():
        return f"192.168.{random.randint(0,4)}.{random.randint(1,254)}"