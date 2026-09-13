import glob
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# NEXORA 2026 - FINAL ML SOLUTION
# ============================================================

DATA_DIR = "data"

TELEMETRY_PATH = r"data\telemetry\month=*\*.parquet"

FIELD_VISITS_FILE = r"data\field_visits.csv"
METER_FILE = r"data\meter_read_success.csv"
MASTER_FILE = r"data\gateway_master.csv"

OUTPUT_FILE = "predictions.csv"

WINDOW_DAYS = 28


# Official challenge prediction weeks
OFFICIAL_WEEKS = pd.date_range(
    "2026-02-02",
    "2026-03-23",
    freq="7D"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_gateway_id(series):

    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
        .str.strip()
    )


def safe_rate(numerator, denominator):

    denominator = denominator.replace(
        0,
        np.nan
    )

    return (
        numerator / denominator
    ).fillna(0)


# ============================================================
# LOAD TELEMETRY
# ============================================================

def load_telemetry():

    print()
    print("Loading telemetry...")

    telemetry_files = sorted(
        glob.glob(TELEMETRY_PATH)
    )

    if not telemetry_files:

        raise FileNotFoundError(
            f"No telemetry parquet files found: "
            f"{TELEMETRY_PATH}"
        )

    print(
        "Telemetry files found:",
        len(telemetry_files)
    )

    telemetry_parts = []

    for file in telemetry_files:

        print(
            "  Reading:",
            file
        )

        df = pd.read_parquet(file)

        if "gateway_id" not in df.columns:

            raise ValueError(
                f"gateway_id missing from {file}"
            )

        if "ts_utc" not in df.columns:

            raise ValueError(
                f"ts_utc missing from {file}"
            )

        df["gateway_id"] = normalize_gateway_id(
            df["gateway_id"]
        )

        df["ts_utc"] = pd.to_datetime(
            df["ts_utc"],
            utc=True,
            errors="coerce"
        )

        df = df.dropna(
            subset=["ts_utc"]
        )

        telemetry_parts.append(df)

    telemetry = pd.concat(
        telemetry_parts,
        ignore_index=True
    )

    print(
        "Telemetry rows loaded:",
        len(telemetry)
    )

    print(
        "Telemetry date range:",
        telemetry["ts_utc"].min(),
        "to",
        telemetry["ts_utc"].max()
    )

    return telemetry


# ============================================================
# GET NUMERIC TELEMETRY COLUMNS
# ============================================================

def get_numeric_telemetry_columns(
    telemetry
):

    excluded = {
        "gateway_id",
        "ts_utc",
        "DateDt"
    }

    numeric_columns = [
        c
        for c in telemetry.select_dtypes(
            include=np.number
        ).columns
        if c not in excluded
    ]

    return numeric_columns


# ============================================================
# BUILD DECISION-TIME TELEMETRY FEATURES
# ============================================================

def build_decision_time_telemetry_features(
    telemetry,
    prediction_weeks,
    numeric_columns
):

    print()
    print(
        "Building decision-time telemetry features..."
    )

    all_features = []

    for prediction_week in prediction_weeks:

        window_start = (
            prediction_week
            - pd.Timedelta(
                days=WINDOW_DAYS
            )
        )

        window_end = prediction_week

        window_start_utc = (
            window_start
            .tz_localize("UTC")
        )

        window_end_utc = (
            window_end
            .tz_localize("UTC")
        )

        window = telemetry[
            (telemetry["ts_utc"] >= window_start_utc)
            &
            (telemetry["ts_utc"] < window_end_utc)
        ].copy()

        print(
            f"  {prediction_week.date()}: "
            f"{len(window):,} telemetry rows"
        )

        if window.empty:

            continue

        # ----------------------------------------------------
        # Aggregate numeric telemetry
        # ----------------------------------------------------

        grouped = (
            window
            .groupby("gateway_id")[numeric_columns]
            .agg(
                [
                    "mean",
                    "sum",
                    "max",
                    "min"
                ]
            )
        )

        grouped.columns = [
            f"{col}_{stat}"
            for col, stat in grouped.columns
        ]

        features = grouped.reset_index()

        # ----------------------------------------------------
        # Telemetry coverage
        # ----------------------------------------------------

        coverage = (
            window
            .groupby("gateway_id")
            .size()
            .rename(
                "telemetry_records"
            )
            .reset_index()
        )

        coverage["telemetry_coverage"] = (
            coverage["telemetry_records"]
            / 672
        ).clip(
            upper=1
        )

        features = features.merge(
            coverage,
            on="gateway_id",
            how="left"
        )

        # ----------------------------------------------------
        # RSSI rates
        # ----------------------------------------------------

        rssi_columns = [
            "rssi_bad_sum",
            "rssi_normal_sum",
            "rssi_good_sum"
        ]

        if all(
            c in features.columns
            for c in rssi_columns
        ):

            total = (
                features["rssi_bad_sum"]
                + features["rssi_normal_sum"]
                + features["rssi_good_sum"]
            )

            features["rssi_bad_rate"] = safe_rate(
                features["rssi_bad_sum"],
                total
            )

            features["rssi_normal_rate"] = safe_rate(
                features["rssi_normal_sum"],
                total
            )

            features["rssi_good_rate"] = safe_rate(
                features["rssi_good_sum"],
                total
            )

        # ----------------------------------------------------
        # RSRP rates
        # ----------------------------------------------------

        rsrp_columns = [
            "rscp_rsrp_bad_sum",
            "rscp_rsrp_normal_sum",
            "rscp_rsrp_good_sum"
        ]

        if all(
            c in features.columns
            for c in rsrp_columns
        ):

            total = (
                features["rscp_rsrp_bad_sum"]
                + features["rscp_rsrp_normal_sum"]
                + features["rscp_rsrp_good_sum"]
            )

            features["rsrp_bad_rate"] = safe_rate(
                features["rscp_rsrp_bad_sum"],
                total
            )

            features["rsrp_normal_rate"] = safe_rate(
                features["rscp_rsrp_normal_sum"],
                total
            )

            features["rsrp_good_rate"] = safe_rate(
                features["rscp_rsrp_good_sum"],
                total
            )

        # ----------------------------------------------------
        # RSRQ rates
        # ----------------------------------------------------

        rsrq_columns = [
            "ecio_rsrq_bad_sum",
            "ecio_rsrq_normal_sum",
            "ecio_rsrq_good_sum"
        ]

        if all(
            c in features.columns
            for c in rsrq_columns
        ):

            total = (
                features["ecio_rsrq_bad_sum"]
                + features["ecio_rsrq_normal_sum"]
                + features["ecio_rsrq_good_sum"]
            )

            features["rsrq_bad_rate"] = safe_rate(
                features["ecio_rsrq_bad_sum"],
                total
            )

            features["rsrq_normal_rate"] = safe_rate(
                features["ecio_rsrq_normal_sum"],
                total
            )

            features["rsrq_good_rate"] = safe_rate(
                features["ecio_rsrq_good_sum"],
                total
            )

        # ----------------------------------------------------
        # Network rates
        # ----------------------------------------------------

        network_columns = [
            "network_2g_sum",
            "network_3g_sum",
            "network_4g_sum",
            "network_unknown_sum"
        ]

        if all(
            c in features.columns
            for c in network_columns
        ):

            total = sum(
                features[c]
                for c in network_columns
            )

            features["network_2g_rate"] = safe_rate(
                features["network_2g_sum"],
                total
            )

            features["network_3g_rate"] = safe_rate(
                features["network_3g_sum"],
                total
            )

            features["network_4g_rate"] = safe_rate(
                features["network_4g_sum"],
                total
            )

            features["network_unknown_rate"] = safe_rate(
                features["network_unknown_sum"],
                total
            )

        # ----------------------------------------------------
        # Event rates
        # ----------------------------------------------------

        if "reboot_cnt_sum" in features.columns:

            features["reboot_rate"] = safe_rate(
                features["reboot_cnt_sum"],
                features["telemetry_records"]
            )

        if "disconnection_cnt_sum" in features.columns:

            features["disconnection_rate"] = safe_rate(
                features["disconnection_cnt_sum"],
                features["telemetry_records"]
            )

        # ----------------------------------------------------
        # Load rates
        # ----------------------------------------------------

        if "load1_bigger1_sum" in features.columns:

            features["load1_bigger1_rate"] = safe_rate(
                features["load1_bigger1_sum"],
                features["telemetry_records"]
            )

        if "load1_bigger2_sum" in features.columns:

            features["load1_bigger2_rate"] = safe_rate(
                features["load1_bigger2_sum"],
                features["telemetry_records"]
            )

        # ----------------------------------------------------
        # Week
        # ----------------------------------------------------

        features["week_start"] = prediction_week

        all_features.append(
            features
        )

    if not all_features:

        raise ValueError(
            "No telemetry features were created."
        )

    features = pd.concat(
        all_features,
        ignore_index=True
    )

    print(
        "Decision-time telemetry rows:",
        len(features)
    )

    return features


# ============================================================
# ADD METER FEATURES
# ============================================================

def add_meter_features(
    features,
    meter,
    prediction_weeks
):

    print()
    print("Loading meter-read data...")

    meter = meter.copy()

    meter["gateway_id"] = normalize_gateway_id(
        meter["gateway_id"]
    )

    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        errors="coerce"
    )

    meter_features = []

    for prediction_week in prediction_weeks:

        previous = meter[
            meter["week_start"]
            < prediction_week
        ].copy()

        if previous.empty:

            continue

        latest_week = (
            previous["week_start"]
            .max()
        )

        latest = previous[
            previous["week_start"]
            == latest_week
        ].copy()

        latest["read_rate"] = np.where(
            latest["meters_expected"] > 0,
            latest["meters_read"]
            / latest["meters_expected"],
            0
        )

        latest = latest[
            [
                "gateway_id",
                "read_rate"
            ]
        ].copy()

        latest["week_start"] = prediction_week

        meter_features.append(
            latest
        )

    if meter_features:

        meter_features = pd.concat(
            meter_features,
            ignore_index=True
        )

        features = features.merge(
            meter_features,
            on=[
                "gateway_id",
                "week_start"
            ],
            how="left"
        )

    else:

        features["read_rate"] = 0

    features["read_rate"] = (
        features["read_rate"]
        .fillna(0)
    )

    return features


# ============================================================
# ADD GATEWAY MASTER FEATURES
# ============================================================

def add_master_features(
    features,
    master
):

    print()
    print("Loading gateway master...")

    master = master.copy()

    master["gateway_id"] = normalize_gateway_id(
        master["gateway_id"]
    )

    for c in [
        "fw_updated_on",
        "installed_on",
        "decommissioned_on"
    ]:

        if c in master.columns:

            master[c] = pd.to_datetime(
                master[c],
                errors="coerce"
            )

    master_columns = [
        "gateway_id",
        "tenant",
        "site_type",
        "region",
        "hw_model",
        "antenna_type",
        "fw_version",
        "fw_updated_on",
        "installed_on",
        "decommissioned_on",
        "n_meters_installed"
    ]

    master_columns = [
        c
        for c in master_columns
        if c in master.columns
    ]

    master_subset = master[
        master_columns
    ].copy()

    features = features.merge(
        master_subset,
        on="gateway_id",
        how="left"
    )

    # --------------------------------------------------------
    # Firmware age
    # --------------------------------------------------------

    if "fw_updated_on" in features.columns:

        features["firmware_age_days"] = (
            features["week_start"]
            - features["fw_updated_on"]
        ).dt.days

        features["firmware_age_days"] = (
            features["firmware_age_days"]
            .clip(lower=0)
            .fillna(-1)
        )

    # --------------------------------------------------------
    # Installation age
    # --------------------------------------------------------

    if "installed_on" in features.columns:

        features["installation_age_days"] = (
            features["week_start"]
            - features["installed_on"]
        ).dt.days

        features["installation_age_days"] = (
            features["installation_age_days"]
            .clip(lower=0)
            .fillna(-1)
        )

    # --------------------------------------------------------
    # Decommissioned flag
    # --------------------------------------------------------

    if "decommissioned_on" in features.columns:

        features["is_decommissioned"] = (
            features["decommissioned_on"].notna()
            &
            (
                features["decommissioned_on"]
                < features["week_start"]
            )
        ).astype(int)

    # --------------------------------------------------------
    # Drop raw date columns
    # --------------------------------------------------------

    features = features.drop(
        columns=[
            c
            for c in [
                "fw_updated_on",
                "installed_on",
                "decommissioned_on"
            ]
            if c in features.columns
        ]
    )

    return features


# ============================================================
# ADD HISTORICAL VISIT FEATURES
# ============================================================

def add_visit_history(
    features,
    visits,
    prediction_weeks
):

    print()
    print("Loading historical field visits...")

    visits = visits.copy()

    visits["gateway_id"] = normalize_gateway_id(
        visits["gateway_id"]
    )

    visits["requested_on"] = pd.to_datetime(
        visits["requested_on"],
        errors="coerce"
    )

    visits["visited_on"] = pd.to_datetime(
        visits["visited_on"],
        errors="coerce"
    )

    history_rows = []

    for prediction_week in prediction_weeks:

        # Requests before prediction week
        prior_requests = visits[
            visits["requested_on"]
            < prediction_week
        ]

        # Completed visits before prediction week
        prior_completed = visits[
            visits["visited_on"].notna()
            &
            (
                visits["visited_on"]
                < prediction_week
            )
        ]

        # Confirmed faults before prediction week
        prior_faults = prior_completed[
            prior_completed["outcome"]
            == "Fehler behoben"
        ]

        requests_count = (
            prior_requests
            .groupby("gateway_id")
            .size()
            .rename(
                "prior_visit_requests"
            )
        )

        completed_count = (
            prior_completed
            .groupby("gateway_id")
            .size()
            .rename(
                "prior_completed_visits"
            )
        )

        faults_count = (
            prior_faults
            .groupby("gateway_id")
            .size()
            .rename(
                "prior_confirmed_faults"
            )
        )

        history = pd.concat(
            [
                requests_count,
                completed_count,
                faults_count
            ],
            axis=1
        ).reset_index()

        history["week_start"] = prediction_week

        history_rows.append(
            history
        )

    if history_rows:

        history = pd.concat(
            history_rows,
            ignore_index=True
        )

        features = features.merge(
            history,
            on=[
                "gateway_id",
                "week_start"
            ],
            how="left"
        )

    else:

        features[
            "prior_visit_requests"
        ] = 0

        features[
            "prior_completed_visits"
        ] = 0

        features[
            "prior_confirmed_faults"
        ] = 0

    for c in [
        "prior_visit_requests",
        "prior_completed_visits",
        "prior_confirmed_faults"
    ]:

        features[c] = (
            features[c]
            .fillna(0)
        )

    return features


# ============================================================
# BUILD COMPLETE FEATURE DATASET
# ============================================================

def build_features(
    telemetry,
    meter,
    master,
    visits,
    prediction_weeks,
    numeric_columns
):

    features = build_decision_time_telemetry_features(
        telemetry,
        prediction_weeks,
        numeric_columns
    )

    features = add_meter_features(
        features,
        meter,
        prediction_weeks
    )

    features = add_master_features(
        features,
        master
    )

    features = add_visit_history(
        features,
        visits,
        prediction_weeks
    )

    # --------------------------------------------------------
    # Remove decommissioned gateway-weeks
    # --------------------------------------------------------

    before = len(features)

    if "is_decommissioned" in features.columns:

        features = features[
            features["is_decommissioned"] == 0
        ].copy()

    removed = before - len(features)

    print(
        "Removed decommissioned rows:",
        removed
    )

    # --------------------------------------------------------
    # Clean missing / infinite values
    # --------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.fillna(0)

    return features


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

def build_training_dataset(
    telemetry,
    meter,
    master,
    visits,
    numeric_columns
):

    print()
    print("=" * 70)
    print("Building historical training dataset...")
    print("=" * 70)

    historical_weeks = pd.date_range(
        "2025-09-01",
        "2026-01-26",
        freq="7D"
    )

    print(
        "Training weeks:",
        historical_weeks.min().date(),
        "to",
        historical_weeks.max().date()
    )

    features = build_features(
        telemetry,
        meter,
        master,
        visits,
        historical_weeks,
        numeric_columns
    )

    # --------------------------------------------------------
    # Exact target logic
    # --------------------------------------------------------

    visits_target = visits.copy()

    visits_target["gateway_id"] = normalize_gateway_id(
        visits_target["gateway_id"]
    )

    visits_target["requested_on"] = pd.to_datetime(
        visits_target["requested_on"],
        errors="coerce"
    )

    visits_target["target"] = (
        visits_target["outcome"]
        == "Fehler behoben"
    ).astype(int)

    visits_target["visit_week"] = (
        visits_target["requested_on"]
        -
        pd.to_timedelta(
            visits_target["requested_on"].dt.weekday,
            unit="D"
        )
    ).dt.normalize()

    labels = (
        visits_target
        .groupby(
            [
                "visit_week",
                "gateway_id"
            ]
        )["target"]
        .max()
        .reset_index()
    )

    labels = labels.rename(
        columns={
            "visit_week": "week_start"
        }
    )

    features["week_start"] = pd.to_datetime(
        features["week_start"]
    ).dt.tz_localize(None)

    labels["week_start"] = pd.to_datetime(
        labels["week_start"]
    ).dt.tz_localize(None)

    features = features.merge(
        labels,
        on=[
            "gateway_id",
            "week_start"
        ],
        how="left"
    )

    features["target"] = (
        features["target"]
        .fillna(0)
        .astype(int)
    )

    print(
        "Training rows:",
        len(features)
    )

    print(
        "Positive targets:",
        int(features["target"].sum())
    )

    print(
        "Negative targets:",
        int(
            (features["target"] == 0).sum()
        )
    )

    return features


# ============================================================
# BUILD PREDICTION DATASET
# ============================================================

def build_prediction_dataset(
    telemetry,
    meter,
    master,
    visits,
    numeric_columns
):

    print()
    print("=" * 70)
    print("Building prediction dataset...")
    print("=" * 70)

    features = build_features(
        telemetry,
        meter,
        master,
        visits,
        OFFICIAL_WEEKS,
        numeric_columns
    )

    return features


# ============================================================
# ALIGN TO VERIFIED 201-FEATURE SCHEMA
# ============================================================

def align_prediction_features(
    training,
    prediction
):

    print()
    print("=" * 70)
    print("ALIGNING VERIFIED 201-FEATURE SCHEMA")
    print("=" * 70)

    # --------------------------------------------------------
    # 48 historical-only aggregate features
    #
    # These were removed when we verified the historical
    # dataset against the exact reference feature schema.
    # --------------------------------------------------------

    excluded_features = {

        # Hour aggregates
        "hour_mean",
        "hour_sum",
        "hour_max",
        "hour_min",

        # operator_3AT
        "operator_3AT_mean",
        "operator_3AT_sum",
        "operator_3AT_max",
        "operator_3AT_min",

        # operator_A1
        "operator_A1_mean",
        "operator_A1_sum",
        "operator_A1_max",
        "operator_A1_min",

        # operator_Eplus
        "operator_Eplus_mean",
        "operator_Eplus_sum",
        "operator_Eplus_max",
        "operator_Eplus_min",

        # operator_O2DE
        "operator_O2DE_mean",
        "operator_O2DE_sum",
        "operator_O2DE_max",
        "operator_O2DE_min",

        # operator_OrangeLU
        "operator_OrangeLU_mean",
        "operator_OrangeLU_sum",
        "operator_OrangeLU_max",
        "operator_OrangeLU_min",

        # operator_Salt
        "operator_Salt_mean",
        "operator_Salt_sum",
        "operator_Salt_max",
        "operator_Salt_min",

        # operator_Swisscom
        "operator_Swisscom_mean",
        "operator_Swisscom_sum",
        "operator_Swisscom_max",
        "operator_Swisscom_min",

        # operator_TelekomDE
        "operator_TelekomDE_mean",
        "operator_TelekomDE_sum",
        "operator_TelekomDE_max",
        "operator_TelekomDE_min",

        # operator_TmobileA
        "operator_TmobileA_mean",
        "operator_TmobileA_sum",
        "operator_TmobileA_max",
        "operator_TmobileA_min",

        # operator_VodafoneDE
        "operator_VodafoneDE_mean",
        "operator_VodafoneDE_sum",
        "operator_VodafoneDE_max",
        "operator_VodafoneDE_min",

        # operator_unknown
        "operator_unknown_mean",
        "operator_unknown_sum",
        "operator_unknown_max",
        "operator_unknown_min",
    }

    # --------------------------------------------------------
    # Verify that all 48 expected columns exist
    # --------------------------------------------------------

    training_present = (
        excluded_features
        .intersection(training.columns)
    )

    prediction_present = (
        excluded_features
        .intersection(prediction.columns)
    )

    print(
        "Excluded features found in training:",
        len(training_present)
    )

    print(
        "Excluded features found in prediction:",
        len(prediction_present)
    )

    # --------------------------------------------------------
    # Remove them
    # --------------------------------------------------------

    training = training.drop(
        columns=[
            c
            for c in excluded_features
            if c in training.columns
        ]
    )

    prediction = prediction.drop(
        columns=[
            c
            for c in excluded_features
            if c in prediction.columns
        ]
    )

    # --------------------------------------------------------
    # Final feature list
    # --------------------------------------------------------

    feature_columns = [
        c
        for c in training.columns
        if c not in {
            "gateway_id",
            "week_start",
            "target"
        }
    ]

    # --------------------------------------------------------
    # Make prediction columns identical
    # --------------------------------------------------------

    for col in feature_columns:

        if col not in prediction.columns:

            prediction[col] = 0

    prediction = prediction[
        [
            "week_start",
            "gateway_id"
        ]
        + feature_columns
    ]

    training = training[
        [
            "week_start",
            "gateway_id"
        ]
        + feature_columns
        + ["target"]
    ]

    # --------------------------------------------------------
    # Exact verification
    # --------------------------------------------------------

    print(
        "Final training feature count:",
        len(feature_columns)
    )

    print(
        "Final prediction feature count:",
        len(feature_columns)
    )

    if len(feature_columns) != 201:

        raise RuntimeError(
            "ERROR: Expected exactly 201 ML features, "
            f"but found {len(feature_columns)}."
        )

    if list(
        training.columns[
            2:-1
        ]
    ) != list(
        prediction.columns[
            2:
        ]
    ):

        raise RuntimeError(
            "ERROR: Training and prediction feature "
            "schemas do not match."
        )

    print(
        "Feature schema verification: PASSED"
    )

    return training, prediction


# ============================================================
# TRAIN MODEL AND PREDICT
# ============================================================

def train_and_predict(
    training,
    prediction
):

    print()
    print("=" * 70)
    print("TRAINING FINAL LOGISTIC REGRESSION")
    print("=" * 70)

    X_train = training.drop(
        columns=[
            "target",
            "week_start"
        ]
    )

    y_train = training["target"]

    X_prediction = prediction.drop(
        columns=[
            "week_start"
        ]
    )

    numeric_features = (
        X_train
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_features = [
        c
        for c in X_train.columns
        if c not in numeric_features
    ]

    print(
        "Features used:",
        len(X_train.columns)
    )

    print(
        "Numeric features:",
        len(numeric_features)
    )

    print(
        "Categorical features:",
        len(categorical_features)
    )

    # --------------------------------------------------------
    # Numeric pipeline
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=0.5,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model trained successfully."
    )

    # --------------------------------------------------------
    # Prediction score
    # --------------------------------------------------------

    prediction["score"] = (
        model.predict_proba(
            X_prediction
        )[:, 1]
    )

    return prediction


# ============================================================
# CREATE TOP-15 WEEKLY RANKINGS
# ============================================================

def create_rankings(
    prediction
):

    print()
    print("=" * 70)
    print("CREATING WEEKLY TOP-15 RANKINGS")
    print("=" * 70)

    results = []

    for week in OFFICIAL_WEEKS:

        week_data = prediction[
            prediction["week_start"]
            == week
        ].copy()

        print(
            f"{week.date()}: "
            f"{len(week_data)} available gateways"
        )

        # Deterministic ranking:
        #
        # 1. Highest ML risk first
        # 2. gateway_id ascending for exact ties
        #

        week_data = week_data.sort_values(
            by=[
                "score",
                "gateway_id"
            ],
            ascending=[
                False,
                True
            ]
        ).head(15)

        week_data["rank"] = range(
            1,
            len(week_data) + 1
        )

        week_data["reason"] = (
            "High predicted fault risk based on "
            "recent telemetry, connectivity and "
            "meter-read signals."
        )

        results.append(
            week_data[
                [
                    "week_start",
                    "rank",
                    "gateway_id",
                    "score",
                    "reason"
                ]
            ]
        )

    predictions = pd.concat(
        results,
        ignore_index=True
    )

    return predictions


# ============================================================
# FINAL OUTPUT VALIDATION
# ============================================================

def validate_output(
    predictions
):

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        "Rows:",
        len(predictions)
    )

    rows_per_week = (
        predictions
        .groupby("week_start")
        .size()
    )

    print()
    print("Rows per week:")
    print(
        rows_per_week
    )

    expected_columns = [
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason"
    ]

    if list(
        predictions.columns
    ) != expected_columns:

        raise ValueError(
            "Incorrect prediction columns."
        )

    if len(predictions) != 120:

        raise ValueError(
            "Expected exactly 120 rows, "
            f"got {len(predictions)}."
        )

    if not all(
        rows_per_week == 15
    ):

        raise ValueError(
            "Every week must contain exactly "
            "15 gateways."
        )

    # --------------------------------------------------------
    # Check every week's ranks
    # --------------------------------------------------------

    for week in OFFICIAL_WEEKS:

        week_data = predictions[
            predictions["week_start"]
            == week
        ]

        expected_ranks = list(
            range(1, 16)
        )

        if (
            week_data["rank"].tolist()
            != expected_ranks
        ):

            raise ValueError(
                f"Incorrect ranks for "
                f"{week.date()}."
            )

        # No duplicate gateway in a week
        if (
            week_data["gateway_id"]
            .nunique()
            != 15
        ):

            raise ValueError(
                f"Duplicate gateway found "
                f"for {week.date()}."
            )

    # --------------------------------------------------------
    # Check scores
    # --------------------------------------------------------

    if predictions["score"].isna().any():

        raise ValueError(
            "Prediction score contains NaN."
        )

    if (
        (predictions["score"] < 0)
        |
        (predictions["score"] > 1)
    ).any():

        raise ValueError(
            "Prediction score outside [0,1]."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    predictions.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Validation passed."
    )

    print()
    print("=" * 70)
    print(
        "SUCCESS: predictions.csv created."
    )
    print("=" * 70)

    print()
    print(
        "Output file:",
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEXORA 2026 - FINAL ML SOLUTION")
    print("=" * 70)

    print()
    print("Prediction period:")

    print(
        OFFICIAL_WEEKS.min().date(),
        "to",
        OFFICIAL_WEEKS.max().date()
    )

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    telemetry = load_telemetry()

    numeric_columns = (
        get_numeric_telemetry_columns(
            telemetry
        )
    )

    print()
    print(
        "Numeric telemetry columns:",
        len(numeric_columns)
    )

    if len(numeric_columns) != 54:

        raise RuntimeError(
            "Expected 54 numeric telemetry columns, "
            f"found {len(numeric_columns)}."
        )

    # --------------------------------------------------------
    # Load remaining data
    # --------------------------------------------------------

    meter = pd.read_csv(
        METER_FILE
    )

    master = pd.read_csv(
        MASTER_FILE,
        encoding="latin1"
    )

    visits = pd.read_csv(
        FIELD_VISITS_FILE,
        encoding="latin1"
    )

    # --------------------------------------------------------
    # Build training dataset
    # --------------------------------------------------------

    training = build_training_dataset(
        telemetry,
        meter,
        master,
        visits,
        numeric_columns
    )

    # --------------------------------------------------------
    # Build prediction dataset
    # --------------------------------------------------------

    prediction = build_prediction_dataset(
        telemetry,
        meter,
        master,
        visits,
        numeric_columns
    )

    # --------------------------------------------------------
    # Align verified schema
    # --------------------------------------------------------

    training, prediction = (
        align_prediction_features(
            training,
            prediction
        )
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    prediction = train_and_predict(
        training,
        prediction
    )

    # --------------------------------------------------------
    # Create rankings
    # --------------------------------------------------------

    predictions = create_rankings(
        prediction
    )

    # --------------------------------------------------------
    # Validate and save
    # --------------------------------------------------------

    validate_output(
        predictions
    )

    # --------------------------------------------------------
    # Show first week
    # --------------------------------------------------------

    print()
    print("First 15 predictions:")

    print(
        predictions
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()