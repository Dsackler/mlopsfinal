# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String

# Define a project for the feature repo
# Match the project name to `feature_store.yaml` so Feast registers features
project = Project(name="final_feature_repo", description="A project for car statistics")

# Define an entity for the athlete. You can think of an entity as a primary key used to
# fetch features.
# Define an entity
car = Entity(name="car_id", join_keys=["car_id"])

car_stats_source = FileSource(
    name="car_stats_source",
    path="data/car_data.parquet",
    timestamp_field="event_timestamp",
)

# Define feature schema
car_features = FeatureView(
    name="car_stats",
    entities=[car],
    ttl=None,
    schema=[
        Field(name="car_id", dtype=Int64),
        Field(name="brand", dtype=String),
        Field(name="year", dtype=Int64),
        Field(name="engine_size", dtype=Float64),
        Field(name="fuel_type", dtype=String),
        Field(name="transmission", dtype=String),
        Field(name="mileage", dtype=Int64),
        Field(name="condition", dtype=String),
        Field(name="price", dtype=Float64),
        Field(name="model", dtype=String),
    ],
    online=True,
    source=car_stats_source
)

# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
# @on_demand_feature_view(
#     sources=[athlete_features],
#     schema=[
#         Field(name="total_lift", dtype=Float64),
#         # Field(name="val_to_add_temp", dtype=Float64),
#     ],
# )
# def transformed_lift(inputs: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame()
#     df["total_lift"] = inputs["candj"] + inputs["snatch"] + inputs["deadlift"] + inputs["backsq"]
#     return df

# # This groups features into a model version
# athlete_activity_v1 = FeatureService(
#     name="athlete_activity_v1",
#     features=[
#         athlete_features[["backsq", "deadlift", "candj", "snatch"]],  # Sub-selects a feature from a feature view
#         transformed_lift,  # Selects all features from the feature view
#     ],
#     logging_config=LoggingConfig(
#         destination=FileLoggingDestination(path="data")
#     ),
# )

# athlete_activity_v2 = FeatureService(
#     name="athlete_activity_v2", features=[athlete_features, transformed_lift]
# )

car_activity = FeatureService(
    name="car_activity", features=[car_features]
)

# Defines a way to push data (to be available offline, online or both) into Feast.
# athlete_stats_push_source = PushSource(
#     name="athlete_stats_push_source",
#     # batch_source=athlete_stats_source,
#     batch_source=None,
# )

# # Defines a slightly modified version of the feature view from above, where the source
# # has been changed to the push source. This allows fresh features to be directly pushed
# # to the online store for this feature view.
# athlete_stats_fresh_fv = FeatureView(
#     name="athlete_stats_fresh",
#     entities=[athlete],
#     ttl=None,
#     schema=[
#         Field(name="athlete_id", dtype=Float32),
#         Field(name="age", dtype=Float32),
#         Field(name="height", dtype=Float32),
#         Field(name="weight", dtype=Float32),
#         Field(name="candj", dtype=Float32),
#         Field(name="snatch", dtype=Float32),
#         Field(name="deadlift", dtype=Float32),
#         Field(name="backsq", dtype=Float32),
#         Field(name="gender_Female", dtype=Float32),
#         Field(name="gender_Male", dtype=Float32),
#     ],
#     online=True,
#     source=athlete_stats_push_source,  # Changed from above
#     tags={},
# )


# # Define an on demand feature view which can generate new features based on
# # existing feature views and RequestSource features
# @on_demand_feature_view(
#     sources=[athlete_stats_fresh_fv, input_request],  # relies on fresh version of FV
#     schema=[
#         Field(name="backsq_plus_val1", dtype=Float64),
#         Field(name="deadlift_plus_val2", dtype=Float64),
#     ],
# )
# def transformed_lift_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame()
#     df["backsq_plus_val1"] = inputs["backsq"] + inputs["val_to_add"]
#     df["deadlift_plus_val2"] = inputs["deadlift"] + inputs["val_to_add_2"]
#     return df


# athlete_activity_v3 = FeatureService(
#     name="athlete_activity_v3",
#     features=[athlete_stats_fresh_fv, transformed_lift_fresh],
# )