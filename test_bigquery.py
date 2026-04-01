import os
from google.cloud import bigquery

project_id = "metre-489201"
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

client = bigquery.Client(project=project_id)

query = """
SELECT *
FROM `physionet-data.mimic_icu.icustays`
LIMIT 10
"""

df = client.query(query).to_dataframe()
print(df.head())