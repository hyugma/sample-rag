{{ config(materialized='view', alias='raw_documents_view') }}
select id, text
from raw_documents
