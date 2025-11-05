from section_loader import MarkdownSectionLoader

import os
import json
import pandas as pd
import spacy

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpEngine, SpacyNlpEngine, NerModelConfiguration
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# Starter Function to sync data loader:
def LoadFiles(source_data):
    loader = MarkdownSectionLoader(source_data) 
    df_rfv = loader.filter_sections("REASON FOR VISIT")
    df_haf = loader.filter_sections("Home and Adaptive Functioning")
    df_all = pd.concat([df_rfv, df_haf])
    return df_all


# Some Config placeholders to get started
# todo - add args

config1 = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
        #{"lang_code": "en", "model_name": "es_core_news_md"},
    ],
}
config2 = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "es_core_news_md"},
    ],
}

configs = [config1, config2]

"""
NER PII Iterator
    Returns anonymized text using presidio and config 
    Returns dict mapping original source idx location of text replaced
"""
class ProcessNER:

    def __init__(self, config):
        self.config = config
        self.provider = NlpEngineProvider(
            nlp_configuration=config
        )
        self.analyzer = AnalyzerEngine(
            nlp_engine=self.provider.create_engine()
        )
        self.anonymizer = AnonymizerEngine()

    def ScanText(self, text):
        # Run Presidio Analyzer
        analyzer_results = self.analyzer.analyze(text=text, language="en")
        # Build ORIGINAL SOURCE reference dictionary with index_start:(index_end, entity_type) for all identified results
        Rdict={res.start:(res.end,res.entity_type, res.score) for res in analyzer_results}

        anonymized_results = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"}),
            }
        )
        return anonymized_results, Rdict





# Basic function to kick off processing loop
# Functional Placeholder - will need to be updated for production

def RunProcess(config):

    NER=ProcessNER(config)

    # Iterate Through All Docs
    # Create dicts for GUID:Source and GUID:Anonymized (with details)
    Master_Ref={}
    Results={}
    for item in list(zip(df_all['GUID'], df_all['content'])):
        Master_Ref[item[0]]=item[1]
        anonymized_results, rdict = NER.ScanText(item[1])
        Results[item[0]]=(anonymized_results.text, anonymized_results.items, rdict)

    #For simplicity sake for now we create dataframes and merge - 
    Master_Ref = pd.DataFrame.from_dict(Master_Ref, orient='index', columns=["Source_Content"])
    Results = pd.DataFrame.from_dict(Results, orient='index', columns=["Anonymized_Text","Anonymized_Items","Source_Item_Idx"])
    Full=Master_Ref.join(Results)
    Full.to_csv("output/Merged.csv")



if __name__ == "__main__":
    source_data='./output/parsed_reports_extracted_20251021_parsed_20251104_MRNRENAME_remapped.parquet'
    df_all = LoadFiles(source_data)
    RunProcess(configs[1])