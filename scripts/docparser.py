import re
import json
import os

import pandas as pd

# load parquet files
fp = 'patient_reports.parquet'
df=pd.read_parquet(fp)
#contents=df.content.to_list()
#print(df.columns)
#print(len(contents))

# Convert to mapping dict
HBN_Base = dict(zip(df['id'], df['content']))
print(len(HBN_Base))

#print(contents[1][:1500])


# Actual Parsing code

def HBN_ReportBreak(text, level):

    match level:
        case 1:
            pattern = r'\*\*((?:[A-Z]+(?: [A-Z]+)*))\*\*' # Match: ** ALL_CAPS **
            cleanup=1 # excludes ** **
        case 2:
            pattern = r'\*\*([^\*]+)\*\*'  #  Match: ** Upper or Lower **
            cleanup=1 # excludes ** **
        case 3:
            pattern = r'# (?:[A-Z]+(?:\s+[A-Z]+)*)(?=[^A-Za-z]|$)' # Match: # ... ALL_CAPS
            cleanup=0 # includes **section header**

    result = {}
    if not re.search(pattern, text):
        stripped = text.strip()
        if stripped:
            result['FULL_TEXT'] = stripped
        return result

    matches = list(re.finditer(pattern, text))
    n = len(matches)

    # Capture intro 
    first = matches[0]
    intro_text = text[:first.start()].strip()
    if intro_text:
        result['INTRO'] = intro_text

    # Headers and sub sections
    for i, m in enumerate(matches):
        title = m.group(cleanup).strip()
        title = title.replace('#', '').strip()
        section_start = m.end()

        if i + 1 < n:
            section_end =  matches[i +1].start()
        else:
            section_end = len(text)

        section_text = text[section_start:section_end].strip()
        result[title] = section_text

    # Trailing Text
    last_end = section_end
    if last_end < len(text):
        trail_text = text[last_end:]
        if trail_text:
            result['TRAIL'] = trail_text

    return result



# Iterate through reports using parsing functions
HBN=[]
for xid,text in HBN_Base.items():
    document={}
    HBN_LO = HBN_ReportBreak(text, 1)
    
    HBN_L1=dict()
    for k,v in HBN_LO.items():
        SubSections=HBN_ReportBreak(v, 2)
        HBN_L1[k]=SubSections
        
    HBN_Report=dict()
    for k,l1 in HBN_L1.items(): # First Pass
        L2 = {}
        for k1, l2 in l1.items():  # Second Pass
            Resp=HBN_ReportBreak(l2, 3)
            L2[k1] = Resp
        HBN_Report[k]=L2
        
    document[xid] = HBN_Report
    HBN.append(document)



# Split into sections for easier comparison across like groups
INTRO,IDENTIFYING_INFORMATION,MEDICAL_HISTORY,RECOMMENDATIONS ={},{},{},{}

for document in HBN:
    for k,v in document.items():
        INTRO[k] = v.get('INTRO')
        IDENTIFYING_INFORMATION[k] = v.get('IDENTIFYING INFORMATION')
        MEDICAL_HISTORY[k] = v.get('MEDICAL HISTORY')
        RECOMMENDATIONS[k] = v.get('RECOMMENDATIONS')
        

print(list(MEDICAL_HISTORY.items())[0])

# Open the file in write mode and use json.dump() to save the list
with open('HBN_Report_Samples', 'w') as f:
    json.dump(HBN, f, indent=4) # indent=4 for pretty-printing, optional