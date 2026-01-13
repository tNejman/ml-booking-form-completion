import pandas as pd
import re
import json

def extract_amenities_from_description(df):
    regex_patterns = {}
    with open("amenity_patterns.json") as file:
        regex_patterns = json.load(file)
        
    def normalize_amenities(val):
        if pd.isna(val) or val == "":
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return [item.strip() for item in val.split(',')]
        return []

    current_amenities_col = df['amenities'].apply(normalize_amenities)

    updated_amenities = []

    for amenities_list, description in zip(current_amenities_col, df['description']):
        amenities_set = set(amenities_list)
        
        if pd.notna(description) and isinstance(description, str):
            for amenity_name, pattern in regex_patterns.items():
                if re.search(pattern, description):
                    amenities_set.add(amenity_name)
        
        updated_amenities.append(list(amenities_set))

    df['amenities'] = updated_amenities
    
    return df
print(extract_amenities_from_description(pd.DataFrame({"description":"wifi blalakitchen"})))