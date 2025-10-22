import json
import pandas as pd

# Read the JSON file
with open('task_generator/data/BuildingsDescriptionMapTemplate.json', 'r') as file:
    data = json.load(file)

# Create a list to store all building data
buildings_data = []

# Extract data for each building
for building_id, attributes in data.items():
    building_info = {
        'Building_ID': building_id,
        'Color': attributes['color'],
        'Height': attributes['height'],
        'Material': attributes['material'],
        'Type': attributes['type'],
        'Extra_Features': ', '.join(attributes['extra_features'])
    }
    buildings_data.append(building_info)

# Create DataFrame
df = pd.DataFrame(buildings_data)

# Save to Excel
df.to_excel('building_attributes.xlsx', index=False)

print("Data has been successfully exported to building_attributes.xlsx") 