from pathlib import Path
import re

from llm_utils.customized_openai_model import CustomizedOpenAIModel

def generate_description(self, templates) -> str:
        """
        Generate a natural language description from templates.
        
        Args:
            templates: List of template strings
            
        Returns:
            Natural language description
        """
        if not templates:
            return ""
            
        model = CustomizedOpenAIModel("gpt-4o")
        response = model.generate(
                [
                {
                "role": "system",
                "content": """You are an assistant that analyzes building images and extracts concise, structured descriptions for use in robot navigation tasks.

                The robot has a limited field of view, approximately 70 cm (27.5 inches) in height, similar to that of a small child. Therefore, you should focus only on visual features that are clearly visible from the ground level. Avoid describing rooftops or any details that are far above the robot’s typical eye level.

                Please output a structured summary in the following JSON format:
                {
                    "color": "<Main color of the building, e.g., red, white, blue, yellow, green, brown, gray, black, mixed>",
                    "height": "<Height description of the building, e.g., one-story, two-story, three-story, tall, medium-height, short, skyscraper>",
                    "material": "<Main exterior material, e.g., brick, glass, concrete, wood, stone, metal, mixed>",
                    "type": "<Type of building, e.g., apartment, house, shop, office building, hospital, restaurant, cafe, gas station, landmark>",
                    "extra_features": [
                        "<Distinctive features at low or mid-level that would help a robot recognize the building, such as: clock tower, fountain, balcony, ivy-covered, large windows, pointed roof, signage, arches, pillars, wall textures, number of entrances, awnings>"
                    ]
                }
                Be brief but specific. Avoid uncertain or speculative language. If any attribute is unclear or not visible, use "unknown"""
                }
                ,
                {"role": "user", "content": f"""Convert the following structured navigation template into a natural language instruction: {templates}"""}
                ]
        )
        return response


def traverse_buildings():
    current_dir = Path(__file__).parent
    buildings_dir = current_dir / "Buildings"
    
    if not buildings_dir.exists():
        print(f"Error: Buildings directory not found at {buildings_dir}")
        return
    
    pattern = re.compile(r'(building|Building)(\d+)', re.IGNORECASE)
    
    for building_folder in sorted(buildings_dir.iterdir()):
        if building_folder.is_dir():
            match = pattern.match(building_folder.name)
            if match:
                building_number = match.group(2)
                print(f"\nProcessing building number: {building_number}")
                
                for image_file in building_folder.glob("*.png"):
                    print(f"  Found image: {image_file.name}")
            else:
                print(f"\nWarning: Could not extract building number from {building_folder.name}")




if __name__ == "__main__":
    traverse_buildings()
