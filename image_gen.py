from g4f.client import Client
import requests
from pathlib import Path
from datetime import datetime

# Create images directory if it doesn't exist
output_dir = Path("reel_output/images")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate image
client = Client()
response = client.images.generate(
    model="sd-3.5-large",
    prompt="trump and joe biden playing chess",
    response_format="url"
)

image_url = response.data[0].url
print(f"Generated image URL: {image_url}")

# Download and save the image
try:
    img_response = requests.get(image_url, timeout=30)
    img_response.raise_for_status()
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.png"
    filepath = output_dir / filename
    
    # Save the image
    with open(filepath, 'wb') as f:
        f.write(img_response.content)
    
    print(f"Image saved successfully to: {filepath}")
except Exception as e:
    print(f"Error saving image: {e}")