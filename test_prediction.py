import requests
import os

# Test with multiple sample images
sample_images = [f for f in os.listdir('mri_sample_images') if f.endswith(('.jpg', '.png'))][:3]

print("=" * 60)
print("MRI BRAIN TUMOR DETECTION - API TEST")
print("=" * 60)

for img_name in sample_images:
    img_path = os.path.join('mri_sample_images', img_name)
    print(f'\n>>> Testing: {img_name}')
    print("-" * 40)
    
    with open(img_path, 'rb') as f:
        response = requests.post('http://127.0.0.1:5000/predict', files={'file': f})
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Has Tumor: {result['has_tumor']}")
        print(f"All Probabilities:")
        for label, prob in result['all_probabilities'].items():
            print(f"  - {label}: {prob}%")
    else:
        print(f'Error: {response.status_code}')
        print(response.text)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
