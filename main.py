import cv2
import numpy as np
from skimage import feature, io, transform
from skimage.color import rgb2gray
from skimage.feature import canny
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_image(image_path, output_size=(256, 256)):
    """Preprocess the image."""
    image = io.imread(image_path)
    image = transform.resize(image, output_size)
    return image

def extract_color_histogram(image):
    """Extract color histogram."""
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 1))
    return hist / sum(hist)

def extract_texture_features(image):
    """Extract texture features using Local Binary Patterns."""
    # Convert the image to grayscale
    image_gray = rgb2gray(image)
    # Compute Local Binary Pattern (LBP)
    lbp = feature.local_binary_pattern(image_gray, P=8, R=1, method='uniform')
    # Calculate histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10), density=True)  # Normalize histogram
    return hist

def extract_shape_features(image):
    """Extract shape features using edge detection."""
    # Convert the image to grayscale
    image_gray = rgb2gray(image)
    # Compute edges using Canny edge detection
    edges = canny(image_gray)
    # Calculate the sum of edge pixels
    return np.sum(edges)

def detect_pus(image):
    """Detect pus by looking for rapid changes in RGB values."""
    # Calculate the mean of RGB values in each region
    mean_rgb = np.mean(image, axis=(0, 1))
    # Calculate the standard deviation of RGB values in each region
    std_rgb = np.std(image, axis=(0, 1))
    # Check for rapid changes in RGB values indicating the presence of pus
    pus_detection = np.any(std_rgb > 50)  # Adjust the threshold as needed
    return pus_detection

def compare_images(target_image, images, image_names):
    """Compare the target image with all other images."""
    similarities = []
    for i, image in enumerate(images):
        color_similarity, texture_similarity, shape_similarity = compare_features(target_image, image)
        similarity_percentage = (color_similarity + texture_similarity + shape_similarity) / 3 * 100
        pus_detection = detect_pus(image)
        similarities.append((image_names[i], similarity_percentage, pus_detection))

    # Rank the similarity of the images
    similarities.sort(key=lambda x: (x[1], ~x[2]), reverse=True)
    return similarities[:3]

def compare_features(image1, image2):
    """Compare two images based on color, texture, and shape."""
    # Color comparison
    color_hist1 = extract_color_histogram(image1)
    color_hist2 = extract_color_histogram(image2)
    color_similarity = cosine_similarity([color_hist1], [color_hist2])[0][0]

    # Texture comparison
    texture_feat1 = extract_texture_features(image1)
    texture_feat2 = extract_texture_features(image2)
    texture_similarity = cosine_similarity([texture_feat1], [texture_feat2])[0][0]

    # Shape comparison
    shape_feat1 = extract_shape_features(image1)
    shape_feat2 = extract_shape_features(image2)
    shape_similarity = 1 - abs(shape_feat1 - shape_feat2) / max(shape_feat1, shape_feat2)

    return color_similarity, texture_similarity, shape_similarity

def get_skin_disease_description(image_name):
    """Get description of the skin disease based on image name."""
    disease_descriptions = {
        'Eczema': 'Eczema - Eczema is a condition where patches of skin become inflamed, itchy, red, cracked, and rough. The most important treatment of eczema is skin hydration and steroids for frequent flare-ups.',
        'Psoriasis': 'Psoriasis - Psoriasis is a chronic skin condition characterized by patches of abnormal skin. There are a multitude of psoriasis medications including  infliximab, bimekizumab, ixekizumab, and risankizumab',
        'Acne' : 'Acne - Acne is a skin condition that occurs when hair follicles become plugged with oil and dead skin cells. The best treatment that treats multiple cases of acne is Antibiotics',
        'Rosacea': 'Rosacea - Rosacea is a common skin condition that causes redness and visible blood vessels in your face.',
        'Ringworm': 'Ringworm - Ringworm is a fungal infection of the skin that causes a red, circular, itchy rash.',
        'Hives': 'Hives - Hives, also known as urticaria, are raised, itchy welts that appear on the surface of the skin. the best treatment is a nonprescription oral antihistamine,',
        'Impetigo': 'Impetigo - Impetigo is a highly contagious bacterial skin infection characterized by red sores that rupture and form a yellow crust.',
        'Cold Sore': 'Cold Sores - Cold sores are small, painful, fluid-filled blisters that occur on or around the lips and mouth. A prescription of antiviral drugs and cold compressing the area affected is the best solution',
        'Cellulitis': 'Cellulitis - Cellulitis is a common bacterial skin infection characterized by redness, swelling, and warmth of the affected area.',
        'Poison Ivy' : 'Poison Ivy - Poison Ivy is a bacterial infection that affects the skin and can cause redness, itching, and swelling. Over the counter creams are usually the solution for poison Ivy',
        'Boil' : 'Boil - Boil is a skin condition characterized by redness, itching, and swelling of the affected area. Resist the temptation to squeeze the boil. Wash the boil with antiseptic soap.',
        'Leprosy' : 'Leprosy - Leprosy is a skin condition characterized by redness, itching, and swelling of the affected area. the best treatment is a nonprescription oral antihistamine',
        'Melanoma' : 'Melanoma - Melanoma is a skin cancer that affects the skin and can cause redness, itching, and swelling. The best treatment is a nonprescription oral antihistamine',
    'Warts' : 'Warts - Warts are small, painful, fluid-filled blisters that occur on or around the lips and mouth. A prescription of antiviral drugs and cold compressing the area affected'
    
    }
    return disease_descriptions.get(image_name, 'Unknown disease description.')

def main():
    # Load and preprocess the images
    image_paths = ['Acne.jpeg', 'Boil.jpeg', 'Cellulitis.jpeg', 'Cold_Sore.jpeg', 'Hives.jpeg', 
                   'Leprosy.jpeg', 'Melanoma.jpeg', 'Poison_Ivy.jpeg', 'Warts.jpeg', 'Melanoma.jpeg']
    image_names = ['Acne', 'Boil', 'Cellulitis', 'Cold Sore', 'Hives', 
                   'Leprosy', 'Melanoma', 'Poison Ivy', 'Warts', 'Acne']
    images = [preprocess_image(path) for path in image_paths]

    # Target image is the last image in the list
    target_image = images[-1]
    other_images = images[:-1]

    # Compare the target image with all other images
    similarities = compare_images(target_image, other_images, image_names)

    # Print the top 3 most similar images
    print("Top 3 most similar images:")
    for i, (name, similarity_percentage, pus_detected) in enumerate(similarities):
        print(f"Rank {i+1}: {name} - Similarity: {similarity_percentage:.2f}%, Pus Detected: {pus_detected}")

    # Print the description of the skin disease for the most similar images
    print("\nDescription of the skin disease:")
    for name, _, _ in similarities:
        print(f"\n{name}: {get_skin_disease_description(name)}")

if __name__ == "__main__":
    main()
