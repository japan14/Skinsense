import streamlit as st 
from PIL import Image
import numpy as np
from google.cloud import storage
from skimage import feature, io, transform
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, canny
from sklearn.metrics.pairwise import cosine_similarity

client = storage.Client()

def load_image_from_storage(image_path):
  try:
      bucket_name = "skin-images-bucket"
      bucket = client.get_bucket(bucket_name)
      blob = bucket.blob(image_path)
      blob.download_to_filename("temp_image.jpg")
      img = Image.open("temp_image.jpg")
      return img
  except Exception as e:
    st.error(f"Error downloading image: {e}")
    return None


def preprocess_image(image, output_size=(256, 256)):
    """Preprocess the image."""
    image = io.imread(image)
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

def compare_images(uploaded_image, stored_images):
  similarities = []
  try:
    for image in stored_images:
        color_similarity, texture_similarity, shape_similarity = compare_features(uploaded_image, image)
        similarity_percentage = (color_similarity + texture_similarity + shape_similarity) / 3 * 100
        pus_detection = detect_pus(image)
        similarities.append((image_names[i], similarity_percentage, pus_detection))
    
    similarities.sort(key=lambda x: (x[1], ~x[2]), reverse=True)
    return similarities[:3]
  
  except Exception as e:
    st.error(f"Error comparing images: {e}")
    return None
  
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
        'Eczema': 'Eczema - Eczema is a condition where patches of skin become inflamed, itchy, red, cracked, and rough.',
        'Psoriasis': 'Psoriasis - Psoriasis is a chronic skin condition characterized by patches of abnormal skin.',
        'Acne' : 'Acne - Acne is a skin condition that occurs when hair follicles become plugged with oil and dead skin cells.',
        'Rosacea': 'Rosacea - Rosacea is a common skin condition that causes redness and visible blood vessels in your face.',
        'Ringworm': 'Ringworm - Ringworm is a fungal infection of the skin that causes a red, circular, itchy rash.',
        'Hives': 'Hives - Hives, also known as urticaria, are raised, itchy welts that appear on the surface of the skin.',
        'Impetigo': 'Impetigo - Impetigo is a highly contagious bacterial skin infection characterized by red sores that rupture and form a yellow crust.',
        'Cold Sore': 'Cold Sores - Cold sores are small, painful, fluid-filled blisters that occur on or around the lips and mouth.',
        'Cellulitis': 'Cellulitis - Cellulitis is a common bacterial skin infection characterized by redness, swelling, and warmth of the affected area.',
    }
    
def main():
  st.title("Skin Condition Image Comparison")
  st.header("Upload Image")
  uploaded_file = st.file_uploader("Choose an image below:", type=["jpg","jpeg", "png"])

  if uploaded_file is None:
    st.warning("Please upload an image!")
  else:
    try:
      uploaded_image = Image.open(uploaded_file)
      st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
      st.error(f"Error opening uploaded image: {e}")
      return

    stored_image_names = ["07Acne081101.jpg", "07AcnePittedScars.jpg","Urticaria-Vasculitis-15.jpg","atypical-nevi-17.jpg","biting-insects-5.jpg","chigger-bites-bullous-2.jpg","eczema-asteatotic-12.jpg","eczema-fingertips-59.jpg","hives-Urticaria-Acute-7.jpg","malignant-melanoma-98.jpg","rhus-dermatitis-13.jpg","rhus-dermatitis-60.jpg"]
    stored_images = [load_image_from_storage(image_name) for image_name in stored_image_names]

    similarities = [compare_images(Image.open(uploaded_file), stored_image) for stored_image in stored_images]
    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[max_similarity_index]

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
        print(f"{name}: {get_skin_disease_description(name)}")

  #images = [preprocess_image(path) for path in image_path ]

if __name__ == '__main__':
    main()
