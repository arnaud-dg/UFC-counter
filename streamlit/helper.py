from PIL import Image, ImageDraw, ImageFont
import pandas as pd

def load_image(image_file):
    """
    Load an image from a file.

    Args:
        image_file (UploadedFile): The image file to load.

    Returns:
        Image: The loaded image.
    """
    return Image.open(image_file)

def add_transparent_mask(image, transparency=0.15):
    """
    Add a white transparent mask to the image.

    Args:
        image (Image): The image to which the mask will be added.
        transparency (float): The transparency level of the mask.

    Returns:
        Image: The image with the transparent mask added.
    """
    width, height = image.size
    mask = Image.new('RGBA', (width, height), (255, 255, 255, int(255 * transparency)))
    return Image.alpha_composite(image.convert('RGBA'), mask)

def combine_images(image1, image2, split_percentage):
    """
    Combine two images by splitting them at a given percentage.

    Args:
        image1 (Image): The first image.
        image2 (Image): The second image.
        split_percentage (int): The percentage at which to split the images.

    Returns:
        Image: The combined image.
    """
    assert 0 <= split_percentage <= 100, "split_percentage must be between 0 and 100"
    
    width, height = image1.size
    split_position = int(width * split_percentage / 100)
    
    combined_image = Image.new('RGB', (width, height))
    combined_image.paste(image1.crop((0, 0, split_position, height)), (0, 0))  # Add the first part of the image
    combined_image.paste(image2.crop((split_position, 0, width, height)), (split_position, 0))  # Add the second part of the image
    return combined_image

def draw_rectangles(image, results_data, show_probabilities=False):
    """
    Draw green rectangles on the image based on prediction results.

    Args:
        image (Image): The image on which to draw.
        results_data (pd.DataFrame): The prediction results containing bounding box coordinates.
        show_probabilities (bool): Whether to display prediction probabilities.

    Returns:
        Image: The image with rectangles drawn.
    """
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    
    # Calculate base size for text and rectangles, proportional to image dimensions
    base_size = int(min(width, height) * 0.02)  # 2% of the smaller dimension
    min_size = 14
    font_size = max(base_size, min_size)
    line_width = max(3, int(base_size * 0.3))
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for _, row in results_data.iterrows():
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        draw.rectangle(bbox, outline='green', fill=(0, 255, 0, 100), width=line_width)
    
    if show_probabilities:
        for _, row in results_data.iterrows():
            confidence = f"{row['confidence'] * 100:.2f}%"
            text_position = (int(row['xmax']) + 2, int(row['ymin']))
            
            # Create a temporary image to measure text size
            temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            text_bbox = temp_draw.textbbox((0, 0), confidence, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Add padding to the text background
            padding = font_size // 2
            text_bg = [text_position[0], text_position[1], 
                       text_position[0] + text_width + padding * 2, 
                       text_position[1] + text_height + padding * 2]
            
            # Draw white background for text
            draw.rectangle(text_bg, fill='white')
            
            # Draw text in dark green
            draw.text((text_position[0] + padding, text_position[1] + padding), 
                      text=confidence, fill='darkgreen', font=font)

    return image
