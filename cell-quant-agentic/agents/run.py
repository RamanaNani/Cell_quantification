# Example usage
from matplotlib import pyplot as plt
vsi_path = '/Users/ramanareddy/Documents/Fraser Sim/295786 slide 7_current_ImageID-22867.vsi'
from file_handler import VSIFileHandler  # Import the class from the file_handler module    

handler = VSIFileHandler()

try:
    metadata = handler.load_vsi(vsi_path)
    print(f"Loaded VSI file with metadata: {metadata}")
    
    # Get the image array (optional)
    image = handler.get_image()
    print(f"Image shape: {image.shape}")
    
    # Display the image using Matplotlib
    plt.imshow(image, cmap='gray')  # Display the image in grayscale
    plt.axis('off')  # Hide axis
    plt.show()  # Show the image
    
finally:
    handler.close()
