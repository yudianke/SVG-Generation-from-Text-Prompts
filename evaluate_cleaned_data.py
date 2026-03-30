import pandas as pd
import numpy as np
from cairosvg import svg2png
from skimage.metrics import structural_similarity as ssim
import io
from PIL import Image
from tqdm import tqdm

def svg_to_numpy(svg_str, res=256):
    """Render SVG to grayscale numpy array for SSIM calculation."""
    try:
        png_data = svg2png(bytestring=svg_str.encode('utf-8'), output_width=res, output_height=res)
        img = Image.open(io.BytesIO(png_data)).convert('L')
        return np.array(img)
    except Exception:
        return None

def run_full_validation(original_csv, cleaned_csv):
    print(f"Loading datasets...")
    df_old = pd.read_csv(original_csv)
    df_new = pd.read_csv(cleaned_csv)
    
    if len(df_old) != len(df_new):
        print("Error: Row counts do not match between files!")
        return

    total_rows = len(df_old)
    ssim_scores = []
    old_lengths = []
    new_lengths = []
    errors = 0

    print(f"Starting validation for {total_rows} rows...")
    
    # Using tqdm for progress tracking
    for i in tqdm(range(total_rows), desc="Comparing SVGs"):
        svg_old = df_old['svg'][i]
        svg_new = df_new['svg'][i]
        
        # Track Lengths
        len_old = len(str(svg_old))
        len_new = len(str(svg_new))
        old_lengths.append(len_old)
        new_lengths.append(len_new)
        
        # Calculate Visual Fidelity (Optional: Sample every 100th for speed)
        # To run all 50k, this will take time. Remove the 'if' to run every row.
        if i % 100 == 0: 
            img_old = svg_to_numpy(svg_old)
            img_new = svg_to_numpy(svg_new)
            
            if img_old is not None and img_new is not None:
                score = ssim(img_old, img_new)
                ssim_scores.append(score)
            else:
                errors += 1

    # --- Statistics Calculation ---
    avg_len_old = np.mean(old_lengths)
    avg_len_new = np.mean(new_lengths)
    compression_rate = (1 - avg_len_new / avg_len_old) * 100
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

    # --- English Output Report ---
    print("\n" + "="*40)
    print("      DATA PREPROCESSING REPORT")
    print("="*40)
    print(f"Total Samples Processed: {total_rows}")
    print(f"Average Original Length: {avg_len_old:.2f} chars")
    print(f"Average Cleaned Length:  {avg_len_new:.2f} chars")
    print(f"Average Compression:     {compression_rate:.2f}%")
    print("-" * 40)
    print(f"Visual Fidelity (SSIM):  {avg_ssim:.6f}")
    print(f"Rendering Errors:        {errors}")
    print("-" * 40)
    
    print("\n[VERDICT]")
    if avg_ssim > 0.99 and compression_rate > 10:
        print("SUCCESS: Data size significantly reduced with negligible visual loss.")
        print("Impact on Score: Compactness (Ci) will increase, Visual Fidelity (Vi) remains stable.")
    else:
        print("WARNING: Significant visual deviation detected. Check precision settings.")
    print("="*40)

if __name__ == "__main__":
    # Ensure you have the correct filenames here
    run_full_validation("train.csv", "train_cleaned_old.csv")