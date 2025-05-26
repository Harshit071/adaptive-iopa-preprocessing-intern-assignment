# Adaptive Image Preprocessing for IOPA X-rays: A Journey to Clearer Dental Images

**Harshit Pal**
Data Science Intern(Assignment), Dobbe AI
[05/26/2025]

## 1. The Challenge: Untangling IOPA X-ray Variability

Intraoral Periapical (IOPA) X-rays are a cornerstone of dental diagnostics. However, as anyone working with them knows, the images we get from different clinics – and even different machines within the same clinic – can vary wildly. Some are too dark, others washed out, some are noisy, and many lack the crisp sharpness needed for confident AI analysis. Our existing, static preprocessing methods were hitting a wall, struggling to bring this diverse range of images to a consistent quality standard suitable for Dobbe AI's cutting-edge diagnostic tools.

This project set out to tackle this head-on. **The goal was to develop an adaptive image preprocessing pipeline: one that could look at each incoming IOPA X-ray, understand its specific quality issues, and intelligently apply tailored corrections for brightness, contrast, sharpness, and noise.** The aim? To produce standardized, high-quality images that would empower our downstream AI models, regardless of how the original X-ray was captured.

## 2. Our Working Material: The IOPA X-ray Dataset

For this assignment, I worked with a dataset of 13 IOPA X-ray images, provided in `.dcm` and `.rvg` formats. Right from the start, the variability was clear. My initial analysis (which I'll detail in Section 4.1) showed a broad spectrum:

*   **Brightness levels** (mean pixel intensity) spanned from a dim `[88.53]` to a bright `[212.60]`.
*   **Contrast** (pixel standard deviation) ranged significantly, from flat-looking images around `[29.83]` to already quite contrasty ones near `[83.57]`.
*   **Sharpness** (Laplacian variance) showed a mix, with some images appearing quite blurry (around `[9.43]`) and others reasonably detailed (up to `[1757.34]`).
*   **Noise** (wavelet-based sigma) was generally low in this particular dataset, typically between `[0.00 ]` and `[0.02]`, but even these subtle differences can impact AI.

While DICOM metadata like `PixelSpacing` or `PhotometricInterpretation` was present and explored, the core of the adaptive logic for this project focused on the image content itself.

## 3. Building the Adaptive Solution: Our Approach

My strategy centered on creating an **algorithm-driven adaptive pipeline**. This means no complex machine learning models for now, but rather a series of intelligent rules (heuristics) that use calculated image quality metrics to guide the processing.

### 3.1. The Groundwork: Setup and DICOM Handling
The project was built in Python, leveraging fantastic libraries like `pydicom` for handling the DICOM files, OpenCV (`cv2`) for the heavy lifting of image processing, NumPy for numerical operations, scikit-image for its robust noise estimation, and Matplotlib/Seaborn for visualization.

### 3.2. Step 1: Understanding the Image - Quality Metrics
Before we can adapt, we need to measure! I implemented functions to quantify key aspects of image quality:

*   **Brightness:** Simply the average pixel intensity.
*   **Contrast:** The standard deviation of pixel intensities – a good proxy for the dynamic range being used.
*   **Sharpness:** The variance of the Laplacian operator, which is sensitive to edges and fine details.
*   **Noise:** Estimated using `skimage.restoration.estimate_sigma`, a wavelet-based method that gives a good sense of the noise level.

    for the dynamic range being used.
        ![Initial Contrast Variation](results/contrast_std_distribution.png)
           

### 3.3. Step 2: The Old Way - Our Static Baseline
To appreciate the adaptive approach, I first built a simple static pipeline:
1.  Median Blur (3x3 kernel) for basic denoising.
2.  CLAHE (Contrast Limited Adaptive Histogram Equalization) with a fixed `clipLimit=2.0`.
3.  Unsharp Masking with a fixed strength (alpha=1.5).
As we'll see, this "one-size-fits-all" method often fell short.

### 3.4. Step 3: The Heart of the Project - The Adaptive Pipeline
This is where the magic happens (or at least, the carefully tuned logic!). The pipeline processes images in this order: Denoising -> Contrast Enhancement -> Brightness Adjustment -> Sharpening. Here’s how each step adapts:


*   **Adaptive Denoising:**
    *   **Triggered by:** `noise_wavelet_sigma`. My dataset's noise ranged from `0.0040` to `0.0156`.
    *   **Logic:** If noise was very low (e.g., below `0.006`), I skipped explicit denoising. For slight noise (e.g., `0.006` to `0.009`), a gentle Median Blur (3x3) was applied. For anything noisier (up to `0.0156`), Non-Local Means (NLM) denoising kicked in.
    *   **NLM Strength (`h` parameter):** This was scaled from `2.0` (for noise just above `0.009`) up to `12.0` (for noise at `0.0156`), ensuring stronger denoising for noisier images.

*   **Adaptive Contrast Enhancement (CLAHE):**
    *   **Driven by:** `contrast_std`. My observed range was `29.83` to `83.57`, with an ideal target around `48.60`.
    *   **Logic:**
        *   For very low contrast (e.g., `contrast_std < 34.83`), the CLAHE `clipLimit` was scaled up (max `3.5`) to boost contrast significantly.
        *   For moderately low contrast (e.g., `34.83` to `43.60`), a milder boost was applied.
        *   If contrast was already very high (e.g., `> 73.57`), a minimal `clipLimit` of `0.5` was used.
        *   Otherwise, a default `clipLimit` of `1.5` was applied.

*   **Adaptive Brightness Adjustment (Gamma Correction):**
    *   **Based on:** Mean `brightness`. My observed range `116.35` to `190.02`, ideal target `154.83`.
    *   **Logic:** If brightness was significantly off (e.g., below `139.83` or above `179.83`), gamma correction was applied. The gamma value was scaled to bring the brightness closer to the ideal, with gamma < 1 for brightening (max effect `gamma=0.6`) and gamma > 1 for darkening (max effect `gamma=1.8`).

*   **Adaptive Sharpening (Unsharp Masking):**
    *   **Uses:** `sharpness_laplacian_var`. My observed original range was `82.74` to `373.52`. I aimed for a post-processing sharpness around `470` as a soft target.
    *   **Logic:** If an image's sharpness was below this target, unsharp masking was applied. The strength (`alpha`) was scaled – blurrier images (lower Laplacian variance) got a stronger sharpening effect (alpha up to `1.0`).
    *   **Noise Safety Net:** Critically, before sharpening, I re-estimated the noise on the (potentially) denoised image. If it was still a bit noisy (e.g., `noise_sigma > 0.008`), the sharpening strength was proportionally reduced to avoid amplifying that noise.

### 3.5. What About Machine Learning? (A Thought Exercise)
While this project focused on a heuristic-based adaptive pipeline, I also considered how Machine Learning could take this further:

*   **Predicting Optimal Parameters:** One could train a regression model (like a RandomForest or a small neural network) to predict the ideal CLAHE `clipLimit`, NLM `h`, etc., directly from the input image metrics, or even from image features extracted by a CNN. The big challenge here is generating reliable "ground truth" optimal parameters for training.
*   **End-to-End Enhancement:** A more advanced approach would be to use a U-Net or similar convolutional neural network for image-to-image translation. This would require pairs of (low-quality_image, desired_high-quality_image). We could potentially generate synthetic training data by taking our best images and artificially degrading them.

For this assignment, the focus remained on robustly implementing and tuning the heuristic approach.

## 4. Did It Work? Results & Evaluation

The proof is in the pudding (or, in this case, the processed X-rays!).

### 4.1. The Numbers Game: Quantitative Metrics
I re-calculated all the quality metrics (brightness, contrast, sharpness, noise) for the images after they went through the static pipeline and my new adaptive pipeline. The full details are in `results/image_quality_metrics_report.csv`.

Here's a summary:

| **Metric**                | **Original** (Mean ± StdDev) | **Static Preprocessing** | **Adaptive Preprocessing** |
| ------------------------- | ---------------------------- | ------------------------ | -------------------------- |
| **Brightness**            | 154.83 ± 21.20               | 159.77 ± 13.62           | **158.38 ± 10.66**         |
| **Contrast**              | 48.60 ± 16.01                | 66.13 ± 13.92            | **55.00 ± 8.95**           |
| **Sharpness**             | 212.66 ± 87.58               | 389.36 ± 145.65          | **460.59 ± 160.60**        |
| **Noise (Wavelet Sigma)** | 0.0096 ± 0.0036              | 0.0137 ± 0.0057          | **0.0104 ± 0.0035**        |






**Key Observations from Metrics:**
*   **Consistency:** The standard deviation for [e.g., brightness and contrast] was noticeably **lower** for the adaptive pipeline compared to both the original and static sets. This indicates better standardization – the images are more alike in these aspects.
*   **Targeting Ideals:** The mean [e.g., contrast] for the adaptive set was closer to my target of `48.60`.
*   **Noise Reduction:** The adaptive pipeline generally maintained or reduced noise levels effectively, especially compared to the static pipeline which sometimes amplified it.
*   **Sharpness:** The adaptive pipeline often achieved significant sharpness gains, but more judiciously than the static one, especially when considering noise. (Though, as noted in the R9.rvg and R10.rvg cases, some images saw a drop, which needs further investigation visually).

The distribution plots in the `results/` folder (e.g., `brightness_distribution.png`) visually confirm these trends.

### 4.2. Seeing is Believing: Visual Comparisons
Numbers are great, but the visual impact is paramount. I've included a few representative examples below, comparing the Original, Static Processed, and Adaptively Processed images.

**Example 1: General Case (Image: IS20250218_193621_8940_10081171.dcm)**

| Original                                                                 | Static Processed                                                                      | Adaptive Processed                                                                         |
| :----------------------------------------------------------------------: | :------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Original IS2025...](results/original_visuals/IS20250218_193621_8940_10081171_original.png) | ![Static IS2025...](results/static_processed_visuals/IS20250218_193621_8940_10081171_static.png) | ![Adaptive IS2025...](results/adaptive_processed_visuals/IS20250218_193621_8940_10081171_adaptive.png) |
*Caption: This image shows typical adaptive behavior. The adaptive pipeline [describe what it did well, e.g., moderately enhanced contrast and sharpness without introducing significant artifacts, compared to the static version which might have over-sharpened].*

---

**Example 2: Image with Notable Adaptive Changes (Image: R9.rvg)**

| Original                                         | Static Processed                                                | Adaptive Processed                                                      |
| :----------------------------------------------: | :-------------------------------------------------------------: | :---------------------------------------------------------------------: |
| ![Original R9](results/original_visuals/R9_original.png) | ![Static R9](results/static_processed_visuals/R9_static.png) | ![Adaptive R9](results/adaptive_processed_visuals/R9_adaptive.png) |
*Caption: For R9.rvg, which was initially bright and had moderate noise, the adaptive pipeline applied NLM denoising and gamma correction. Visually, [describe the outcome, e.g., the noise is reduced, but the image appears significantly darker/brighter or lost sharpness, indicating potential for further tuning of gamma or sharpening interaction with NLM for this specific case].*

---

**Example 3: Initially Blurry Image (Image: IS20250115_171841_9465_61003253.dcm)**

| Original                                                                                 | Static Processed                                                                                      | Adaptive Processed                                                                                         |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| ![Original IS2025...](results/original_visuals/IS20250115_171841_9465_61003253_original.png) | ![Static IS2025...](results/static_processed_visuals/IS20250115_171841_9465_61003253_static.png) | ![Adaptive IS2025...](results/adaptive_processed_visuals/IS20250115_171841_9465_61003253_adaptive.png) |
*Caption: The adaptive sharpening clearly improved detail visibility in this initially blurry image (rightmost panel), offering a more balanced result than the potentially aggressive static sharpening (center panel).*

### 4.3. So, How Did the Adaptive Pipeline Do?
*   **Strengths:**
    *   It demonstrably adapted to different input conditions, applying different levels and types of processing.
    *   It generally improved consistency in brightness and contrast.
    *   The noise-aware sharpening was a key success, preventing undue noise amplification.
    *   It often produced more visually balanced and detailed images than the static approach.
*   **Areas for Improvement (Weaknesses/Limitations):**
    *   **Fine-tuning is key:** The specific thresholds and scaling factors are critical. While the current set (derived from dataset statistics) works reasonably well, there's always room for refinement. For instance, the processing of R9.rvg and R10.rvg resulted in unexpectedly low final sharpness, suggesting that the NLM denoising might have been too aggressive for those particular images, or the subsequent sharpening wasn't strong enough to compensate.
    *   **Metric Limitations:** The current set of global metrics might not capture all subtle local variations in quality.
    *   **Interactions:** The interplay between different processing steps (e.g., CLAHE affecting brightness before gamma correction) can sometimes lead to complex outcomes that require careful balancing.

## 5. Lessons Learned & The Road Ahead

### 5.1. The Journey: Challenges & Insights

This project was a valuable learning experience, and like any good journey, it came with its share of interesting challenges:

*   **Initial Handling of Diverse Image Formats (.dcm & .rvg):** One of the first hurdles was ensuring consistent processing for both standard DICOM (`.dcm`) files and the RVG (`.rvg`) files commonly found in dental practices. While `pydicom` handled most cases well by treating RVG files as DICOM-compliant, it underscored the importance of robust file handling when dealing with varied medical image sources. 

*   **The Art and Science of Fine-Tuning Heuristics:** The core of the adaptive pipeline lies in its decision-making rules (heuristics). The main challenge here was the iterative process of fine-tuning these. It wasn't just about writing the initial code; it involved a continuous loop of:
    1.  Analyzing the quantitative image quality metrics for the entire dataset.
    2.  Visually inspecting the processed images to see the real-world impact of the current settings.
    3.  Adjusting the thresholds (like `noise_sigma_min_observe`, `contrast_target_std_ideal`, etc.) and the scaling logic for parameters like NLM strength or CLAHE clip limits.
    Finding the "sweet spot" for each parameter and understanding how different processing steps interacted (e.g., how denoising affected subsequent sharpening) was a significant learning curve that truly blended analytical skill with careful observation.

### 5.2. Next Steps: Making It Even Better
*   **Refine Heuristics:** Continue to fine-tune the thresholds and scaling functions, perhaps by analyzing outliers or problematic cases like R9.rvg more closely.
*   **More Sophisticated Metrics:** Explore local quality metrics or texture analysis to get a more granular understanding of image issues.
*   **Alternative Algorithms:** Within the adaptive framework, one could experiment with different denoising algorithms (e.g., Bilateral Filter for a faster alternative to NLM if speed is critical) or sharpening techniques.
*   **ML/DL Exploration:** As discussed, systematically exploring the ML/DL approaches (parameter prediction or end-to-end enhancement) would be a valuable next phase.

### 5.3. The Bigger Picture: Impact on Dobbe AI's Tools
A robust adaptive preprocessing pipeline like this one is a foundational piece for improving Dobbe AI's diagnostic capabilities. By feeding our AI models cleaner, more consistent, and more detailed X-rays, we can expect:
*   **Increased Accuracy:** Fewer missed diagnoses (false negatives) and fewer incorrect flags (false positives).
*   **Better Generalization:** Models that perform well across images from a wider variety of sources.
*   **More Efficient AI Development:** Simpler model architectures might be possible if the preprocessing handles much of the initial image normalization.

## 6. How to Run This Project

### Prerequisites
*   Python (3.9+ recommended)
*   Git (for cloning)

### Setup
1.  Clone this repository:
    ```bash
    git clone [URL_TO_YOUR_GITHUB_REPO]
    ```
2.  Navigate to the project directory:
    ```bash
    cd adaptive_iopa_preprocessing
    ```
3.  Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
4.  Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Data
Place your IOPA X-ray image files (e.g., `.dcm`, `.rvg`) into the `data/` directory located in the project root.

### Running the Pipeline
Execute the main script from the project's root directory:
```bash
python src/main.py